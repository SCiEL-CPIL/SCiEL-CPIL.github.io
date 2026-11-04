"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random
import itertools
from collections import defaultdict
from scipy.optimize import minimize

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor.
    T[a, b, c] where a indexes (i,j) for A, b indexes (k,l) for B, c indexes (r = l*n+i) for C
    T[a,b,c] = 1 if j==k else 0
    """
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == k:
                        a = i * m + j
                        b = k * m + l
                        c = l * n + i
                        tensor[a, b, c] = 1
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    recon = np.zeros_like(tensor, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.max(np.abs(recon - tensor)) < tol

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_pair_merge(n, m, p, deadline):
    """Try to find lower rank by merging pairs of schoolbook terms."""
    tensor = _build_tensor(n, m, p)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Schoolbook terms indexed by (i, j, l) where i in [0,n), j in [0,m), l in [0,p)
    # Term (i,j,l): U[i*m+j]=1, V[j*p+l]=1, W[l*n+i]=1
    # Total schoolbook rank = n*m*p = 50
    
    # Try to find a decomposition with rank < 50
    # Strategy: try various structured decompositions
    
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Approach 1: Try to use the structure that n=2
    # For each j (column of A / row of B), we have n*p = 2*5 = 10 terms
    # Maybe we can do better than 10 for each j
    
    # Approach 2: Try random search with better initialization
    # Use the fact that valid decompositions have specific structure
    
    # Let's try a more systematic approach: 
    # For rank r, we need to find U (n_m x r), V (m_p x r), W (p_n x r)
    # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    
    # This is equivalent to: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Let's try to find a solution using iterative refinement
    
    # First, try small improvements: rank 49, 48, ...
    for target_rank in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        found = _search_rank(n, m, p, target_rank, tensor, deadline, max_iters=200)
        if found is not None:
            best_rank = target_rank
            best_result = found
            # Continue searching for even lower rank
            break
    
    return best_rank, best_result

def _search_rank(n, m, p, target_rank, tensor, deadline, max_iters=1000):
    """Search for a rank-r decomposition using iterative methods."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Strategy: use the structure of the problem
    # For each output element c = l*n + i, we need:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b
    
    # Let's try a different approach: fix W and solve for U,V or vice versa
    
    # Actually, let's try a more direct approach:
    # The tensor T has a specific structure. Let's exploit it.
    
    # T[i*m+j, j*p+l, l*n+i] = 1 for all i,j,l
    # T[a,b,c] = 0 otherwise
    
    # This means: for each (i,j,l), there's a 1 at position (i*m+j, j*p+l, l*n+i)
    
    # A rank-1 tensor is U ⊗ V ⊗ W where U is a vector, etc.
    # The sum of rank-1 tensors must equal T.
    
    # Let's try: for each "fiber" of the tensor, we can derive constraints.
    
    # Alternative: use the fact that this is a bilinear map
    # C = A * B where A is 2x5, B is 5x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A rank-r decomposition means: C[i,k] = sum_{r=1}^r L_r(A) * M_r(B)
    # where L_r is linear in A, M_r is linear in B
    
    # L_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # M_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    # And the coefficient of C[i,k] from term r is W[k*n+i, r]
    
    # So: sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''])
    # must equal sum_j A[i,j] * B[j,k]
    
    # This gives us the constraint on the coefficients.
    
    # Let's try to find a solution by treating this as a system of equations.
    
    # For simplicity, let's try to find a decomposition where each term uses only ±1, 0 values.
    
    # Generate random sparse matrices and check
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate sparse U, V, W with values in {-1, 0, 1}
        U = np.zeros((n_m, target_rank), dtype=np.float64)
        V = np.zeros((m_p, target_rank), dtype=np.float64)
        W = np.zeros((p_n, target_rank), dtype=np.float64)
        
        # Each column of U, V, W should have a specific structure
        # For term r, let's say it "covers" a subset of the multiplication
        
        for r in range(target_rank):
            # Randomly choose which (i,j) pairs are in U[:,r]
            # and which (j,k) pairs are in V[:,r], and which (k,i) in W[:,r]
            
            # To make it work, the term should contribute to specific (i,j,k) triples
            # Let's try: for each r, pick a random (i0, j0, k0) and set
            # U[i0*m+j0, r] = 1, V[j0*p+k0, r] = 1, W[k0*n+i0, r] = 1
            # But this is just schoolbook. We need to combine terms.
            
            # Try: for each r, pick 2-3 (i,j,k) triples and set the corresponding entries
            num_terms = random.randint(1, 3)
            for _ in range(num_terms):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                sign = random.choice([-1, 1])
                U[i*m+j, r] += sign
                V[j*p+k, r] += sign
                W[k*n+i, r] += sign
        
        # Check if this gives the right tensor
        recon = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        err = np.max(np.abs(recon - tensor))
        if err < 1e-9:
            # Found it! Convert to fractions
            U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U]
            V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V]
            W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W]
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _structured_search(n, m, p, deadline):
    """Try structured approaches based on known results."""
    tensor = _build_tensor(n, m, p)
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # For 2x5x5, let's try to use the fact that we can split the 5 into 2+3
    # and use known results for 2x2x2 and 2x3x3 etc.
    
    # Actually, let's try a different structured approach:
    # The multiplication 2x5 * 5x5 -> 2x5 can be seen as:
    # For each of the 5 columns j of A (and rows of B), we compute contributions to all 10 elements of C
    # C[i,k] += A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2,3,4}
    
    # So for each j, we need to compute 2*5 = 10 products A[i,j]*B[j,k]
    # and add them to the appropriate C[i,k]
    
    # The schoolbook uses 10 terms per j (one for each (i,k) pair)
    # Can we do better? For fixed j, the sub-problem is:
    # Given a 2x1 vector A[:,j] and a 1x5 vector B[j,:], compute their outer product (2x5)
    # This is a 2x1 * 1x5 -> 2x5 multiplication, which has rank 10 (schoolbook)
    # But this is just an outer product, so rank 10 is minimal for this sub-problem
    
    # However, we can share structure across different j values!
    
    # Let's try: group the 5 j values and see if we can save
    # For example, if we combine j=0 and j=1:
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is like multiplying a 2x2 matrix [A[:,0], A[:,1]] by a 2x5 matrix [B[0,:], B[1,:]]
    # Wait, that's not quite right.
    
    # Actually: sum_{j=0}^{1} A[i,j]*B[j,k] = (A[:,0:2] * B[0:2,:])[i,k]
    # So we're doing a 2x2 * 2x5 -> 2x5 multiplication for the first two j's
    # Similarly for j=2,3,4: 2x3 * 3x5 -> 2x5
    
    # Rank of 2x2 * 2x5: schoolbook is 2*2*5 = 20
    # Rank of 2x3 * 3x5: schoolbook is 2*3*5 = 30
    # Total: 20 + 30 = 50 (same as schoolbook)
    
    # Can we do 2x2 * 2x5 with rank < 20?
    # 2x2 * 2x2 has rank 7 (Strassen). For 2x2 * 2x5, we can split the 5 into 2+3:
    # 2x2 * 2x2 (rank 7) + 2x2 * 2x3 (rank ?)
    # 2x2 * 2x3 schoolbook is 12. Can we do better?
    
    # This is getting complex. Let's just try random search more aggressively.
    
    # Try various ranks with more iterations
    for target_rank in [45, 40, 35, 30, 28, 27, 26, 25]:
        if time.time() > deadline:
            break
        
        found = _search_rank(n, m, p, target_rank, tensor, deadline, max_iters=500)
        if found is not None:
            best_rank = target_rank
            best_result = found
            break
    
    return best_rank, best_result

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try pair merge approach
    if time.time() < deadline - 5:
        new_rank, new_result = _try_pair_merge(n, m, p, deadline - 5)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    # If still at schoolbook, try structured search
    if best_rank == n * m * p and time.time() < deadline - 3:
        new_rank, new_result = _structured_search(n, m, p, deadline - 3)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    return best_result

# EVOLVE-BLOCK-END

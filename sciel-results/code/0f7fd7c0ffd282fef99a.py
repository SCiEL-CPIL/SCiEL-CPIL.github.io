"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random
import itertools
from collections import defaultdict

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

def _try_structured_merge(n, m, p, deadline):
    """Try to find a lower-rank decomposition by merging schoolbook terms using Strassen-like identities."""
    tensor = _build_tensor(n, m, p)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Schoolbook terms: each term r has (i,j) in U, (j,l) in V, (l,i) in W
    # Term index: r = i*m*p + j*p + l where i in [0,n), j in [0,m), l in [0,p)
    # Wait, let me re-index: for term (i,j,l): U[i*m+j]=1, V[j*p+l]=1, W[l*n+i]=1
    
    # Build list of terms
    terms = []
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                terms.append((a, b, c))
    
    schoolbook_rank = len(terms)
    
    best_rank = schoolbook_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Strategy 1: Try to find a lower rank by random search
    # Strategy 2: Try structured block decompositions
    
    # For 2x5x5, schoolbook rank is 50
    # Known lower bounds: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we need at least the rank of the tensor
    
    # Try ranks from schoolbook-1 down to a reasonable lower bound
    lower_bound = max(n*m, m*p, p*n)  # = 25
    
    # Use time budget wisely
    time_per_rank = (deadline - time.time()) / (schoolbook_rank - lower_bound + 1)
    
    for target_rank in range(schoolbook_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank: more time for higher ranks (closer to schoolbook)
        iters = max(50, int(time_per_rank * 10))
        
        found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=iters)
        if found is not None:
            best_rank = target_rank
            best_result = found
            # Continue searching for even lower rank
            break
    
    # If we found something, try to go lower
    if best_rank < schoolbook_rank:
        for target_rank in range(best_rank - 1, lower_bound - 1, -1):
            if time.time() > deadline:
                break
            
            found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=100)
            if found is not None:
                best_rank = target_rank
                best_result = found
                break
    
    return best_rank, best_result

def _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=1000):
    """Random search for a rank-r decomposition with integer/half-integer entries."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Sparse random initialization with values from a larger set
    # Use sparse matrices: most entries are 0, some are ±1, ±1/2
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate sparse random U, V, W
        # Each row has 1-3 non-zero entries
        U = np.zeros((n_m, target_rank), dtype=np.float64)
        V = np.zeros((m_p, target_rank), dtype=np.float64)
        W = np.zeros((p_n, target_rank), dtype=np.float64)
        
        for a in range(n_m):
            nnz = random.randint(1, 3)
            cols = random.sample(range(target_rank), min(nnz, target_rank))
            for c in cols:
                U[a, c] = random.choice([1, -1, 0.5, -0.5, 2, -2])
        
        for b in range(m_p):
            nnz = random.randint(1, 3)
            cols = random.sample(range(target_rank), min(nnz, target_rank))
            for c in cols:
                V[b, c] = random.choice([1, -1, 0.5, -0.5, 2, -2])
        
        for c in range(p_n):
            nnz = random.randint(1, 3)
            cols = random.sample(range(target_rank), min(nnz, target_rank))
            for cc in cols:
                W[c, cc] = random.choice([1, -1, 0.5, -0.5, 2, -2])
        
        # Quick check using matrix multiplication
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Compute efficiently: for each r, outer product
        recon = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        err = np.max(np.abs(recon - tensor))
        if err < 1e-9:
            # Found a valid decomposition - convert to fractions
            U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U]
            V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V]
            W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W]
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _structured_rank_search(n, m, p, deadline):
    """Try structured approaches to find lower rank decompositions."""
    tensor = _build_tensor(n, m, p)
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try block-based Strassen-like decompositions
    # For 2x5x5: split into blocks and use known results for smaller matrices
    
    # Known: 2x2x2 matrix mult has rank 7 (Strassen)
    # We can try to decompose 2x5x5 using blocks
    
    # Split 5 into 2+3:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 0x2, A22 is 0x3
    # Wait, n=2, so A is 2x5. Let's split the 5 columns into 2+3.
    # B is 5x5, split rows into 2+3 and cols into 2+3.
    # C is 5x2, split rows into 2+3 and cols into 1+1 (since n=2).
    
    # This is complex. Let's try a different approach:
    # Use the fact that for fixed i, the multiplication is:
    # C[i,:] = A[i,:] * B
    # So we need to compute 2 rows of C, each being a 1x5 times 5x5 = 1x5
    
    # For 1x5 * 5x5 = 1x5, the schoolbook rank is 25
    # Can we do better? The tensor is sum_{j,k} A[j]*B[k,j]*C[k]
    # This is a bilinear map from (1x5, 5x5) to 1x5
    
    # Try random search for the 1x5*5x5 case embedded in the full problem
    # Actually, let's just do more intensive random search on the full problem
    
    # Intensive random search with better initialization
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Try various ranks
    for target_rank in [45, 40, 35, 30]:
        if time.time() > deadline:
            break
        
        found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=500)
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
    
    # Try structured merge with time budget
    if time.time() < deadline - 5:
        new_rank, new_result = _try_structured_merge(n, m, p, deadline - 5)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    # If still at schoolbook, try intensive random search
    if best_rank == n * m * p and time.time() < deadline - 3:
        tensor = _build_tensor(n, m, p)
        # Try a few specific ranks
        for target_rank in [49, 48, 47, 46, 45]:
            if time.time() > deadline - 3:
                break
            found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline - 3, max_iters=300)
            if found is not None:
                best_rank = target_rank
                best_result = found
                break
    
    return best_result

# EVOLVE-BLOCK-END

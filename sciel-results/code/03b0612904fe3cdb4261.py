import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]
    # A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor index: (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r] follows trace convention
    
    # The tensor T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    
    # We need U @ V^T @ W^T to reconstruct T
    # Specifically: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    # Start with schoolbook (rank 36)
    rank_schoolbook = n * m * p
    
    # Try to find lower rank
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify function - check exact decomposition
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        # Check all 36 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check ALL zero entries for correctness
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    # Verify schoolbook
    if not verify(best_U, best_V, best_W):
        print("Schoolbook verification failed!")
    
    # Strategy: Use structured Kronecker/permutation approach
    # For M_{n,m,p}, we can exploit the tensor structure.
    # Key idea: The multiplication tensor is a sum of rank-1 tensors.
    # Each term corresponds to a specific (i,j,k) triple.
    # We want to find a minimal set of rank-1 terms whose sum equals T.
    
    # Approach: Use the fact that for small dimensions, we can try
    # various structured decompositions.
    
    # Method 1: Try to use the structure where p = n*m (here 6 = 2*3)
    # This suggests a Kronecker product structure might work.
    
    # Let's try a systematic search over candidate ranks
    # using structured initializations based on permutation matrices.
    
    def try_structured_rank(r, seed_offset=0):
        """Try to find a rank-r decomposition using structured initialization."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + seed_offset)
        
        # Generate structured random matrices with sparsity
        # Use patterns that respect the tensor structure
        
        # Initialize with sparse random matrices
        density = max(0.1, 2.0/r)  # Lower density for higher rank
        
        U_r = np.zeros((n*m, r))
        V_r = np.zeros((m*p, r))
        W_r = np.zeros((p*n, r))
        
        # Fill with sparse random values
        for col in range(r):
            # Choose random non-zero positions
            num_nonzero_U = max(1, int(n*m * density))
            num_nonzero_V = max(1, int(m*p * density))
            num_nonzero_W = max(1, int(p*n * density))
            
            nz_U = np.random.choice(n*m, num_nonzero_U, replace=False)
            nz_V = np.random.choice(m*p, num_nonzero_V, replace=False)
            nz_W = np.random.choice(p*n, num_nonzero_W, replace=False)
            
            U_r[nz_U, col] = np.random.choice([-1, 1], size=num_nonzero_U)
            V_r[nz_V, col] = np.random.choice([-1, 1], size=num_nonzero_V)
            W_r[nz_W, col] = np.random.choice([-1, 1], size=num_nonzero_W)
        
        # Now try to solve for corrections using linear algebra
        # Fix U and V, solve for W
        # The constraint: for each (a,b,c):
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Build linear system for W
        num_constraints = n*m*p + 100  # 36 non-zero + 100 zero constraints
        
        # Collect zero constraints
        zero_constraints = []
        np.random.seed(12345 + seed_offset)
        all_zero_indices = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        all_zero_indices.append((a, b, c))
        
        if len(all_zero_indices) > 100:
            zero_sample = np.random.choice(len(all_zero_indices), 100, replace=False)
            zero_constraints = [all_zero_indices[idx] for idx in zero_sample]
        else:
            zero_constraints = all_zero_indices
        
        # Build system
        A_sys = np.zeros((num_constraints, p*n*r))
        b_sys = np.zeros(num_constraints)
        
        constraint_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    for r_idx in range(r):
                        A_sys[constraint_idx, c*r+r_idx] = U_r[a,r_idx]*V_r[b,r_idx]
                    b_sys[constraint_idx] = 1
                    constraint_idx += 1
        
        for (a, b, c) in zero_constraints:
            for r_idx in range(r):
                A_sys[constraint_idx, c*r+r_idx] = U_r[a,r_idx]*V_r[b,r_idx]
            b_sys[constraint_idx] = 0
            constraint_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            W_r = W_flat.reshape(p*n, r)
            
            # Verify
            if verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        except:
            pass
        
        return None
    
    # Try various structured approaches
    
    # Approach 1: Try Kronecker product structure
    # Since p = n*m = 2*3, try to exploit this
    
    # Generate candidate rank-1 terms using permutation-like structures
    def generate_rank1_terms():
        """Generate a set of candidate rank-1 terms."""
        terms = []
        
        # Basic schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    terms.append((u, v, w))
        
        # Try some structured combinations
        # For example, group terms by i or j
        for i in range(n):
            u = np.zeros(n*m)
            for j in range(m):
                u[i*m+j] = 1
            
            for k in range(p):
                v = np.zeros(m*p)
                for j in range(m):
                    v[j*p+k] = 1
                w = np.zeros(p*n)
                w[k*n+i] = 1
                terms.append((u, v, w))
        
        for j in range(m):
            v = np.zeros(m*p)
            for k in range(p):
                v[j*p+k] = 1
            
            for i in range(n):
                u = np.zeros(n*m)
                u[i*m+j] = 1
                w = np.zeros(p*n)
                for k in range(p):
                    w[k*n+i] = 1
                terms.append((u, v, w))
        
        return terms
    
    # Try to find minimal subset of terms that sums to T
    # This is a set cover / linear algebra problem
    
    # For now, try random combinations of terms
    terms = generate_rank1_terms()
    num_terms = len(terms)
    
    print(f"Generated {num_terms} candidate rank-1 terms")
    
    # Try to find minimal rank by trying subsets
    # Start from low rank and increase
    
    for r in range(12, best_rank):
        if time.time() > deadline:
            break
        
        # Try random subsets of terms of size r
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 100 + r)
            
            # Choose r random terms
            indices = np.random.choice(num_terms, r, replace=False)
            
            U_r = np.zeros((n*m, r))
            V_r = np.zeros((m*p, r))
            W_r = np.zeros((p*n, r))
            
            for idx, term_idx in enumerate(indices):
                u, v, w = terms[term_idx]
                U_r[:, idx] = u
                V_r[:, idx] = v
                W_r[:, idx] = w
            
            # Check if this combination works
            if verify(U_r, V_r, W_r):
                best_rank = r
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                print(f"Found rank {r} decomposition!")
                found = True
                break
        
        if found:
            break
    
    # If no structured solution found, try the linear algebra approach
    if best_rank == rank_schoolbook:
        for r in range(15, best_rank):
            if time.time() > deadline:
                break
            
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                result = try_structured_rank(r, seed_offset=attempt)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    print(f"Found rank {r} decomposition via linear algebra!")
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    # Final verification
    if verify(best_U, best_V, best_W):
        print(f"Final rank: {best_rank}")
    else:
        print("Warning: Final verification failed, using schoolbook")
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank_schoolbook
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

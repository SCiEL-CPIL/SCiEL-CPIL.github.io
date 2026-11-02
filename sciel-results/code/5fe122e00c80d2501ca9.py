import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook decomposition (rank 36) as fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify that U, V, W exactly decompose the 2x3*3x6 matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if and only if there exists r such that U[i,r]*V[j,r]*W[k,r] = 1.
    Indexing: i in [0, n*m), j in [0, m*p), k in [0, p*n)
    T[i,j,k] = 1 iff (i//m == k%n) and (i%m == j//p)
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the decomposed tensor
    # T_decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # We need T_decomp[i,j,k] == 1 iff i//m == k%n and i%m == j//p
    
    # Efficient check: for each (i,j,k), compute the sum
    # T_decomp = einsum('ir,jr,kr->ijk', U, V, W)
    T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ii = i // m
        ij = i % m
        for j in range(m*p):
            jj = j // p
            if jj == ij:
                for k in range(p*n):
                    kk = k % n
                    if kk == ii:
                        T_expected[i, j, k] = 1.0
    
    return np.allclose(T_decomp, T_expected, atol=1e-10)

def _try_rank_patterns(rank, deadline, n=2, m=3, p=6):
    """
    Try to find a valid decomposition of given rank by searching over sparse patterns.
    Uses the structure: for each rank-1 term r, U[i,r] is nonzero only for specific i,
    V[j,r] for specific j, W[k,r] for specific k.
    
    Strategy: For each column r, the support in U determines which (i//m, i%m) pairs are used.
    We search over assignments of supports.
    """
    start = time.time()
    
    # For a rank-r decomposition of [n,m,p] matmul:
    # Each term r has:
    #   - a set S_U(r) of rows in U (subset of [0, n*m))
    #   - a set S_V(r) of rows in V (subset of [0, m*p))
    #   - a set S_W(r) of rows in W (subset of [0, p*n))
    #   - scalar values u_r, v_r, w_r (typically ±1 or 0)
    #
    # The constraint: for all (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = 1 iff i//m==k%n and i%m==j//p
    #
    # For simplicity, assume U[i,r], V[j,r], W[k,r] ∈ {0, 1, -1}
    #
    # Key insight: For each i, the set of r where U[i,r]≠0 must be such that
    # the contributions cover exactly the right (j,k) pairs.
    
    # Let's try a structured approach:
    # For the 2x3x6 case, think of it as two 2x3x3 multiplications sharing the A matrix.
    # A is 2x3, B1 is 3x3 (columns 0-2 of B), B2 is 3x3 (columns 3-5 of B).
    # C1 = A*B1 (2x3), C2 = A*B2 (2x3), C = [C1 C2] (2x6)
    
    # For 2x3x3 matmul, schoolbook is rank 18. But we can do better.
    # Actually for 2x3x3, the rank is at most 18 (schoolbook) and at least... 
    # Let me just try to find a good decomposition by search.
    
    # Alternative: use the fact that 2x3x6 = (2x3x3) ⊕ (2x3x3) with shared A.
    # If we have a rank-R decomposition for 2x3x3, we can use it twice for the two blocks.
    # But the A matrix is shared, so we can potentially save.
    
    # Let's try: for each of the 6 columns of B (3 for B1, 3 for B2),
    # we need to compute A * B_col which is 2x3.
    
    # Actually, let me try a direct search for small ranks.
    # For rank r, we need to assign to each of the r terms:
    #   - which i indices (out of 6) have U[i,r] = ±1
    #   - which j indices (out of 18) have V[j,r] = ±1  
    #   - which k indices (out of 12) have W[k,r] = ±1
    
    # This is a huge search space. Let's use a smarter approach.
    
    # The tensor equation: for all i,j,k:
    #   sum_r U[i,r] V[j,r] W[k,r] = delta(i//m == k%n) * delta(i%m == j//p)
    
    # Fix i. Then for all j,k:
    #   sum_r U[i,r] V[j,r] W[k,r] = delta(i%m == j//p) * delta(i//m == k%n)
    
    # Let's denote for fixed i: ii=i//m, ij=i%m
    # RHS = 1 iff j//p == ij AND k%n == ii
    # So for fixed i, the RHS is a rank-1 matrix in (j,k): it's 1 on the block j in [ij*p, (ij+1)*p) and k%n==ii
    
    # So for each i, we need: sum_r U[i,r] (V[:,r] ⊗ W[:,r]) = E_i
    # where E_i is the indicator matrix.
    
    # This means the vectors {V[:,r] * U[i,r]} and {W[:,r] * U[i,r]} must combine to give E_i.
    
    # Let me try a different approach: assume the decomposition has a block structure.
    # For the 2x3x6 case, try rank = 2 * rank(2x3x3) or less.
    
    # Let's just do a randomized search for low-rank decompositions.
    best_result = None
    
    for r in range(max(1, rank - 10), min(rank + 1, 40)):
        if time.time() > deadline:
            break
        
        # Try random sparse patterns
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with sparsity
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # For each column r, assign random ±1 values to a subset of entries
            for col in range(r):
                # U column: pick ~2-3 entries
                n_u = np.random.randint(1, 4)
                idx_u = np.random.choice(n*m, n_u, replace=False)
                U[idx_u, col] = np.random.choice([-1, 1], n_u)
                
                # V column: pick ~3-5 entries  
                n_v = np.random.randint(2, 6)
                idx_v = np.random.choice(m*p, n_v, replace=False)
                V[idx_v, col] = np.random.choice([-1, 1], n_v)
                
                # W column: pick ~2-4 entries
                n_w = np.random.randint(1, 5)
                idx_w = np.random.choice(p*n, n_w, replace=False)
                W[idx_w, col] = np.random.choice([-1, 1], n_w)
            
            # Verify
            T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_expected = np.zeros((n*m, m*p, p*n))
            for i in range(n*m):
                ii = i // m
                ij = i % m
                for j in range(m*p):
                    if j // p == ij:
                        for k in range(p*n):
                            if k % n == ii:
                                T_expected[i, j, k] = 1.0
            
            if np.allclose(T_decomp, T_expected, atol=1e-10):
                # Found a valid decomposition!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Convert to integers if close
                for i in range(n*m):
                    for j in range(r):
                        if abs(U_list[i][j] - round(U_list[i][j])) < 1e-9:
                            U_list[i][j] = int(round(U_list[i][j]))
                for i in range(m*p):
                    for j in range(r):
                        if abs(V_list[i][j] - round(V_list[i][j])) < 1e-9:
                            V_list[i][j] = int(round(V_list[i][j]))
                for i in range(p*n):
                    for j in range(r):
                        if abs(W_list[i][j] - round(W_list[i][j])) < 1e-9:
                            W_list[i][j] = int(round(W_list[i][j]))
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _structured_search(deadline):
    """
    Structured search exploiting the block structure of 2x3x6 matmul.
    View B as [B1; B2] where B1, B2 are 3x3.
    C = A * [B1; B2] = [A*B1, A*B2]
    
    Try to find a shared decomposition for both A*B1 and A*B2.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    p_half = p // 2  # 3
    
    start = time.time()
    
    # For 2x3x3 matmul, let's try to find a rank-R decomposition.
    # Then use it for both halves, potentially sharing some terms.
    
    # First, try to find a good 2x3x3 decomposition.
    # Schoolbook for 2x3x3 is rank 18.
    # Can we do better?
    
    # Let's try rank 12 for 2x3x3 (optimistic).
    # Actually, let me try a specific structure.
    
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try a decomposition where each term computes a partial product.
    # Term r: U[i,r] = A[i, j_r], V[j,r] = B[j_r, k_r], W[k,r] = 1
    # This is schoolbook with rank = 2*3*3 = 18.
    
    # To reduce rank, we need to combine terms.
    # E.g., if two terms have the same j_r and k_r, we can combine their U vectors.
    
    # Let me try a different approach: use the known result that rank(2x3x3) <= 18
    # and see if we can find something better by search.
    
    # For now, let's construct the 2x3x3 schoolbook and use it twice.
    # That gives rank 36 total, same as schoolbook for 2x3x6.
    
    # To improve, we need to share terms between the two halves.
    
    # Idea: For column k of C (0<=k<6), C[:,k] = A * B[:,k]
    # If k < 3: C[:,k] = A * B1[:,k]
    # If k >= 3: C[:,k] = A * B2[:,k-3]
    
    # Let me try a rank-24 construction:
    # For each of the 6 columns of B, and for each of the 3 rows of A (well, 2 rows),
    # ... this is getting complicated.
    
    # Let me just do a more careful random search with better initialization.
    
    best = None
    best_rank = n * m * p  # 36
    
    # Try ranks from 35 down to 20
    for r in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Initialize with structure: each term should contribute to a "block"
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Assign each term to a random (i_block, j_block, k_block)
            # i_block in [0, n), j_block in [0, m), k_block in [0, p)
            for col in range(r):
                ii = np.random.randint(n)
                jj = np.random.randint(m)
                kk = np.random.randint(p)
                
                # U: nonzero for i where i//m == ii, i%m == jj
                i_idx = ii * m + jj
                U[i_idx, col] = np.random.choice([-1, 1])
                
                # V: nonzero for j where j//p == jj, j%p in some subset
                for dp in range(p):
                    if np.random.random() < 0.5:
                        j_idx = jj * p + dp
                        V[j_idx, col] = np.random.choice([-1, 1])
                
                # W: nonzero for k where k%n == ii, k//n in some subset
                for dn in range(n):
                    if np.random.random() < 0.5:
                        k_idx = kk * n + ii
                        if k_idx < p * n:
                            W[k_idx, col] = np.random.choice([-1, 1])
            
            # Verify
            T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_expected = np.zeros((n*m, m*p, p*n))
            for i in range(n*m):
                ii = i // m
                ij = i % m
                for j in range(m*p):
                    if j // p == ij:
                        for k in range(p*n):
                            if k % n == ii:
                                T_expected[i, j, k] = 1.0
            
            if np.allclose(T_decomp, T_expected, atol=1e-10):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                for i in range(n*m):
                    for j in range(r):
                        if abs(U_list[i][j] - round(U_list[i][j])) < 1e-9:
                            U_list[i][j] = int(round(U_list[i][j]))
                for i in range(m*p):
                    for j in range(r):
                        if abs(V_list[i][j] - round(V_list[i][j])) < 1e-9:
                            V_list[i][j] = int(round(V_list[i][j]))
                for i in range(p*n):
                    for j in range(r):
                        if abs(W_list[i][j] - round(W_list[i][j])) < 1e-9:
                            W_list[i][j] = int(round(W_list[i][j]))
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    """
    start = time.time()
    deadline = start + 110  # Leave 10s margin for the 120s limit
    
    # First, try structured search
    result = _structured_search(deadline)
    if result is not None:
        return result
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

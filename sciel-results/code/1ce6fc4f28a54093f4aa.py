import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Use Kronecker product structure.
    The 2x2x12 multiplication can be viewed as combining:
    - 2x2x2 matrix multiplication (rank 7 via Strassen)
    - 1x6 vector scaling (rank 6)
    
    Total rank = 7 * 6 = 42
    
    The tensor T_{i,j,k} = sum_l A_{i,l} B_{l,k}
    can be decomposed using the Kronecker product of the two smaller tensors.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    t_start = time.time()
    deadline = t_start + 115  # safety margin before 120s limit
    
    # Strassen's rank-7 decomposition for 2x2x2 matrix multiplication
    # U indices: (i, l) for i in {0,1}, l in {0,1} -> 4 values
    # V indices: (l, k) for l in {0,1}, k in {0,1} -> 4 values  
    # W indices: (k, i) for k in {0,1}, i in {0,1} -> 4 values
    # 
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Rank-7 decomposition for 2x2x2
    U_222 = np.array([
        [1, 1, 1, 0, 1, 0, 1],  # A00
        [0, 0, 0, 0, 1, 1, 0],  # A01
        [1, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    V_222 = np.array([
        [1, 0, 1, 1, 0, 1, 1],  # B00
        [1, 0, -1, -1, 1, 1, 1], # B01
        [1, 1, 0, 0, 1, 0, 0],  # B10
        [1, 1, -1, 0, 0, 0, 1]  # B11
    ], dtype=int)
    
    W_222 = np.array([
        [1, 0, 0, 0, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    # Verify 2x2x2 decomposition
    T_222 = np.einsum('ic,jc,kc->ijk', U_222, V_222, W_222)
    T_222_target = np.ones((2, 2, 2))
    
    if not np.allclose(T_222, T_222_target, atol=1e-9):
        # Fallback to naive rank-4
        U_222 = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=int)
        V_222 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=int)
        W_222 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=int)
    
    # Rank-6 decomposition for 1x6 vector scaling
    # This is just the identity: sum_{r=0}^{5} x_r * e_r
    # U_1x6[i, r] = delta(i, r) for i in {0}, r in {0,...,5}
    # V_1x6[j, r] = delta(j, r) for j in {0,...,5}, r in {0,...,5}
    # W_1x6[k, r] = delta(k, r) for k in {0,...,5}, r in {0,...,5}
    #
    # Actually, for 1x6, we need:
    # (1x6) * (6x1) = 1x1, which is just a dot product
    # But we're doing 2x2 * 2x12, so we need to think differently.
    #
    # The 2x12 matrix has 2 rows and 12 columns.
    # We can view it as 2 copies of a 1x6 vector (each row has 6 elements per "block")
    # Actually, let's group the 12 columns into 2 groups of 6.
    # Then 2x2 * 2x12 = (2x2 * 2x6) for each group of 6 columns.
    #
    # So we need 2x2x6 multiplication, which is 2x2x2 * (1x3 scaling)
    # Rank = 7 * 3 = 21 per group, total = 42
    
    # Let's use a simpler approach: direct Kronecker product
    # 2x2x12 = (2x2x2) ⊗ (1x6)
    # where 1x6 is the "scaling" part
    
    # Actually, let me reconsider. The current approach with rank 36 might be better.
    # Let me verify if the current rank-36 decomposition is correct.
    
    # For now, let's stick with the Strassen-based approach but optimize it
    # Group 12 columns into 6 pairs, use rank-7 Strassen for each pair
    # Total rank = 6 * 7 = 42
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=int)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=int)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)   # 24 x 42
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            # Use Strassen's decomposition for each pair of columns
            if s == 0:
                # M1 = (A00+A11)(B00+B11)
                U[0, c] = 1
                U[3, c] = 1
                V[0 * p + 2*t, c] = 1
                V[0 * p + 2*t+1, c] = 1
                V[1 * p + 2*t, c] = 1
                V[1 * p + 2*t+1, c] = 1
                W[(2*t)*n + 0, c] = 1
                W[(2*t)*n + 1, c] = 1
                W[(2*t+1)*n + 0, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 1:
                # M2 = (A10+A11)B00
                U[2, c] = 1
                U[3, c] = 1
                V[0 * p + 2*t, c] = 1
                W[(2*t)*n + 1, c] = 1
                W[(2*t+1)*n + 1, c] = -1
            elif s == 2:
                # M3 = A00(B10-B11)
                U[0, c] = 1
                V[1 * p + 2*t, c] = 1
                V[1 * p + 2*t+1, c] = -1
                W[(2*t+1)*n + 0, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 3:
                # M4 = A11(B01-B00)
                U[3, c] = 1
                V[0 * p + 2*t+1, c] = 1
                V[0 * p + 2*t, c] = -1
                W[(2*t)*n + 0, c] = 1
                W[(2*t)*n + 1, c] = 1
            elif s == 4:
                # M5 = (A00+A01)B11
                U[0, c] = 1
                U[1, c] = 1
                V[1 * p + 2*t+1, c] = 1
                W[(2*t)*n + 0, c] = -1
                W[(2*t+1)*n + 0, c] = 1
            elif s == 5:
                # M6 = (A10-A00)(B00+B01)
                U[2, c] = 1
                U[0, c] = -1
                V[0 * p + 2*t, c] = 1
                V[0 * p + 2*t+1, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 6:
                # M7 = (A01-A11)(B10+B11)
                U[1, c] = 1
                U[3, c] = -1
                V[1 * p + 2*t, c] = 1
                V[1 * p + 2*t+1, c] = 1
                W[(2*t)*n + 0, c] = 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Target: T[i,j,k] = 1 if this is a valid multiplication tensor
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fall back to naive rank 48
            rank_naive = n * m * p  # 48
            U_n = np.zeros((n * m, rank_naive), dtype=int)
            V_n = np.zeros((m * p, rank_naive), dtype=int)
            W_n = np.zeros((p * n, rank_naive), dtype=int)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_n[i * m + j, idx] = 1
                        V_n[j * p + k, idx] = 1
                        W_n[k * n + i, idx] = 1
                        idx += 1
            
            return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

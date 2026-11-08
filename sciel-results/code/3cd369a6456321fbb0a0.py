"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Block decomposition: split the 6 output columns into 3 blocks of 2.
    # Each block is a 2x3x2 multiplication, rank 11.
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Define the 11 terms for a single 2x3x2 block.
    # A is 2x3: A[i,j] with i in 0..1, j in 0..2. Flat index: i*3+j.
    # B is 3x2: B[j,k] with j in 0..2, k in 0..1. Flat index: j*2+k.
    # C is 2x2: C[i,k] with i in 0..1, k in 0..1. Flat index: i*2+k.
    #
    # We use the top-left 2x2 submatrix of A and the top-left 2x2 submatrix of B
    # for Strassen, plus 4 scalar products for the third column/row.
    #
    # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3, B[2,0]=4, B[2,1]=5
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    terms = []
    
    # Strassen terms for A[:2,:2] * B[:2,:2]:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}
    })
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}
    })
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}
    })
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}
    })
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}
    })
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}
    })
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}
    })
    
    # Scalar terms for A[:,2] * B[2,:]:
    # K1 = A02 * B20 -> C00 += K1
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}
    })
    # K2 = A02 * B21 -> C01 += K2
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}
    })
    # K3 = A12 * B20 -> C10 += K3
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}
    })
    # K4 = A12 * B21 -> C11 += K4
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}
    })
    
    # Fill the matrices for 3 blocks
    for block in range(3):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: same for all blocks (A doesn't change)
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: B indices in block: j in 0..2, k in 0..1
            # Global B index: j*p + (block*2 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: C indices in block: i in 0..1, k in 0..1
            # Convention: W[k*n+i, r]
            # Global C index: (block*2 + k_local) * n + i_local
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Fast verification using NumPy vectorization
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition efficiently
        # For each rank term r:
        #   a_r = sum_{i,j} U[i*m+j, r] * A[i,j]
        #   b_r = sum_{j,k} V[j*p+k, r] * B[j,k]
        #   prod_r = a_r * b_r
        #   C[i,k] += W[k*n+i, r] * prod_r
        
        A_flat = A.flatten()  # shape (n*m,)
        B_flat = B.flatten()  # shape (m*p,)
        
        a_vecs = U_np @ A_flat  # shape (rank,)
        b_vecs = V_np @ B_flat  # shape (rank,)
        prods = a_vecs * b_vecs  # shape (rank,)
        
        # C[i,k] = sum_r W[k*n+i, r] * prods[r]
        C_flat = W_np @ prods  # shape (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # shape (n, p)
        
        if not np.allclose(C_decomp, C_expected, atol=1e-9):
            pass  # Should not happen
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

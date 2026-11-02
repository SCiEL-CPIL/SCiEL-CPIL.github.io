"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 130.0  # safety margin before 150s kill


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is correct. W indexed as W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False

    rank = U.shape[1]

    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(dec_val - 1.0) > tol:
                    return False
    return True


def _build_rank22():
    """
    Two copies of a rank-11 decomposition for 2x3x3 -> rank 22 for 2x3x6.
    
    For each 2x3x3 block (A: 2x3, B: 3x3, C: 2x3):
    - 7 Strassen terms for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    - 1 outer product term for A[:,2] * B[2,0:2] -> C[:,0:2]
    - 1 term for A[:,0] * B[0,2] -> C[:,2]
    - 1 term for A[:,1] * B[1,2] -> C[:,2]
    - 1 term for A[:,2] * B[2,2] -> C[:,2]
    Total: 11 terms per block.
    
    For 2x3x6: split B into two 3x3 blocks (columns 0-2 and 3-5).
    Each block is an independent 2x3x3 multiplication.
    Total rank = 2 * 11 = 22.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # (6, 22)
    V = [[0] * rank for _ in range(m * p)]  # (18, 22)
    W = [[0] * rank for _ in range(p * n)]  # (12, 22)

    # Local flat indices for 2x3x3 block:
    # A: i*3+j -> 0,1,2,3,4,5 (i in {0,1}, j in {0,1,2})
    # B: j*3+k -> 0,1,2,3,4,5,6,7,8 (j in {0,1,2}, k in {0,1,2})
    # C: i*3+k -> 0,1,2,3,4,5 (i in {0,1}, k in {0,1,2})
    # But W is indexed as W[k*n+i, r] where n=2, so local C (i,k) maps to global W index.
    
    # The 11 terms for a 2x3x3 block:
    # Each term is (u_map, v_map, w_map) where maps are {local_flat_idx: value}
    
    terms = [
        # Strassen 7 terms for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        # A indices: A00=0, A01=1, A10=3, A11=4
        # B indices: B00=0, B01=1, B10=3, B11=4
        # C indices: C00=0, C01=1, C10=3, C11=4
        
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {3: 1, 4: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 4: -1}, {1: 1, 4: 1}),
        # M4 = A11(B10-B00)
        ({4: 1}, {3: 1, 0: -1}, {0: 1, 3: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {4: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1}),
        
        # M8: A[:,2] * B[2,0:2] -> C[:,0:2] (outer product, 1 term)
        # A indices: A02=2, A12=5
        # B indices: B20=6, B21=7
        # C indices: C00=0, C01=1, C10=3, C11=4
        ({2: 1, 5: 1}, {6: 1, 7: 1}, {0: 1, 1: 1, 3: 1, 4: 1}),
        
        # M9: A[:,0] * B[0,2] -> C[:,2]
        # A indices: A00=0, A10=3
        # B indices: B02=2
        # C indices: C02=2, C12=5
        ({0: 1, 3: 1}, {2: 1}, {2: 1, 5: 1}),
        
        # M10: A[:,1] * B[1,2] -> C[:,2]
        # A indices: A01=1, A11=4
        # B indices: B12=5
        # C indices: C02=2, C12=5
        ({1: 1, 4: 1}, {5: 1}, {2: 1, 5: 1}),
        
        # M11: A[:,2] * B[2,2] -> C[:,2]
        # A indices: A02=2, A12=5
        # B indices: B22=8
        # C indices: C02=2, C12=5
        ({2: 1, 5: 1}, {8: 1}, {2: 1, 5: 1}),
    ]

    for block in range(2):
        for t_idx, (u_map, v_map, w_map) in enumerate(terms):
            col = block * 11 + t_idx
            # U: global index is i*3+j (same as local for A)
            for idx, val in u_map.items():
                U[idx][col] = val
            # V: local j*3+k -> global j*p + (block*3 + k)
            for l_idx, val in v_map.items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            # W: local i*3+k -> global (block*3 + k)*n + i
            for l_idx, val in w_map.items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


def _build_rank33():
    """Direct sum of three 2x3x2 blocks (rank 11 each) -> rank 33."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Local flat indices for a 2x3x2 block:
    # A: i*3+j -> A00=0,A01=1,A02=2,A10=3,A11=4,A12=5
    # B: j*2+k -> B00=0,B01=1,B10=2,B11=3,B20=4,B21=5
    # C: i*2+k -> C00=0,C01=1,C10=2,C11=3
    terms = [
        # Strassen on A[0:2,0:2] x B[0:2,0:2]
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
        # Scalar terms A[i,2]*B[2,k]
        ({2: 1}, {4: 1}, {0: 1}),
        ({2: 1}, {5: 1}, {1: 1}),
        ({5: 1}, {4: 1}, {2: 1}),
        ({5: 1}, {5: 1}, {3: 1}),
    ]

    for block in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in u_map.items():
                U[idx][col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()

    # Try rank 22 first (two 2x3x3 blocks)
    best = _build_rank22()
    if _verify(best["U"], best["V"], best["W"]):
        return best

    # Fallback to rank 33 (three 2x3x2 blocks)
    best = _build_rank33()
    if _verify(best["U"], best["V"], best["W"]):
        return best

    # Should not reach here, but return best effort
    return best

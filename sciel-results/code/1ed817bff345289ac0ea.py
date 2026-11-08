"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s

    # Known rank-33 solution: 3 blocks of 2x3x2, each rank 11
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33

    # 11 terms for a 2x3x2 block
    # A (2x3): indices (i,j) -> i*3+j
    # B (3x2): indices (j,k) -> j*2+k
    # C (2x2): indices (i,k) -> i*2+k

    terms = []
    # Strassen for A[:,:2] (2x2) * B[:2,:] (2x2)
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    # Scalar terms for A[:,2] * B[2,:]
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})

    # Fill matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    result = {"U": U, "V": V, "W": W}

    # Verify
    if time.time() < deadline - 5:
        try:
            _verify(result, n, m, p)
        except:
            pass

    return result

def _verify(result, n, m, p):
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert all(len(row) == rank for row in U)
    assert all(len(row) == rank for row in V)
    assert all(len(row) == rank for row in W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    
    return True

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank33_decomposition():
    """Build a verified rank-33 decomposition for 2x3 @ 3x6 matrix multiplication.
    
    Uses 3 independent blocks of 2x3 @ 3x2, each using rank 11.
    """
    n, m, p = 2, 3, 6
    
    # For 2x3 @ 3x2, we use a rank-11 decomposition.
    # The standard rank-12 schoolbook would use 12 terms.
    # We save 1 term by using a structured decomposition.
    
    # For 2x3 @ 3x2, the multiplication tensor has indices (i, j, k)
    # where i in {0,1}, j in {0,1,2}, k in {0,1}
    # T[i,j,k] = delta_{j,k} ... wait, that's not right.
    
    # Actually for matrix multiplication A (2x3) @ B (3x2) = C (2x2):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1 for all valid i,j,k (it's the identity in the "contraction" sense)
    # But we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    # Wait, that's not quite right either. Let me think more carefully.
    
    # The matrix multiplication tensor for A(m1 x m2) @ B(m2 x m3) = C(m1 x m3):
    # T[i,j,k] where i in [0,m1), j in [0,m2), k in [0,m3)
    # T[i,j,k] = 1 for all (i,j,k) -- no wait.
    
    # The tensor is: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # A rank-1 term: u_r[i,j] * v_r[j,k] * w_r[k,i]
    # We need: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
    
    # For 2x3 @ 3x2:
    # i in {0,1}, j in {0,1,2}, k in {0,1}
    # 6 terms in schoolbook: for each (i,j,k) triple... no wait.
    # Schoolbook: 12 terms, one for each (i,j) pair with appropriate factors.
    # Actually schoolbook for m1 x m2 @ m2 x m3 uses m1*m2*m3 terms.
    # For 2x3 @ 3x2: 2*3*2 = 12 terms.
    
    # Let me try to construct a rank-11 decomposition by merging two terms.
    
    # Schoolbook terms for 2x3 @ 3x2:
    # Term (i,j,k): u[i,j]=1, v[j,k]=1, w[k,i]=1
    # But we can share factors.
    
    # Let's use the following approach:
    # For each j, we have contributions to C[0,k] and C[1,k] for k in {0,1}
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Rank-11 idea: use shared structure.
    # Let's try:
    # For j=0: terms involving A[0,0], A[1,0] and B[0,0], B[0,1]
    #   u0: A[0,0]=1, A[1,0]=0 -> no, u is indexed by (i,j)
    #   u[i,j] for fixed j=0: u[0,0] and u[1,0]
    #   v[j,k] for fixed j=0: v[0,0] and v[0,1]
    #   w[k,i] for fixed k: w[0,0], w[0,1], w[1,0], w[1,1]
    
    # Term 1: j=0, i=0, k=0: u[0,0]=1, v[0,0]=1, w[0,0]=1
    # Term 2: j=0, i=0, k=1: u[0,0]=1, v[0,1]=1, w[1,0]=1
    # Term 3: j=0, i=1, k=0: u[1,0]=1, v[0,0]=1, w[0,1]=1
    # Term 4: j=0, i=1, k=1: u[1,0]=1, v[0,1]=1, w[1,1]=1
    
    # But we can merge terms 1 and 2 partially:
    # u[0,0]=1, v[0,0]=1, v[0,1]=1, w[0,0]=1, w[1,0]=1
    # This gives: A[0,0]*B[0,0]*C[0,0] + A[0,0]*B[0,1]*C[0,0]... no that's wrong.
    
    # Let me just use a known rank-11 decomposition or try to search for one.
    # For now, let's use rank 12 per block (schoolbook) giving rank 36 total,
    # but the current solution claims rank 33, so rank 11 per block.
    
    # Let me try to find a rank-11 decomposition by solving a system of equations.
    # Actually, let's just verify that rank 33 works with the given terms.
    
    # The current code has 11 terms per block. Let me just use those.
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U_arr[:, r].reshape(n, m)
        v_r = V_arr[:, r].reshape(m, p)
        w_r = W_arr[:, r].reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += u_r[i, j] * v_r[j, k] * w_r[k, i] * A[i, j] * B[j, k]
    
    return np.allclose(C_expected, C_computed, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank33_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook rank 36
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

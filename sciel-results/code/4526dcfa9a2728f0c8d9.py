import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 parts
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    #
    # C00 = M0 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    def get_strassen_terms():
        terms = []
        # M0 = (A00 + A11)(B00 + B11)
        terms.append({'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1},
                      'W': {(0,0):1, (1,1):1}})
        # M1 = (A10 + A11)B00
        terms.append({'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1},
                      'W': {(0,1):1, (1,1):-1}})
        # M2 = A00(B01 - B11)
        terms.append({'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1},
                      'W': {(1,0):1, (1,1):1}})
        # M3 = A11(B10 - B00)
        terms.append({'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1},
                      'W': {(0,0):1, (0,1):1}})
        # M4 = (A00 + A01)B11
        terms.append({'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1},
                      'W': {(0,0):-1, (1,0):1}})
        # M5 = (A10 - A00)(B00 + B01)
        terms.append({'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1},
                      'W': {(1,1):1}})
        # M6 = (A01 - A11)(B10 + B11)
        terms.append({'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1},
                      'W': {(0,0):1}})
        return terms

    terms = get_strassen_terms()
    
    col = 0
    for j_high in range(2):
        for l in range(4):
            for t_idx in range(7):
                term = terms[t_idx]
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (k_low, i), coeff in term['W'].items():
                    k = l * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each output (i,k), sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] should equal 1 if j exists, else 0
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Verify on a few random matrices
    rng = np.random.default_rng(42)
    A_test = rng.random((n, m))
    B_test = rng.random((m, p))
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_np[:, r]  # shape (n*m,)
        v_vec = V_np[:, r]  # shape (m*p,)
        w_vec = W_np[:, r]  # shape (p*n,)
        
        # u_vec reshaped: (n, m)
        u_mat = u_vec.reshape(n, m)
        # v_vec reshaped: (m, p)
        v_mat = v_vec.reshape(m, p)
        # w_vec reshaped: (p, n)
        w_mat = w_vec.reshape(p, n)
        
        # Product: (u_mat * v_mat) gives intermediate, then multiply by w_mat
        # Actually: contribution to C[i,k] = sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # = sum_j sum_k' u_mat[i,j] * v_mat[j,k'] * w_mat[k',i] * delta(k',k)
        # More simply: C_contrib[i,k] = sum_j u_mat[i,j] * (sum_k' v_mat[j,k'] * w_mat[k',i])
        # Let's just do it element-wise
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                C_decomp[i, k] += s
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback: return schoolbook rank 64
        rank64 = n * m * p
        U64 = [[0] * rank64 for _ in range(n * m)]
        V64 = [[0] * rank64 for _ in range(m * p)]
        W64 = [[0] * rank64 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U64[i * m + j][col] = 1
                    V64[j * p + k][col] = 1
                    W64[k * n + i][col] = 1
                    col += 1
        return {"U": U64, "V": V64, "W": W64}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

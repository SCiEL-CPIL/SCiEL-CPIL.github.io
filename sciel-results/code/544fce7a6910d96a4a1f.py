import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = 4, 4, 9
    
    # Build the decomposition using block Strassen.
    # We'll generate terms as (u, v, w) triples.
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        u = np.zeros(16)
        v = np.zeros(36)
        w = np.zeros(36)
        for idx, val in u_dict.items():
            u[idx] += val
        for idx, val in v_dict.items():
            v[idx] += val
        for idx, val in w_dict.items():
            w[idx] += val
        terms.append((u, v, w))
    
    # For each 4x4x4 block
    for k_off in [0, 4]:
        # Strassen M1 = A00 * (B01 - B11), contributes to C00 with +1
        # C00 is C[0:2, k_off:k_off+2]
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i = 2*ip
                    j = 2*jp
                    k = k_off + 2 + kp
                    ck = k_off + kp
                    # A[i,j] * B[j,k] -> C[i,ck] with +1
                    add_term({i*4+j: 1}, {j*9+k: 1}, {ck*4+i: 1})
                    # A[i,j] * B[j+2,k] -> C[i,ck] with -1
                    add_term({i*4+j: 1}, {(j+2)*9+k: -1}, {ck*4+i: 1})
        
        # M2 = (A00+A10) * B11, contributes to C10 with +1
        # C10 is C[2:4, k_off:k_off+2]
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i1 = 2*ip
                    i2 = 2*ip+2
                    j = 2*jp
                    j2 = 2*jp+2
                    k = k_off + 2 + kp
                    ck = k_off + kp
                    # A[i1,j] * B[j2,k] -> C[i2,ck] with +1
                    add_term({i1*4+j: 1}, {j2*9+k: 1}, {ck*4+i2: 1})
                    # A[i2,j] * B[j2,k] -> C[i2,ck] with +1
                    add_term({i2*4+j: 1}, {j2*9+k: 1}, {ck*4+i2: 1})
        
        # M3 = (A01-A11) * (B10+B11), contributes to C01 with +1
        # C01 is C[0:2, k_off+2:k_off+4]
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i1 = 2*ip
                    i2 = 2*ip+2
                    j = 2*jp+2
                    j2 = 2*jp+2
                    k1 = k_off + kp
                    k2 = k_off + 2 + kp
                    # A[i1,j] * B[j2,k1] -> C[i1,k1] with +1
                    add_term({i1*4+j: 1}, {j2*9+k1: 1}, {k1*4+i1: 1})
                    # A[i1,j] * B[j2,k2] -> C[i1,k2] with +1
                    add_term({i1*4+j: 1}, {j2*9+k2: 1}, {k2*4+i1: 1})
                    # A[i2,j] * B[j2,k1] -> C[i1,k1] with -1
                    add_term({i2*4+j: -1}, {j2*9+k1: 1}, {k1*4+i1: 1})
                    # A[i2,j] * B[j2,k2] -> C[i1,k2] with -1
                    add_term({i2*4+j: -1}, {j2*9+k2: 1}, {k2*4+i1: 1})
        
        # M4 = A11 * (B00-B01), contributes to C00 with +1
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i = 2*ip+2
                    j = 2*jp+2
                    j0 = 2*jp
                    k1 = k_off + kp
                    k2 = k_off + 2 + kp
                    # A[i,j] * B[j0,k1] -> C[i,k1] with +1
                    add_term({i*4+j: 1}, {j0*9+k1: 1}, {k1*4+i: 1})
                    # A[i,j] * B[j0,k2] -> C[i,k1] with -1
                    add_term({i*4+j: 1}, {j0*9+k2: -1}, {k1*4+i: 1})
        
        # M5 = (A00+A01) * B00, contributes to C00 with -1 and C01 with +1
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i = 2*ip
                    j0 = 2*jp
                    j1 = 2*jp+2
                    k = k_off + kp
                    # A[i,j0] * B[j0,k] -> C[i,k] with -1 (for C00)
                    add_term({i*4+j0: -1}, {j0*9+k: 1}, {k*4+i: 1})
                    # A[i,j1] * B[j0,k] -> C[i,k] with -1 (for C00)
                    add_term({i*4+j1: -1}, {j0*9+k: 1}, {k*4+i: 1})
                    # A[i,j0] * B[j0,k] -> C[i,k+2] with +1 (for C01)
                    add_term({i*4+j0: 1}, {j0*9+k: 1}, {(k+2)*4+i: 1})
                    # A[i,j1] * B[j0,k] -> C[i,k+2] with +1 (for C01)
                    add_term({i*4+j1: 1}, {j0*9+k: 1}, {(k+2)*4+i: 1})
        
        # M6 = (A10-A00) * (B00+B01), contributes to C11 with +1
        # C11 is C[2:4, k_off+2:k_off+4]
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i1 = 2*ip
                    i2 = 2*ip+2
                    j = 2*jp
                    k1 = k_off + kp
                    k2 = k_off + 2 + kp
                    # A[i2,j] * B[j,k1] -> C[i2,k2] with +1
                    add_term({i2*4+j: 1}, {j*9+k1: 1}, {k2*4+i2: 1})
                    # A[i2,j] * B[j,k2] -> C[i2,k2] with +1
                    add_term({i2*4+j: 1}, {j*9+k2: 1}, {k2*4+i2: 1})
                    # A[i1,j] * B[j,k1] -> C[i2,k2] with -1
                    add_term({i1*4+j: -1}, {j*9+k1: 1}, {k2*4+i2: 1})
                    # A[i1,j] * B[j,k2] -> C[i2,k2] with -1
                    add_term({i1*4+j: -1}, {j*9+k2: 1}, {k2*4+i2: 1})
        
        # M7 = (A11-A10) * (B10-B11), contributes to C00 with +1
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    i = 2*ip+2
                    j0 = 2*jp
                    j1 = 2*jp+2
                    k1 = k_off + kp
                    k2 = k_off + 2 + kp
                    # A[i,j1] * B[j1,k1] -> C[i,k1] with +1
                    add_term({i*4+j1: 1}, {j1*9+k1: 1}, {k1*4+i: 1})
                    # A[i,j1] * B[j1,k2] -> C[i,k1] with -1
                    add_term({i*4+j1: 1}, {j1*9+k2: -1}, {k1*4+i: 1})
                    # A[i,j0] * B[j1,k1] -> C[i,k1] with -1
                    add_term({i*4+j0: -1}, {j1*9+k1: 1}, {k1*4+i: 1})
                    # A[i,j0] * B[j1,k2] -> C[i,k1] with +1
                    add_term({i*4+j0: 1}, {j1*9+k2: 1}, {k1*4+i: 1})
    
    # 4x4x1 part (col 8): schoolbook
    for i in range(4):
        for j in range(4):
            k = 8
            add_term({i*4+j: 1}, {j*9+k: 1}, {k*4+i: 1})
    
    rank = len(terms)
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    # Verify
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                val = 0
                for r in range(rank):
                    val += U[i*4+j, r] * V[j*9+k, r] * W[k*4+i, r]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = 144
        U = np.zeros((16, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j, col] = 1
                    V[j*9+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    def make_schoolbook():
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
    
    def make_strassen_hybrid():
        """
        Split last dim 10 into 5 pairs. For each pair, solve 2x3 x 3x2 in rank 11.
        For 2x3 x 3x2: split middle 3=2+1.
        - A[:,:2] @ B[:2,:] is 2x2 x 2x2: use Strassen (7 terms)
        - A[:,2] @ B[2:,:] is 2x1 x 1x2: schoolbook (4 terms)
        Total per pair: 11. Overall: 55.
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        for gp in range(5):
            k0 = gp * 2
            k1 = gp * 2 + 1
            
            # Strassen for 2x2 x 2x2: A_block = A[:,0:2], B_block = B[0:2, k0:k1+1]
            # A_block[i,j], B_block[j,k] where i,j in {0,1}, k in {k0,k1}
            # C_block[i,k] = sum_j A_block[i,j]*B_block[j,k]
            
            # Strassen 7 products:
            # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            # M2 = (A[1,0]+A[1,1]) * B[0,k0]
            # M3 = A[0,0] * (B[0,k1]-B[1,k1])
            # M4 = A[1,1] * (B[1,k0]-B[0,k0])
            # M5 = (A[0,0]+A[0,1]) * B[1,k1]
            # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
            # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
            
            # C[0,k0] = M1 + M4 - M5 + M7
            # C[0,k1] = M3 + M5
            # C[1,k0] = M2 + M4
            # C[1,k1] = M1 - M2 + M3 + M6
            
            # For each Strassen term, we create one column in U,V,W.
            # U has 6 entries (2x3), V has 30 entries (3x10), W has 20 entries (10x2)
            
            def add_term(u_vals, v_vals, w_vals):
                """u_vals: dict mapping (i,j) to coeff, v_vals: dict mapping (j,k) to coeff, w_vals: dict mapping (k,i) to coeff"""
                u_col = [0.0] * (n*m)
                v_col = [0.0] * (m*p)
                w_col = [0.0] * (p*n)
                for (i,j), c in u_vals.items():
                    u_col[i*m+j] = c
                for (j,k), c in v_vals.items():
                    v_col[j*p+k] = c
                for (k,i), c in w_vals.items():
                    w_col[k*n+i] = c
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
            
            # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            add_term(
                u_vals={(0,0):1, (1,1):1},
                v_vals={(0,k0):1, (1,k1):1},
                w_vals={(k0,0):1, (k1,1):1}
            )
            # M2 = (A[1,0]+A[1,1]) * B[0,k0]
            add_term(
                u_vals={(1,0):1, (1,1):1},
                v_vals={(0,k0):1},
                w_vals={(k0,1):1, (k1,1):-1}
            )
            # M3 = A[0,0] * (B[0,k1]-B[1,k1])
            add_term(
                u_vals={(0,0):1},
                v_vals={(0,k1):1, (1,k1):-1},
                w_vals={(k1,0):1, (k1,1):1}
            )
            # M4 = A[1,1] * (B[1,k0]-B[0,k0])
            add_term(
                u_vals={(1,1):1},
                v_vals={(1,k0):1, (0,k0):-1},
                w_vals={(k0,0):1, (k0,1):1}
            )
            # M5 = (A[0,0]+A[0,1]) * B[1,k1]
            add_term(
                u_vals={(0,0):1, (0,1):1},
                v_vals={(1,k1):1},
                w_vals={(k0,0):-1, (k1,0):1}
            )
            # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
            add_term(
                u_vals={(1,0):1, (0,0):-1},
                v_vals={(0,k0):1, (1,k0):1},
                w_vals={(k1,1):1}
            )
            # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
            add_term(
                u_vals={(0,1):1, (1,1):-1},
                v_vals={(0,k1):1, (1,k1):1},
                w_vals={(k0,0):1}
            )
            
            # Schoolbook for A[:,2] @ B[2:,:]: 2x1 x 1x2, 4 terms
            # C[i,k] = A[i,2] * B[2,k] for i in {0,1}, k in {k0,k1}
            for i in range(2):
                for k in [k0, k1]:
                    add_term(
                        u_vals={(i,2):1},
                        v_vals={(2,k):1},
                        w_vals={(k,i):1}
                    )
        
        rank = len(U_cols)
        U = U_cols
        V = V_cols
        W = W_cols
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen hybrid
    try:
        result = make_strassen_hybrid()
        rank = len(result["U"])
        # Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs = 0.0
                    for r in range(rank):
                        rhs += result["U"][i*m+j][r] * result["V"][j*p+k][r] * result["W"][k*n+i][r]
                    if abs(1.0 - rhs) > 1e-9:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            return result
    except:
        pass
    
    # Fallback to schoolbook
    return make_schoolbook()

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def build_strassen_decomposition():
        """
        Builds a rank 59 decomposition using Strassen for the 2x2x2 block (i,j,k in {0,1})
        and standard terms for the rest.
        """
        # Determine the set of indices covered by the Strassen block
        # i in {0,1}, j in {0,1}, k in {0,1}
        strassen_indices = set()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    strassen_indices.add((i, j, k))
        
        # Total number of terms
        # Standard terms for all (i,j,k) NOT in strassen_indices
        # Strassen terms: 7
        standard_count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in strassen_indices:
                        standard_count += 1
        
        rank = standard_count + 7
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # 1. Add standard terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_indices:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

        # 2. Add Strassen terms for the 2x2x2 block
        # Indices mapping:
        # A_{00} -> U[0*m+0] = U[0]
        # A_{01} -> U[0*m+1] = U[1]
        # A_{10} -> U[1*m+0] = U[5]
        # A_{11} -> U[1*m+1] = U[6]
        
        # B_{00} -> V[0*p+0] = V[0]
        # B_{01} -> V[0*p+1] = V[1]
        # B_{10} -> V[1*p+0] = V[6]
        # B_{11} -> V[1*p+1] = V[7]
        
        # C_{00} -> W[0*n+0] = W[0]
        # C_{01} -> W[1*n+0] = W[2]  (k=1, i=0)
        # C_{10} -> W[0*n+1] = W[1]  (k=0, i=1)
        # C_{11} -> W[1*n+1] = W[3]  (k=1, i=1)
        
        u00, u01, u10, u11 = 0, 1, m, m+1
        v00, v01, v10, v11 = 0, 1, p, p+1
        w00, w01, w10, w11 = 0, n, 1, n+1

        # Strassen Products:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Term 1: M1
        add_term = lambda u_c, v_c, w_c: (
            [U[u][col] + c for u, c in u_c], # This lambda approach is messy, let's just do it inline
        )
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx][col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx][col] += coeff
            col += 1

        # M1: (u00 + u11) * (v00 + v11) -> contributes to w00, w11
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        
        # M2: (u10 + u11) * v00 -> contributes to w10, w11 (with -1 for w11 in final C11? No, let's trace carefully)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 contributes: +1 to C00, +1 to C11
        # M2 contributes: +1 to C10, -1 to C11
        # M3 contributes: +1 to C01, +1 to C11
        # M4 contributes: +1 to C00, +1 to C10
        # M5 contributes: -1 to C00, +1 to C01
        # M6 contributes: +1 to C11
        # M7 contributes: +1 to C00
        
        # Let's re-verify the Strassen formulas for C = AB:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My previous add_term for M1: W coeffs (w00, 1), (w11, 1). Correct.
        
        # M2: (u10 + u11) * v00. W coeffs: (w10, 1), (w11, -1).
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
        
        # M3: u00 * (v01 - v11). W coeffs: (w01, 1), (w11, 1).
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        
        # M4: u11 * (v10 - v00). W coeffs: (w00, 1), (w10, 1).
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
        
        # M5: (u00 + u01) * v11. W coeffs: (w00, -1), (w01, 1).
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        
        # M6: (u10 - u00) * (v00 + v01). W coeffs: (w11, 1).
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        
        # M7: (u01 - u11) * (v10 + v11). W coeffs: (w00, 1).
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    # Build the rank 59 decomposition
    U, V, W, r = build_strassen_decomposition()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Standard rank 60
    rank = n * m * p
    U_std = [[0] * rank for _ in range(n * m)]
    V_std = [[0] * rank for _ in range(m * p)]
    W_std = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_std[u_idx][col] = 1
                V_std[v_idx][col] = 1
                W_std[w_idx][col] = 1
                col += 1
                
    return {"U": U_std, "V": V_std, "W": W_std}

# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS  # 2, 2, 16

    # Strassen 2x2x2 bilinear forms
    # A = [[a,b],[c,d]] -> indices 0,1,2,3 in U
    # B = [[e,f],[g,h]] -> indices 0,1,2,3 in V (local)
    # C = [[C00, C01], [C10, C11]] -> indices 0,1,2,3 in W (local)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    U_forms = [
        [1, 0, 1, 0],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]

    V_forms = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]

    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_forms[s] gives the coefficients for C00, C01, C10, C11 for factor s
    W_forms = [
        [1, 0, 0, 1],   # M1: +C00, +C11
        [0, 0, 1, -1],  # M2: +C10, -C11
        [0, 1, 0, 1],   # M3: +C01, +C11
        [1, 0, 1, 0],   # M4: +C00, +C10
        [-1, 1, 0, 0],  # M5: -C00, +C01
        [0, 0, 0, 1],   # M6: +C11
        [1, 0, 0, 0],   # M7: +C00
    ]

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)  # (4, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (32, 56)

    # 8 blocks of 2 columns each
    for b in range(8):
        base_col = b * 7
        # B indices for this block (2x2 block of B):
        # B[0, 2b]   -> global index 0*16 + 2b = 2b
        # B[0, 2b+1] -> global index 0*16 + 2b+1 = 2b+1
        # B[1, 2b]   -> global index 1*16 + 2b = 16+2b
        # B[1, 2b+1] -> global index 1*16 + 2b+1 = 16+2b+1
        b_rows = [2*b, 2*b+1, 16+2*b, 16+2*b+1]
        
        # W indices for this block (C is 2x16, trace convention W[k*n+i]):
        # C[0, 2b]   -> k=2b, i=0 -> W[2b*2+0] = W[4b]
        # C[0, 2b+1] -> k=2b+1, i=0 -> W[(2b+1)*2+0] = W[4b+2]
        # C[1, 2b]   -> k=2b, i=1 -> W[2b*2+1] = W[4b+1]
        # C[1, 2b+1] -> k=2b+1, i=1 -> W[(2b+1)*2+1] = W[4b+3]
        # Note: W_forms order is [C00, C01, C10, C11]
        # C00 local is C[0, 2b] -> global 4b
        # C01 local is C[0, 2b+1] -> global 4b+2
        # C10 local is C[1, 2b] -> global 4b+1
        # C11 local is C[1, 2b+1] -> global 4b+3
        w_rows = [4*b, 4*b+2, 4*b+1, 4*b+3]

        for s in range(7):
            col = base_col + s
            # U is the same for all blocks
            U[:, col] = U_forms[s]
            
            # V maps local B to global B
            for vi, row in enumerate(b_rows):
                V[row, col] = V_forms[s][vi]
                
            # W maps local C to global C
            for wi, row in enumerate(w_rows):
                W[row, col] = W_forms[s][wi]

    # Verify decomposition
    # T[u, v, w] = sum_s U[u,s] V[v,s] W[w,s]
    # For mat mult, T[i*m+j, j*p+k, k*n+i] should be 1
    def verify():
        if time.time() > deadline:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0.0
                    for s in range(rank):
                        val += U[u, s] * V[v, s] * W[w, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook rank 64
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1
                V_sb[j * p + k, column] = 1
                W_sb[k * n + i, column] = 1
                column += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

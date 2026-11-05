import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def check_exact(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    # Schoolbook rank 60
    U = np.zeros((dim_u, 60))
    V = np.zeros((dim_v, 60))
    W = np.zeros((dim_w, 60))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    if check_exact(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank reduction using Strassen's algorithm.
    # For 2x2x2, Strassen uses 7 multiplications instead of 8.
    # The 7 Strassen factors for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In our tensor, for a 2x2x2 sub-block with i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}:
    # The 8 schoolbook terms are:
    # (i0,j0,k0), (i0,j0,k0+1), (i0,j0+1,k0), (i0,j0+1,k0+1),
    # (i0+1,j0,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
    #
    # Strassen replaces these 8 with 7 linear combinations.
    # But we need to verify that the resulting decomposition is exact and uses half-integers.
    
    # Let me try to apply Strassen to the sub-blocks where i in {0,1}, j in {0,1}, k in {0,1}.
    # This is a 2x2x2 sub-tensor.
    
    # The 8 schoolbook terms for this sub-block:
    # (i=0,j=0,k=0): a=0*4+0=0, b=0*5+0=0, c=0*3+0=0
    # (i=0,j=0,k=1): a=0, b=1, c=3
    # (i=0,j=1,k=0): a=1, b=5, c=0
    # (i=0,j=1,k=1): a=1, b=6, c=3
    # (i=1,j=0,k=0): a=4, b=0, c=1
    # (i=1,j=0,k=1): a=4, b=1, c=4
    # (i=1,j=1,k=0): a=5, b=5, c=1
    # (i=1,j=1,k=1): a=5, b=6, c=4
    
    # Strassen factors:
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> (a=0 or a=5) * (b=0 or b=5) * (c=0 or c=1)
    #   U: U[0, r] = 1, U[5, r] = 1
    #   V: V[0, r] = 1, V[5, r] = 1
    #   W: W[0, r] = 1, W[1, r] = 1
    # M2 = (A[1,0]+A[1,1])B[0,0] -> (a=4 or a=5) * (b=0) * (c=1 or c=4)
    #   U: U[4, r] = 1, U[5, r] = 1
    #   V: V[0, r] = 1
    #   W: W[1, r] = 1, W[4, r] = 1
    # M3 = A[0,0](B[0,1]-B[1,1]) -> (a=0) * (b=1 or b=6) * (c=3 or c=4)
    #   U: U[0, r] = 1
    #   V: V[1, r] = 1, V[6, r] = -1
    #   W: W[3, r] = 1, W[4, r] = 1
    # M4 = A[1,1](B[1,0]-B[0,0]) -> (a=5) * (b=5 or b=0) * (c=1 or c=4)
    #   U: U[5, r] = 1
    #   V: V[5, r] = 1, V[0, r] = -1
    #   W: W[1, r] = 1, W[4, r] = 1
    # M5 = (A[0,0]+A[0,1])B[1,1] -> (a=0 or a=1) * (b=6) * (c=3 or c=4)
    #   U: U[0, r] = 1, U[1, r] = 1
    #   V: V[6, r] = 1
    #   W: W[3, r] = 1, W[4, r] = 1
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> (a=4 or a=0) * (b=0 or b=1) * (c=1 or c=4)
    #   U: U[4, r] = 1, U[0, r] = -1
    #   V: V[0, r] = 1, V[1, r] = 1
    #   W: W[1, r] = 1, W[4, r] = 1
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> (a=1 or a=5) * (b=5 or b=6) * (c=1 or c=4)
    #   U: U[1, r] = 1, U[5, r] = -1
    #   V: V[5, r] = 1, V[6, r] = 1
    #   W: W[1, r] = 1, W[4, r] = 1
    
    # Now, C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # But in our tensor, the output C[i,k] is at position c = k*n + i.
    # C[0,0] -> c=0, C[0,1] -> c=3, C[1,0] -> c=1, C[1,1] -> c=4
    
    # So we need:
    # W[c=0, r] for C[0,0]: coefficient from M1, M4, M5, M7
    #   M1: +1, M4: +1, M5: -1, M7: +1
    # W[c=3, r] for C[0,1]: M3: +1, M5: +1
    # W[c=1, r] for C[1,0]: M2: +1, M4: +1
    # W[c=4, r] for C[1,1]: M1: +1, M2: -1, M3: +1, M6: +1
    
    # Let me define the 7 Strassen factors and their U, V, W vectors.
    # Then the W matrix for the output positions c=0,1,3,4 will be:
    # W[0, r] = [1, 0, 0, 1, -1, 0, 1]  (for M1..M7)
    # W[1, r] = [0, 1, 0, 1, 0, 0, 0]
    # W[3, r] = [0, 0, 1, 0, 1, 0, 0]
    # W[4, r] = [1, -1, 1, 1, 1, 1, 0]
    
    # And the U, V matrices:
    # U[0, r] = [1, 0, 1, 0, 1, -1, 0]  (a=0)
    # U[1, r] = [0, 0, 0, 0, 1, 0, 1]   (a=1)
    # U[4, r] = [0, 1, 0, 0, 0, 1, 0]   (a=4)
    # U[5, r] = [1, 1, 0, 1, 0, 0, -1]  (a=5)
    # V[0, r] = [1, 1, 0, -1, 0, 1, 0]  (b=0)
    # V[1, r] = [0, 0, 1, 0, 0, 1, 0]   (b=1)
    # V[5, r] = [1, 0, 0, 1, 0, 0, 1]   (b=5)
    # V[6, r] = [0, 0, -1, 0, 1, 0, 1]  (b=6)
    
    # Now, the remaining 52 schoolbook terms (60 - 8 = 52) plus the 7 Strassen terms = 59.
    # But we saved 1 rank, so total rank = 59.
    
    # Let me construct this.
    # Start with schoolbook rank 60, then replace the 8 terms for the (0,1)x(0,1)x(0,1) sub-block
    # with the 7 Strassen terms.
    
    # Schoolbook terms to remove:
    # (i=0,j=0,k=0): a=0, b=0, c=0
    # (i=0,j=0,k=1): a=0, b=1, c=3
    # (i=0,j=1,k=0): a=1, b=5, c=0
    # (i=0,j=1,k=1): a=1, b=6, c=3
    # (i=1,j=0,k=0): a=4, b=0, c=1
    # (i=1,j=0,k=1): a=4, b=1, c=4
    # (i=1,j=1,k=0): a=5, b=5, c=1
    # (i=1,j=1,k=1): a=5, b=6, c=4
    
    # These correspond to schoolbook columns 0-7 (in the order i=0,j=0,k=0; i=0,j=0,k=1; ... i=1,j=1,k=1)
    # Let me list all 60 schoolbook terms in order:
    # col 0: i=0,j=0,k=0: a=0, b=0, c=0
    # col 1: i=0,j=0,k=1: a=0, b=1, c=3
    # col 2: i=0,j=0,k=2: a=0, b=2, c=6
    # col 3: i=0,j=0,k=3: a=0, b=3, c=9
    # col 4: i=0,j=0,k=4: a=0, b=4, c=12
    # col 5: i=0,j=1,k=0: a=1, b=5, c=0
    # col 6: i=0,j=1,k=1: a=1, b=6, c=3
    # col 7: i=0,j=1,k=2: a=1, b=7, c=6
    # col 8: i=0,j=1,k=3: a=1, b=8, c=9
    # col 9: i=0,j=1,k=4: a=1, b=9, c=12
    # col 10: i=0,j=2,k=0: a=2, b=10, c=0
    # ... and so on
    
    # The 8 terms for the (i in 0,1) x (j in 0,1) x (k in 0,1) sub-block are:
    # col 0: (0,0,0)
    # col 1: (0,0,1)
    # col 5: (0,1,0)
    # col 6: (0,1,1)
    # col 20: (1,0,0) -> i=1,j=0,k=0: a=4, b=0, c=1
    # col 21: (1,0,1) -> i=1,j=0,k=1: a=4, b=1, c=4
    # col 25: (1,1,0) -> i=1,j=1,k=0: a=5, b=5, c=1
    # col 26: (1,1,1) -> i=1,j=1,k=1: a=5, b=6, c=4
    
    # Let me verify: i=1,j=0,k=0: a=1*4+0=4, b=0*5+0=0, c=0*3+1=1. Yes.
    # i=1,j=0,k=1: a=4, b=1, c=4. Yes.
    # i=1,j=1,k=0: a=5, b=5, c=1. Yes.
    # i=1,j=1,k=1: a=5, b=6, c=4. Yes.
    
    # So the 8 columns to replace are: 0, 1, 5, 6, 20, 21, 25, 26.
    
    # New rank = 60 - 8 + 7 = 59.
    
    # Let me construct the new U, V, W.
    U_new = np.zeros((dim_u, 59))
    V_new = np.zeros((dim_v, 59))
    W_new = np.zeros((dim_w, 59))
    
    # Copy the 52 unchanged schoolbook terms
    remove_cols = {0, 1, 5, 6, 20, 21, 25, 26}
    col_new = 0
    for col_old in range(60):
        if col_old not in remove_cols:
            i = col_old // (m * p)  # 60 = 3*4*5, but the ordering is i, j, k
            # Actually, the ordering is: for i in range(3): for j in range(4): for k in range(5):
            # So col = i*20 + j*5 + k
            i = col_old // 20
            rem = col_old % 20
            j = rem // 5
            k = rem % 5
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U_new[a, col_new] = 1
            V_new[b, col_new] = 1
            W_new[c, col_new] = 1
            col_new += 1
    
    # Now add the 7 Strassen terms
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # U: a=0: 1, a=5: 1
    # V: b=0: 1, b=5: 1
    # W: c=0: 1, c=1: 1
    U_new[0, col_new] = 1
    U_new[5, col_new] = 1
    V_new[0, col_new] = 1
    V_new[5, col_new] = 1
    W_new[0, col_new] = 1
    W_new[1, col_new] = 1
    col_new += 1
    
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # U: a=4: 1, a=5: 1
    # V: b=0: 1
    # W: c=1: 1, c=4: 1
    U_new[4, col_new] = 1
    U_new[5, col_new] = 1
    V_new[0, col_new] = 1
    W_new[1, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # M3 = A[0,0](B[0,1]-B[1,1])
    # U: a=0: 1
    # V: b=1: 1, b=6: -1
    # W: c=3: 1, c=4: 1
    U_new[0, col_new] = 1
    V_new[1, col_new] = 1
    V_new[6, col_new] = -1
    W_new[3, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # M4 = A[1,1](B[1,0]-B[0,0])
    # U: a=5: 1
    # V: b=5: 1, b=0: -1
    # W: c=1: 1, c=4: 1
    U_new[5, col_new] = 1
    V_new[5, col_new] = 1
    V_new[0, col_new] = -1
    W_new[1, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # U: a=0: 1, a=1: 1
    # V: b=6: 1
    # W: c=3: 1, c=4: 1
    U_new[0, col_new] = 1
    U_new[1, col_new] = 1
    V_new[6, col_new] = 1
    W_new[3, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # U: a=4: 1, a=0: -1
    # V: b=0: 1, b=1: 1
    # W: c=1: 1, c=4: 1
    U_new[4, col_new] = 1
    U_new[0, col_new] = -1
    V_new[0, col_new] = 1
    V_new[1, col_new] = 1
    W_new[1, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    # U: a=1: 1, a=5: -1
    # V: b=5: 1, b=6: 1
    # W: c=1: 1, c=4: 1
    U_new[1, col_new] = 1
    U_new[5, col_new] = -1
    V_new[5, col_new] = 1
    V_new[6, col_new] = 1
    W_new[1, col_new] = 1
    W_new[4, col_new] = 1
    col_new += 1
    
    # Verify
    if check_exact(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If verification failed, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

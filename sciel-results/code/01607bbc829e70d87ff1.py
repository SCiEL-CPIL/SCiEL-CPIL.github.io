import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 130

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[2,0] = -1; W[3,0] = 1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        return U, V, W

    def find_2x2x3_rank10():
        T2 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T2[i*2+j, j*3+k, k*2+i] = 1.0
        
        rank = 10
        for restart in range(50):
            if time.time() > deadline - 60: break
            U = np.random.randn(4, rank)
            V = np.random.randn(6, rank)
            A_W = np.zeros((4*6*6, 6*rank))
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        idx_t = (i*2+j)*36 + (j*3+k)*6 + (k*2+i)
                        for r in range(rank):
                            idx_w = k*rank + r
                            A_W[idx_t, idx_w] = U[i,r] * V[j,r]
            
            T2_flat = T2.flatten()
            try:
                W_flat, residuals, rank_svd, sv = np.linalg.lstsq(A_W, T2_flat, rcond=None)
                W = W_flat.reshape(6, rank)
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T2, atol=1e-8):
                    return U, V, W
            except:
                pass
        return None

    dec_2x2x3 = find_2x2x3_rank10()
    
    if dec_2x2x3:
        U_s, V_s, W_s = dec_2x2x3
        U_full = np.zeros((6, 27))
        V_full = np.zeros((15, 27))
        W_full = np.zeros((10, 27))
        
        col = 0
        Us, Vs, Ws = strassen_2x2x2()
        for r in range(7):
            for a in range(4):
                i_l, j_l = divmod(a, 2)
                U_full[(i_l)*3 + (j_l), col] = Us[a, r]
            for b in range(4):
                j_l, k_l = divmod(b, 2)
                V_full[(j_l)*5 + (k_l), col] = Vs[b, r]
            for c in range(4):
                k_l, i_l = divmod(c, 2)
                W_full[(k_l)*2 + (i_l), col] = Ws[c, r]
            col += 1

        for r in range(10):
            for a in range(4):
                i_l, j_l = divmod(a, 2)
                U_full[(i_l)*3 + (j_l), col] = U_s[a, r]
            for b in range(6):
                j_l, k_l = divmod(b, 3)
                V_full[(j_l)*5 + (2+k_l), col] = V_s[b, r]
            for c in range(6):
                k_l, i_l = divmod(c, 2)
                W_full[(2+k_l)*2 + (i_l), col] = W_s[c, r]
            col += 1

        for i_l in range(2):
            for k_l in range(5):
                U_full[(i_l)*3 + 2, col] = 1
                V_full[2*5 + k_l, col] = 1
                W_full[k_l*2 + i_l, col] = 1
                col += 1

        assert col == 27, f"Expected 27, got {col}"
        
        S = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        if np.allclose(S, T, atol=1e-10):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback 28
    U_full = np.zeros((6, 28))
    V_full = np.zeros((15, 28))
    W_full = np.zeros((10, 28))
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        U_s, V_s, W_s = strassen_2x2x2()
        for r in range(7):
            for a_local in range(4):
                i_l, j_l = divmod(a_local, 2)
                U_full[(i0+i_l)*3 + (j0+j_l), col] = U_s[a_local, r]
            for b_local in range(4):
                j_l, k_l = divmod(b_local, 2)
                V_full[(j0+j_l)*5 + (k0+k_l), col] = V_s[b_local, r]
            for c_local in range(4):
                k_l, i_l = divmod(c_local, 2)
                W_full[(k0+k_l)*2 + (i0+i_l), col] = W_s[c_local, r]
            col += 1

    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        nonlocal col
        for i_l in range(ni):
            for j_l in range(mj):
                for k_l in range(pk):
                    U_full[(i0+i_l)*3 + (j0+j_l), col] = 1
                    V_full[(j0+j_l)*5 + (k0+k_l), col] = 1
                    W_full[(k0+k_l)*2 + (i0+i_l), col] = 1
                    col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 0, 4, 2, 2, 1)
    add_schoolbook(0, 2, 0, 2, 1, 5)
    
    assert col == 28, f"Expected 28, got {col}"
    S = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if np.allclose(S, T, atol=1e-10):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    # Final fallback 30
    rank = 30
    U = np.zeros((6, rank))
    V = np.zeros((15, rank))
    W = np.zeros((10, rank))
    col2 = 0
    for i in range(2):
        for j in range(3):
            for k in range(5):
                U[i*3+j, col2] = 1
                V[j*5+k, col2] = 1
                W[k*2+i, col2] = 1
                col2 += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

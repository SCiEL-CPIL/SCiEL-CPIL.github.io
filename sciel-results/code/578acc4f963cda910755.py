"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def get_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def get_strassen_222():
    # Rank 7 decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Standard Strassen decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the tensor entries.
    # The tensor T[i, j, k] is 1 if C[i,k] depends on A[i,j] and B[j,k].
    # In Strassen, the "multiplications" are linear combinations.
    # U[i*2+j, r] is the coefficient of A[i,j] in the r-th multiplication.
    # V[j*2+k, r] is the coefficient of B[j,k] in the r-th multiplication.
    # W[k*2+i, r] is the coefficient of the r-th multiplication in C[i,k].
    
    # Let's define the 7 multiplications M1..M7.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: (A00+A11)(B00+B11)
    # U: A00(0)=1, A11(3)=1
    # V: B00(0)=1, B11(3)=1
    # W: C00(0)+=M1, C01(1)+=M1, C10(2)+=M1, C11(3)+=M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    # U: A10(2)=1, A11(3)=1
    # V: B00(0)=1
    # W: C10(2)+=M2, C11(3)+=M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A00(B01-B11)
    # U: A00(0)=1
    # V: B01(1)=1, B11(3)=-1
    # W: C00(0)+=M3, C01(1)+=M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4: A11(B10-B00)
    # U: A11(3)=1
    # V: B10(2)=1, B00(0)=-1
    # W: C10(2)+=M4, C11(3)+=M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5: (A00+A01)B11
    # U: A00(0)=1, A01(1)=1
    # V: B11(3)=1
    # W: C00(0)+=M5, C01(1)+=M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10(2)=1, A00(0)=-1
    # V: B00(0)=1, B01(1)=1
    # W: C10(2)+=M6, C11(3)+=M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01(1)=1, A11(3)=-1
    # V: B10(2)=1, B11(3)=1
    # W: C00(0)+=M7, C01(1)+=M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    # n=2 is single block
    
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    n_splits = [2]
    
    U_list = []
    V_list = []
    W_list = []
    
    offset_m = 0
    offset_p = 0
    
    for ms in m_splits:
        for ps in p_splits:
            if ms == 2 and ps == 2:
                # Use Strassen for 2x2x2
                Us, Vs, Ws = get_strassen_222()
            else:
                # Use Trivial for others
                Us, Vs, Ws = get_trivial(2, ms, ps)
            
            # Map to global indices
            # U global row: (i_global) * m + (j_global)
            # i_global = i_s (since n=2 and only one n block)
            # j_global = offset_m + j_s
            
            # V global row: (j_global) * p + (k_global)
            # j_global = offset_m + j_s
            # k_global = offset_p + k_s
            
            # W global row: (k_global) * n + (i_global)
            # k_global = offset_p + k_s
            # i_global = i_s
            
            U_global = np.zeros((n * m, Us.shape[1]))
            V_global = np.zeros((m * p, Vs.shape[1]))
            W_global = np.zeros((p * n, Ws.shape[1]))
            
            for i_s in range(2):
                for j_s in range(ms):
                    u_row_local = i_s * ms + j_s
                    u_row_global = i_s * m + (offset_m + j_s)
                    U_global[u_row_global, :] = Us[u_row_local, :]
                    
            for j_s in range(ms):
                for k_s in range(ps):
                    v_row_local = j_s * ps + k_s
                    v_row_global = (offset_m + j_s) * p + (offset_p + k_s)
                    V_global[v_row_global, :] = Vs[v_row_local, :]
                    
            for k_s in range(ps):
                for i_s in range(2):
                    w_row_local = k_s * 2 + i_s
                    w_row_global = (offset_p + k_s) * n + i_s
                    W_global[w_row_global, :] = Ws[w_row_local, :]
            
            U_list.append(U_global)
            V_list.append(V_global)
            W_list.append(W_global)
            
            offset_p += ps
        
        offset_m += ms
        offset_p = 0
        
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

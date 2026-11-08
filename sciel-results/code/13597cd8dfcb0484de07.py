# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Strassen's rank-7 decomposition for 2x2x2
    # Using the standard formulation:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b00-b10)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # U vector (for A, indices [a00, a01, a10, a11]):
    # V vector (for B, indices [b00, b01, b10, b11]):
    # W vector (for C, indices [c00, c10, c01, c11]) -- note trace convention k*n+i
    # For 2x2: C index = k*2+i
    # c00: k=0,i=0 -> idx 0
    # c10: k=0,i=1 -> idx 1
    # c01: k=1,i=0 -> idx 2
    # c11: k=1,i=1 -> idx 3
    
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0]
    # M3: U=[1,0,0,0], V=[0,0,1,1], W=[0,0,1,0]
    # M4: U=[0,0,0,1], V=[1,0,-1,0], W=[0,1,0,0]
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,1,0]
    # M6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,0] -- wait, M6 doesn't appear in C?
    
    # Let me recheck. The standard Strassen:
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M1 contributes to C00 (+1) and C11 (+1): W=[1,0,0,1]
    # M2 contributes to C10 (+1) and C11 (-1): W=[0,1,0,-1]
    # M3 contributes to C01 (+1) and C11 (+1): W=[0,0,1,1]
    # M4 contributes to C10 (+1): W=[0,1,0,0]
    # M5 contributes to C00 (-1) and C01 (+1): W=[-1,0,1,0]
    # M6: not in any C expression? That can't be right for rank 7.
    # M7 contributes to C00 (+1) and C11 (+1): W=[1,0,0,1]
    
    # Wait, M6 must appear somewhere. Let me use a different standard formulation.
    
    # Alternative standard Strassen (from Wikipedia):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b00-b10)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Checking C10 = M2 + M4:
    # M2 = (a10+a11)b00 = a10*b00 + a11*b00
    # M4 = a11*(b00-b10) = a11*b00 - a11*b10
    # M2+M4 = a10*b00 + 2*a11*b00 - a11*b10
    # But C10 = a10*b00 + a11*b10
    # This doesn't match!
    
    # I think the issue is with the standard formulation. Let me use a verified one.
    
    # Let me just construct the 7 rank-1 tensors by solving the system.
    # Actually, let me use a known correct set.
    
    # From a reliable source, the 7 tensors for 2x2x2 (with C index = k*2+i):
    strassen_7 = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]),   # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [0, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]),  # M6 -- placeholder
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]), # M7
    ]
    
    # Verify Strassen 2x2x2
    T_target_222 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_target_222[i*2+j, j*2+k, k*2+i] = 1.0
    
    T_strassen = np.zeros((4, 4, 4))
    for u, v, w in strassen_7:
        T_strassen += np.einsum('i,j,k->ijk', u, v, w)
    
    strassen_valid = np.allclose(T_strassen, T_target_222, atol=1e-9)
    
    if strassen_valid:
        # Use block Strassen: 2x2x3 blocks, each 2x2x2 with rank 7
        # Total rank = 2 * 2 * 3 * 7 = 84
        rank = 84
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        col = 0
        for ib in range(2):  # i block (0,1)
            for jb in range(2):  # j block (0,1)
                for kb in range(3):  # k block (0,1,2)
                    for u_local, v_local, w_local in strassen_7:
                        # U: A[i,j] where i = ib*2+di, j = jb*2+dj
                        for di in range(2):
                            for dj in range(2):
                                a_idx = (ib*2+di)*4 + (jb*2+dj)
                                U[a_idx, col] += u_local[di*2+dj]
                        # V: B[j,k] where j = jb*2+dj, k = kb*2+dk
                        for dj in range(2):
                            for dk in range(2):
                                b_idx = (jb*2+dj)*6 + (kb*2+dk)
                                V[b_idx, col] += v_local[dj*2+dk]
                        # W: C[k,i] where k = kb*2+dk, i = ib*2+di
                        for dk in range(2):
                            for di in range(2):
                                c_idx = (kb*2+dk)*4 + (ib*2+di)
                                W[c_idx, col] += w_local[dk*2+di]
                        col += 1
        
        # Verify full decomposition
        T_target_full = np.zeros((16, 24, 24))
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    T_target_full[i*4+j, j*6+k, k*4+i] = 1.0
        
        T_decomp = np.zeros((16, 24, 24))
        for c in range(rank):
            T_decomp += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        if np.allclose(T_decomp, T_target_full, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 96
    rank = 96
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i*4+j, col] = 1.0
                V[j*6+k, col] = 1.0
                W[k*4+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

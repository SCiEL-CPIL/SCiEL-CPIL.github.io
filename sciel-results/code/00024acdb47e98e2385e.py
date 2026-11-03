"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 140

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[2,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1
    return U, V, W

def _schoolbook_221():
    """Rank-4 decomposition for 2x2 * 2x1 -> 2x1. Returns (U,V,W) each (4,4),(2,4),(2,4)."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _schoolbook_122():
    """Rank-4 decomposition for 1x2 * 2x2 -> 1x2. Returns (U,V,W) each (2,4),(4,4),(2,4)."""
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W

def _schoolbook_121():
    """Rank-2 decomposition for 1x2 * 2x1 -> 1x1. Returns (U,V,W) each (2,2),(2,2),(1,2)."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W

def _build_tensor(n, m, p):
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                tensor[a,b,c] = 1.0
    return tensor

def _verify(U, V, W, tensor, tol=1e-9):
    r = U.shape[1]
    # Reconstruct tensor via einsum and compare
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    recon = np.einsum('at,bt,ct->abc', U, V, W)
    return bool(np.allclose(recon, tensor, atol=tol, rtol=0))

def _to_lists(U, V, W):
    def conv(mat):
        return [[int(x) if abs(x-round(x))<1e-9 else float(x) for x in row] for row in mat]
    return conv(U), conv(V), conv(W)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (rank 84)
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_sb

    def _deadline():
        return time.time() - start_time > TIME_LIMIT - 10

    # Block decomposition: 3x4x7
    # Split A rows: top 2 (2x4) and bottom 1 (1x4)
    # Split B rows: top 2 (2x7) and bottom 2 (2x7)
    # 
    # C[0:2,:] = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    #   = 2x2x7 + 2x2x7
    #   Each 2x2x7 = 2x2x2(7) + 2x2x2(7) + 2x2x2(7) + 2x2x1(4) = 25
    #   Total top: 50
    #
    # C[2,:] = A[2,0:2]*B[0:2,:] + A[2,2:4]*B[2:4,:]
    #   = 1x2x7 + 1x2x7
    #   Each 1x2x7 = 1x2x2(4) + 1x2x2(4) + 1x2x2(4) + 1x2x1(2) = 14
    #   Total bottom: 28
    #
    # Grand total: 78
    
    try:
        if _deadline():
            return {"U": best_U, "V": best_V, "W": best_W}
        Uss, Vss, Wss = _strassen_222()      # 2x2x2, rank 7
        U221, V221, W221 = _schoolbook_221()  # 2x2x1, rank 4
        U122, V122, W122 = _schoolbook_122()  # 1x2x2, rank 4
        U121, V121, W121 = _schoolbook_121()  # 1x2x1, rank 2
        
        total_rank = 78
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        
        def add_block(Ub, Vb, Wb, u_idx, v_idx, w_idx):
            nonlocal col
            r = Ub.shape[1]
            for t in range(r):
                for ri, val in enumerate(Ub[:, t]):
                    if val != 0:
                        U[u_idx[ri], col] += val
                for ri, val in enumerate(Vb[:, t]):
                    if val != 0:
                        V[v_idx[ri], col] += val
                for ri, val in enumerate(Wb[:, t]):
                    if val != 0:
                        W[w_idx[ri], col] += val
                col += 1
        
        # A indices: A[i,j] -> i*m+j = i*4+j
        # Top 2 rows of A, cols 0-1: [0,1,4,5]
        # Top 2 rows of A, cols 2-3: [2,3,6,7]
        # Bottom row of A, cols 0-1: [8,9]
        # Bottom row of A, cols 2-3: [10,11]
        
        u_top_L = [0, 1, 4, 5]   # A[0,0],A[0,1],A[1,0],A[1,1]
        u_top_R = [2, 3, 6, 7]   # A[0,2],A[0,3],A[1,2],A[1,3]
        u_bot_L = [8, 9]         # A[2,0],A[2,1]
        u_bot_R = [10, 11]       # A[2,2],A[2,3]
        
        # B indices: B[j,k] -> j*p+k = j*7+k
        # B rows 0-1, cols 0-1: [0,1,7,8]
        # B rows 0-1, cols 2-3: [2,3,9,10]
        # B rows 0-1, cols 4-5: [4,5,11,12]
        # B rows 0-1, col 6:    [6,13]
        # B rows 2-3, cols 0-1: [14,15,21,22]
        # B rows 2-3, cols 2-3: [16,17,23,24]
        # B rows 2-3, cols 4-5: [18,19,25,26]
        # B rows 2-3, col 6:    [20,27]
        
        v_Btop_00 = [0, 1, 7, 8]     # B[0,0],B[0,1],B[1,0],B[1,1]
        v_Btop_01 = [2, 3, 9, 10]    # B[0,2],B[0,3],B[1,2],B[1,3]
        v_Btop_02 = [4, 5, 11, 12]   # B[0,4],B[0,5],B[1,4],B[1,5]
        v_Btop_03 = [6, 13]          # B[0,6],B[1,6]
        v_Bbot_00 = [14, 15, 21, 22] # B[2,0],B[2,1],B[3,0],B[3,1]
        v_Bbot_01 = [16, 17, 23, 24] # B[2,2],B[2,3],B[3,2],B[3,3]
        v_Bbot_02 = [18, 19, 25, 26] # B[2,4],B[2,5],B[3,4],B[3,5]
        v_Bbot_03 = [20, 27]         # B[2,6],B[3,6]
        
        # C indices: C[i,k] -> k*n+i = k*3+i
        # C rows 0-1, cols 0-1: [0,1,3,4]  (C[0,0]=0,C[0,1]=1,C[1,0]=3,C[1,1]=4)
        # C rows 0-1, cols 2-3: [2,3,5,6]  (C[0,2]=2,C[0,3]=3,C[1,2]=5,C[1,3]=6)
        # C rows 0-1, cols 4-5: [4,5,7,8]  (C[0,4]=4,C[0,5]=5,C[1,4]=7,C[1,5]=8)
        # C rows 0-1, col 6:    [6,7]      (C[0,6]=6,C[1,6]=7)
        # C row 2, cols 0-1:    [12,13]    (C[2,0]=12,C[2,1]=13)
        # C row 2, cols 2-3:    [14,15]    (C[2,2]=14,C[2,3]=15)
        # C row 2, cols 4-5:    [16,17]    (C[2,4]=16,C[2,5]=17)
        # C row 2, col 6:       [18]       (C[2,6]=18)
        
        # W mapping for 2x2x2 Strassen: Wss[c_idx, t] where c_idx 0-3 maps to C positions
        # Wss uses C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        # So w_idx must map: 0->C[i0,k0], 1->C[i0,k1], 2->C[i1,k0], 3->C[i1,k1]
        
        w_Ctop_00 = [0, 3, 1, 4]     # C[0,0]=0,C[0,1]=3,C[1,0]=1,C[1,1]=4
        w_Ctop_01 = [2, 5, 3, 6]     # C[0,2]=2,C[0,3]=5,C[1,2]=3,C[1,3]=6
        w_Ctop_02 = [4, 7, 5, 8]     # C[0,4]=4,C[0,5]=7,C[1,4]=5,C[1,5]=8
        w_Ctop_03 = [6, 7]           # C[0,6]=6,C[1,6]=7
        w_Cbot_00 = [12, 13]         # C[2,0],C[2,1]
        w_Cbot_01 = [14, 15]         # C[2,2],C[2,3]
        w_Cbot_02 = [16, 17]         # C[2,4],C[2,5]
        w_Cbot_03 = [18]             # C[2,6]
        
        # Top 2 rows of C:
        # A_top_L * B_top -> C_top: 2x2x7
        add_block(Uss, Vss, Wss, u_top_L, v_Btop_00, w_Ctop_00)  # 7
        add_block(Uss, Vss, Wss, u_top_L, v_Btop_01, w_Ctop_01)  # 7
        add_block(Uss, Vss, Wss, u_top_L, v_Btop_02, w_Ctop_02)  # 7
        add_block(U221, V221, W221, u_top_L, v_Btop_03, w_Ctop_03)  # 4
        # A_top_R * B_bot -> C_top: 2x2x7
        add_block(Uss, Vss, Wss, u_top_R, v_Bbot_00, w_Ctop_00)  # 7
        add_block(Uss, Vss, Wss, u_top_R, v_Bbot_01, w_Ctop_01)  # 7
        add_block(Uss, Vss, Wss, u_top_R, v_Bbot_02, w_Ctop_02)  # 7
        add_block(U221, V221, W221, u_top_R, v_Bbot_03, w_Ctop_03)  # 4
        
        # Bottom row of C:
        # A_bot_L * B_top -> C_bot: 1x2x7
        add_block(U122, V122, W122, u_bot_L, v_Btop_00, w_Cbot_00)  # 4
        add_block(U122, V122, W122, u_bot_L, v_Btop_01, w_Cbot_01)  # 4
        add_block(U122, V122, W122, u_bot_L, v_Btop_02, w_Cbot_02)  # 4
        add_block(U121, V121, W121, u_bot_L, v_Btop_03, w_Cbot_03)  # 2
        # A_bot_R * B_bot -> C_bot: 1x2x7
        add_block(U122, V122, W122, u_bot_R, v_Bbot_00, w_Cbot_00)  # 4
        add_block(U122, V122, W122, u_bot_R, v_Bbot_01, w_Cbot_01)  # 4
        add_block(U122, V122, W122, u_bot_R, v_Bbot_02, w_Cbot_02)  # 4
        add_block(U121, V121, W121, u_bot_R, v_Bbot_03, w_Cbot_03)  # 2
        
        if col == total_rank:
            tensor = _build_tensor(n, m, p)
            if _verify(U, V, W, tensor):
                best_rank = col
                best_U, best_V, best_W = _to_lists(U, V, W)
    except Exception:
        pass

    if _deadline():
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}

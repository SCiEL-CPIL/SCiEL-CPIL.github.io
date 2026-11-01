"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def build_strassen_2x2x2(offset_u=0, offset_v=0, offset_w=0):
        """Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen rank 7.
        
        U shape: (4, 7) indexed [i*m+j, term]
        V shape: (4, 7) indexed [j*p+k, term]  where p=2 here
        W shape: (4, 7) indexed [k*n+i, term]  where n=2 here
        
        Strassen formulas:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B11)
        M4 = A11*(B01-B11)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B10)
        M7 = (A01-A11)*(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # U rows: [A00, A01, A10, A11] -> indices 0,1,2,3
        # V rows: [B00, B01, B10, B11] -> indices 0,1,2,3
        # W rows: [C00, C01, C10, C11] -> indices 0,1,2,3 (k*n+i: k=0,i=0 -> 0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
        
        # M3 = A00*(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
        
        # M4 = A11*(B01-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)*(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1  # C11 += M6
        
        # M7 = (A01-A11)*(B01+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00 += M7
        
        return U, V, W

    def build_schoolbook_2x2x1():
        """Return (U, V, W) for 2x2 * 2x1 -> 2x1 using rank 4 schoolbook.
        
        U shape: (4, 4) indexed [i*m+j, term]
        V shape: (2, 4) indexed [j*p+k, term]  where p=1
        W shape: (2, 4) indexed [k*n+i, term]  where n=2, k is always 0
        
        C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        """
        R = 4
        U = np.zeros((4, R))
        V = np.zeros((2, R))
        W = np.zeros((2, R))
        
        # Term 0: A00 * B00 -> C00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1  # k=0, i=0 -> index 0
        
        # Term 1: A01 * B10 -> C00
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1  # k=0, i=0 -> index 0
        
        # Term 2: A10 * B00 -> C10
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1  # k=0, i=1 -> index 1
        
        # Term 3: A11 * B10 -> C10
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1  # k=0, i=1 -> index 1
        
        return U, V, W

    # Build the full decomposition:
    # 7 blocks of 2 columns each (k=0,1), (k=2,3), ..., (k=12,13) using Strassen rank 7
    # 1 block of 1 column (k=14) using schoolbook rank 4
    # Total rank = 7*7 + 4 = 53
    
    total_rank = 7 * 7 + 4  # 53
    U_full = np.zeros((n * m, total_rank))  # (4, 53)
    V_full = np.zeros((m * p, total_rank))  # (2*15=30, 53)
    W_full = np.zeros((p * n, total_rank))  # (15*2=30, 53)
    
    col_offset = 0
    
    for block in range(7):
        k_start = 2 * block
        k_end = k_start + 1  # k_start, k_start+1
        
        Us, Vs, Ws = build_strassen_2x2x2()
        
        # U: same for all blocks (A matrix doesn't depend on k)
        U_full[:, col_offset:col_offset+7] = Us
        
        # V: V[j*p+k, term] where k is in {k_start, k_start+1}
        # V row index: j*p + k
        # For j=0: rows k_start, k_start+1 -> V[0*p+k_start, :] and V[0*p+k_start+1, :]
        # For j=1: rows p+k_start, p+k_start+1
        # Vs has rows: [B00, B01, B10, B11] = [V[0,0], V[0,1], V[1,0], V[1,1]] for local k=0,1
        # So Vs[0,:] -> V[0*p+k_start, :], Vs[1,:] -> V[0*p+k_start+1, :]
        #    Vs[2,:] -> V[1*p+k_start, :], Vs[3,:] -> V[1*p+k_start+1, :]
        V_full[0 * p + k_start, col_offset:col_offset+7] = Vs[0, :]
        V_full[0 * p + k_start + 1, col_offset:col_offset+7] = Vs[1, :]
        V_full[1 * p + k_start, col_offset:col_offset+7] = Vs[2, :]
        V_full[1 * p + k_start + 1, col_offset:col_offset+7] = Vs[3, :]
        
        # W: W[k*n+i, term]
        # Ws has rows: [C00, C01, C10, C11] = [W[0,0], W[0,1], W[1,0], W[1,1]] for local k=0,1
        # So Ws[0,:] -> W[k_start*n+0, :], Ws[1,:] -> W[k_start*n+1, :]
        #    Ws[2,:] -> W[k_start+1*n+0, :], Ws[3,:] -> W[k_start+1*n+1, :]
        W_full[k_start * n + 0, col_offset:col_offset+7] = Ws[0, :]
        W_full[k_start * n + 1, col_offset:col_offset+7] = Ws[1, :]
        W_full[k_start + 1, col_offset:col_offset+7] = Ws[2, :]  # (k_start+1)*n+0 = k_start+1 since n=2? No!
        # Wait: W index is k*n+i. For k=k_start+1, i=0: index = (k_start+1)*2+0 = 2*k_start+2
        # For k=k_start+1, i=1: index = (k_start+1)*2+1 = 2*k_start+3
        # Let me redo:
        W_full[(k_start) * n + 0, col_offset:col_offset+7] = Ws[0, :]  # k=k_start, i=0
        W_full[(k_start) * n + 1, col_offset:col_offset+7] = Ws[1, :]  # k=k_start, i=1
        W_full[(k_start + 1) * n + 0, col_offset:col_offset+7] = Ws[2, :]  # k=k_start+1, i=0
        W_full[(k_start + 1) * n + 1, col_offset:col_offset+7] = Ws[3, :]  # k=k_start+1, i=1
        
        col_offset += 7
    
    # Last block: k=14 (single column)
    Us, Vs, Ws = build_schoolbook_2x2x1()
    
    U_full[:, col_offset:col_offset+4] = Us
    
    # V: V[j*p+k, term] where k=14
    # Vs has rows: [B00, B10] = [V[0,0], V[1,0]] for local k=0 (only one k)
    V_full[0 * p + 14, col_offset:col_offset+4] = Vs[0, :]
    V_full[1 * p + 14, col_offset:col_offset+4] = Vs[1, :]
    
    # W: W[k*n+i, term] where k=14
    # Ws has rows: [C00, C10] = [W[0,0], W[1,0]] for local k=0
    W_full[14 * n + 0, col_offset:col_offset+4] = Ws[0, :]
    W_full[14 * n + 1, col_offset:col_offset+4] = Ws[1, :]
    
    # Verify the decomposition
    # Tensor T[i,j,k] should equal 1 for all i,j,k
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    val += U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
                T_reconstructed[i, j, k] = val
    
    # Check if all entries are 1
    if np.allclose(T_reconstructed, 1.0, atol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook rank 60
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END

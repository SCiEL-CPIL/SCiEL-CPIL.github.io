import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Fallback: Schoolbook decomposition (Rank 64)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Candidate: Rank 56 Strassen-based decomposition
    # Strategy: 
    # A (2x4) = [A_L, A_R] where A_L, A_R are 2x2
    # B (4x8) = [B_0, B_1, B_2, B_3] where B_g is 4x2
    # C (2x8) = [C_0, C_1, C_2, C_3] where C_g is 2x2
    # C_g = A_L * B_g[0:2, :] + A_R * B_g[2:4, :]
    # Each term is a 2x2 * 2x2 multiplication.
    # Using Strassen (rank 7) for each 2x2*2x2 gives rank 14 per C_g.
    # Total rank = 4 * 14 = 56.

    def strassen_rank56():
        total_rank = 56
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Helper to add a Strassen 2x2*2x2 decomposition
        # a_indices: indices of the 4 elements of the first 2x2 matrix (row-major)
        # b_indices: indices of the 4 elements of the second 2x2 matrix (row-major)
        # c_indices: indices of the 4 elements of the resulting 2x2 matrix (row-major)
        # base_col: starting column in U, V, W
        def add_strassen(a_idx, b_idx, c_idx, base_col):
            # a_idx = [a00, a01, a10, a11]
            # b_idx = [b00, b01, b10, b11]
            # c_idx = [c00, c01, c10, c11]
            
            a00, a01, a10, a11 = a_idx
            b00, b01, b10, b11 = b_idx
            c00, c01, c10, c11 = c_idx
            
            # M1 = (A00+A11)(B00+B11)
            U[a00, base_col] += 1
            U[a11, base_col] += 1
            V[b00, base_col] += 1
            V[b11, base_col] += 1
            W[c00, base_col] += 1
            W[c11, base_col] += 1
            
            # M2 = (A10+A11)B00
            U[a10, base_col + 1] += 1
            U[a11, base_col + 1] += 1
            V[b00, base_col + 1] += 1
            W[c10, base_col + 1] += 1
            W[c11, base_col + 1] -= 1
            
            # M3 = A00(B10-B11)
            U[a00, base_col + 2] += 1
            V[b10, base_col + 2] += 1
            V[b11, base_col + 2] -= 1
            W[c01, base_col + 2] += 1
            W[c11, base_col + 2] += 1
            
            # M4 = A11(B01-B10)
            U[a11, base_col + 3] += 1
            V[b01, base_col + 3] += 1
            V[b10, base_col + 3] -= 1
            W[c00, base_col + 3] += 1
            W[c10, base_col + 3] += 1
            
            # M5 = (A00+A01)B11
            U[a00, base_col + 4] += 1
            U[a01, base_col + 4] += 1
            V[b11, base_col + 4] += 1
            W[c00, base_col + 4] -= 1
            W[c01, base_col + 4] += 1
            
            # M6 = (A10-A00)(B00+B10)
            U[a10, base_col + 5] += 1
            U[a00, base_col + 5] -= 1
            V[b00, base_col + 5] += 1
            V[b10, base_col + 5] += 1
            W[c11, base_col + 5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[a01, base_col + 6] += 1
            U[a11, base_col + 6] -= 1
            V[b10, base_col + 6] += 1
            V[b11, base_col + 6] += 1
            W[c00, base_col + 6] += 1

        # Loop over the 4 blocks of B (and corresponding C columns)
        for g in range(4):
            # Block g corresponds to columns 2g, 2g+1 of B and C
            
            # --- Term 1: A_L * B_g_top ---
            # A_L is A[:, 0:2] -> indices in U (n*m=8, row-major i*m+j):
            # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
            a_L_idx = [0, 1, 4, 5]
            
            # B_g_top is B[0:2, 2g:2g+2] -> indices in V (m*p=32, row-major j*p+k):
            # B[0, 2g] = 0*8 + 2g = 2g
            # B[0, 2g+1] = 0*8 + 2g+1 = 2g+1
            # B[1, 2g] = 1*8 + 2g = 8+2g
            # B[1, 2g+1] = 1*8 + 2g+1 = 8+2g+1
            b_top_idx = [2*g, 2*g+1, 8+2*g, 8+2*g+1]
            
            # C_g is C[:, 2g:2g+2] -> indices in W (p*n=16, row-major k*n+i):
            # C[0, 2g] = 2g*2 + 0 = 4g
            # C[0, 2g+1] = (2g+1)*2 + 0 = 4g+2
            # C[1, 2g] = 2g*2 + 1 = 4g+1
            # C[1, 2g+1] = (2g+1)*2 + 1 = 4g+3
            # Note: W index is k*n + i.
            # C[k, i]. 
            # For C_g (2x2), rows are k=2g, 2g+1. Cols are i=0, 1.
            # C[2g, 0] -> W[4g]
            # C[2g, 1] -> W[4g+1]
            # C[2g+1, 0] -> W[4g+2]
            # C[2g+1, 1] -> W[4g+3]
            # So c_idx = [4g, 4g+2, 4g+1, 4g+3] corresponding to c00, c01, c10, c11?
            # Wait.
            # c00 is C[0,0] of the block -> C[2g, 0] -> W[4g]
            # c01 is C[0,1] of the block -> C[2g, 1] -> W[4g+1]
            # c10 is C[1,0] of the block -> C[2g+1, 0] -> W[4g+2]
            # c11 is C[1,1] of the block -> C[2g+1, 1] -> W[4g+3]
            c_g_idx = [4*g, 4*g+1, 4*g+2, 4*g+3]
            
            # Add Strassen for A_L * B_g_top
            add_strassen(a_L_idx, b_top_idx, c_g_idx, g * 14)
            
            # --- Term 2: A_R * B_g_bottom ---
            # A_R is A[:, 2:4] -> indices in U:
            # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
            a_R_idx = [2, 3, 6, 7]
            
            # B_g_bottom is B[2:4, 2g:2g+2] -> indices in V:
            # B[2, 2g] = 2*8 + 2g = 16+2g
            # B[2, 2g+1] = 2*8 + 2g+1 = 16+2g+1
            # B[3, 2g] = 3*8 + 2g = 24+2g
            # B[3, 2g+1] = 3*8 + 2g+1 = 24+2g+1
            b_bot_idx = [16+2*g, 16+2*g+1, 24+2*g, 24+2*g+1]
            
            # C_g is the same C block, so same c_g_idx.
            # We add the contributions to the SAME C indices, but using different U/V indices and different Strassen columns.
            # The Strassen columns for this term should be offset by 7.
            add_strassen(a_R_idx, b_bot_idx, c_g_idx, g * 14 + 7)

        return U, V, W

    # Verification function
    def is_valid(U, V, W, rank):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Try Rank 56
    U56, V56, W56 = strassen_rank56()
    if is_valid(U56, V56, W56, 56):
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    else:
        # Fallback to Schoolbook
        Us, Vs, Ws = schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
# EVOLVE-BLOCK-END

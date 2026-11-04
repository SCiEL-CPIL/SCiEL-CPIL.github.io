import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution (Rank 54)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 140
    
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Build target tensor for 2x3x9
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Search for a low-rank decomposition of the sub-block 2x3x3
    # If found, replicate it 3 times to get a decomposition for 2x3x9.
    # 2x3x3 dimensions: nm3=6, mp3=9, pn3=9
    nm3, mp3, pn3 = 6, 9, 9
    T3 = np.zeros((nm3, mp3, pn3))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                T3[i*3+j, j*3+k, k*2+i] = 1.0

    def find_233_decomp(rank):
        # Search for U3, V3, W3 of rank 'rank' for T3
        for trial in range(500):
            if time.time() > deadline - 5:
                return None
            np.random.seed(100 + rank * 1000 + trial)
            V3 = np.random.randint(-1, 2, size=(mp3, rank))
            W3 = np.random.randint(-1, 2, size=(pn3, rank))
            
            M3 = (V3[:, None, :] * W3[None, :, :]).reshape(mp3 * pn3, rank)
            if np.linalg.matrix_rank(M3) < rank:
                continue
                
            U3 = np.zeros((nm3, rank))
            valid = True
            for a in range(nm3):
                target = T3[a, :, :].flatten()
                try:
                    sol, residuals, r, sv = np.linalg.lstsq(M3, target, rcond=None)
                    if residuals and residuals[0] > 1e-9:
                        valid = False
                        break
                    U3[a, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            if is_half_integer(U3):
                U3_snapped = np.round(2.0 * U3) / 2.0
                if verify_solution(U3_snapped, V3, W3, T3):
                    return U3_snapped, V3, W3
        return None

    # Try ranks 16, 17, 18 for the 2x3x3 block
    # Rank 18 is schoolbook for 2x3x3, which we already have.
    # We want < 18 to get < 54 total.
    for r3 in range(16, 18):
        if time.time() > deadline - 10:
            break
            
        result = find_233_decomp(r3)
        if result:
            U3, V3, W3 = result
            # Construct 2x3x9 solution by block duplication
            # U_total: 6 x (3*r3)
            # V_total: 27 x (3*r3)
            # W_total: 18 x (3*r3)
            
            total_rank = 3 * r3
            U_tot = np.zeros((nm, total_rank))
            V_tot = np.zeros((mp, total_rank))
            W_tot = np.zeros((pn, total_rank))
            
            # Block 0: k=0,1,2
            # V rows: j*9 + k. For k in 0..2, rows are j*9, j*9+1, j*9+2
            # W rows: k*2 + i. For k in 0..2, rows are 0..5
            # U rows: i*3 + j. Same for all blocks.
            
            for block in range(3):
                U_tot[:, block*r3:(block+1)*r3] = U3
                
                # V mapping
                # V3 is 9x r3. Rows correspond to j*3+k for j in 0..2, k in 0..2.
                # V_tot is 27x total_rank. Rows correspond to j*9+k for j in 0..2, k in 0..8.
                # For block 'block' (k_offset = block*3), the k in V3 (0..2) maps to k_total = block*3 + k_v3.
                # Row in V3: j*3 + k_v3.
                # Row in V_tot: j*9 + (block*3 + k_v3).
                for j in range(3):
                    for k_v3 in range(3):
                        row_v3 = j*3 + k_v3
                        row_v_tot = j*9 + (block*3 + k_v3)
                        V_tot[row_v_tot, block*r3:(block+1)*r3] = V3[row_v3, :]
                        
                # W mapping
                # W3 is 9x r3. Rows correspond to k*2+i for k in 0..2, i in 0..1.
                # W_tot is 18x total_rank. Rows correspond to k*2+i for k in 0..8, i in 0..1.
                # For block 'block' (k_offset = block*3), the k in W3 (0..2) maps to k_total = block*3 + k_w3.
                # Row in W3: k_w3*2 + i.
                # Row in W_tot: (block*3 + k_w3)*2 + i.
                for k_w3 in range(3):
                    for i in range(2):
                        row_w3 = k_w3*2 + i
                        row_w_tot = (block*3 + k_w3)*2 + i
                        W_tot[row_w_tot, block*r3:(block+1)*r3] = W3[row_w3, :]
                        
            # Verify the full solution
            if verify_solution(U_tot, V_tot, W_tot, T):
                best_rank = total_rank
                best_U, best_V, best_W = U_tot, V_tot, W_tot
                break # Found a lower rank, stop searching lower blocks? 
                # Actually, if we found rank 16 for 2x3x3, we get 48.
                # If we found rank 17, we get 51.
                # We search 16 first. If found, we break.
                
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}

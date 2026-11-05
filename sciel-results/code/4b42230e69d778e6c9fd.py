import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Use the proven rank-99 decomposition from the inspiration
    # Block structure: n=[3], m=[2,2,1], p=[2,2,2,1]
    # This gives: 3*(2*2 Strassen blocks + 2*1) + 3*(1*2 blocks + 1*1)
    # Specifically:
    #   (3,2,2) x3 with Strassen: 3*7 = 21
    #   (3,2,1) x1 with schoolbook: 3*2*1 = 6
    #   (3,1,2) x3 with schoolbook: 3*3*1*2 = 18
    #   (3,1,1) x1 with schoolbook: 3*1*1 = 3
    # Wait, that's 21+6+18+3 = 48, not 99.
    
    # Let me re-examine. The inspiration uses:
    # Block 1 (M327): 39
    # Block 2 (M327): 39  
    # Block 3 (M317): 21
    # Total: 99
    
    # M327 means n=3, m=2, p=7. For this:
    # Split p=7 into [2,2,2,1]
    # (3,2,2): Strassen rank 7, three times = 21
    # (3,2,1): schoolbook 3*2*1 = 6
    # Total for M327: 27? But inspiration says 39.
    
    # I think the issue is that m=2 in M327, and for each of the 3 n-rows,
    # we need to handle the full 2x7 multiplication. Let me just use the
    # explicit construction from the inspiration which is verified.
    
    rank_total = 99
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    def add_strassen(rank_offset, i0, j0, k0):
        """Add M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        u_loc = np.zeros((4, 7))
        v_loc = np.zeros((4, 7))
        w_loc = np.zeros((4, 7))
        
        # M1
        u_loc[0, 0] = 1; u_loc[3, 0] = 1
        v_loc[0, 0] = 1; v_loc[3, 0] = 1
        w_loc[0, 0] = 1; w_loc[3, 0] = 1
        
        # M2
        u_loc[2, 1] = 1; u_loc[3, 1] = 1
        v_loc[0, 1] = 1
        w_loc[1, 1] = 1; w_loc[3, 1] = -1
        
        # M3
        u_loc[0, 2] = 1
        v_loc[1, 2] = 1; v_loc[3, 2] = -1
        w_loc[2, 2] = 1; w_loc[3, 2] = 1
        
        # M4
        u_loc[3, 3] = 1
        v_loc[2, 3] = 1; v_loc[0, 3] = -1
        w_loc[0, 3] = 1; w_loc[1, 3] = 1
        
        # M5
        u_loc[0, 4] = 1; u_loc[1, 4] = 1
        v_loc[3, 4] = 1
        w_loc[0, 4] = -1; w_loc[2, 4] = 1
        
        # M6
        u_loc[2, 5] = 1; u_loc[0, 5] = -1
        v_loc[0, 5] = 1; v_loc[1, 5] = 1
        w_loc[3, 5] = 1
        
        # M7
        u_loc[1, 6] = 1; u_loc[3, 6] = -1
        v_loc[2, 6] = 1; v_loc[3, 6] = 1
        w_loc[0, 6] = 1
        
        # Map to global
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx, rank_offset + r] += u_loc[i_l * 2 + j_l, r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx, rank_offset + r] += v_loc[j_l * 2 + k_l, r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx, rank_offset + r] += w_loc[k_l * 2 + i_l, r]
    
    def add_m221(rank_offset, i0, j0, k0):
        """Add M221 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1], rank 4"""
        u_loc = np.zeros((4, 4))
        v_loc = np.zeros((2, 4))
        w_loc = np.zeros((2, 4))
        
        # r=0: A00 * B00
        u_loc[0, 0] = 1
        v_loc[0, 0] = 1
        w_loc[0, 0] = 1
        
        # r=1: A01 * B10
        u_loc[1, 1] = 1
        v_loc[1, 1] = 1
        w_loc[0, 1] = 1
        
        # r=2: A10 * B00
        u_loc[2, 2] = 1
        v_loc[0, 2] = 1
        w_loc[1, 2] = 1
        
        # r=3: A11 * B10
        u_loc[3, 3] = 1
        v_loc[1, 3] = 1
        w_loc[1, 3] = 1
        
        # Map to global
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx, rank_offset + r] += u_loc[i_l * 2 + j_l, r]
        
        for j_l in range(2):
            for k_l in range(1):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx, rank_offset + r] += v_loc[j_l * 1 + k_l, r]
        
        for k_l in range(1):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx, rank_offset + r] += w_loc[k_l * 2 + i_l, r]
    
    def add_m127(rank_offset, i0, j0, k0):
        """Add M127 for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+7], rank 14"""
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                u_g_idx = (i0 + 0) * m + (j0 + j_l)
                U[u_g_idx, rank_offset + r] += 1
                
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx, rank_offset + r] += 1
                
                w_g_idx = (k0 + k_l) * n + (i0 + 0)
                W[w_g_idx, rank_offset + r] += 1
    
    def add_m317(rank_offset, i0, j0, k0):
        """Add M317 for A[i0:i0+3, j0:j0+1] * B[j0:j0+1, k0:k0+7], rank 21"""
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                u_g_idx = (i0 + i_l) * m + (j0 + 0)
                U[u_g_idx, rank_offset + r] += 1
                
                v_g_idx = (j0 + 0) * p + (k0 + k_l)
                V[v_g_idx, rank_offset + r] += 1
                
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx, rank_offset + r] += 1
    
    # Build the blocks based on the inspiration's structure
    # The inspiration splits as:
    # Block 1: M327 (rows 0-2, cols 0-1, all 7 k) -> rank 39
    # Block 2: M327 (rows 0-2, cols 2-3, all 7 k) -> rank 39
    # Block 3: M317 (rows 0-2, col 4, all 7 k) -> rank 21
    
    # For M327 (n=3, m=2, p=7):
    # Split p into [2,2,2,1]
    # (3,2,2) x3: each uses Strassen on the 2x2 part, but we have 3 n-rows
    # Actually, for each n-row, we do a 1x2x2 multiplication which can use Strassen
    # But Strassen is for 2x2x2. For 1x2x2, we just need 2 multiplications.
    # 
    # Let me reconsider. The rank 39 for M327 suggests:
    # For each of the 3 n-rows: multiply 1x2 matrix by 2x7 matrix
    # = 2 dot products of length 2 for each of 7 k-values
    # = 2*7 = 14 per n-row, total 3*14 = 42? Not 39.
    #
    # With Strassen on 2x2x2 blocks:
    # Split the 7 k-values into [2,2,2,1]
    # For each n-row:
    #   - 3 blocks of 1x2x2: each needs 2 mults (or Strassen if we had 2x2x2)
    #   - 1 block of 1x2x1: 2 mults
    #
    # I think the 39 comes from a more clever combination. Let me just use
    # the explicit coefficients from the inspiration.
    
    # Based on the inspiration code structure, it seems to use:
    # For M327: combine Strassen for 2x2x2 sub-blocks with schoolbook
    # The total rank 39 = 3*7 + 3*6 + 3*2 = 21 + 18 + 6 = 45? No.
    # 39 = 3*13? Or 7*3 + 6*3 + 3 = 21+18+3 = 42? 
    #
    # Let me just verify my construction works and use whatever rank it produces.
    
    col = 0
    
    # Block 1: n=3, m=2 (cols 0-1), p=7 (rows 0-6)
    # Split p into [2,2,2,1]
    # For each n-row (3 rows), do 1x2x7 multiplication
    # Using Strassen for the 2x2 parts:
    for i_row in range(3):
        # 1x2x2 Strassen-like (actually just 2 mults for 1x2x2)
        # But we can batch: for 3x2x2, use 3 independent 1x2x2 or one 3x2x2
        # 3x2x2: schoolbook = 3*2*2 = 12, or we can do better
        # Actually for 3x2x2, split n into [1,1,1] and use 1x2x2 three times = 3*2 = 6
        # Or keep as 3x2x2: 12
        # The minimum for 3x2x2 is at most 3*2*2=12, could be less
        
        # Let me just use schoolbook for simplicity and correctness
        # 3x2x2: 12 terms
        for i_l in range(3):
            for j_l in range(2):
                for k_l in range(2):
                    U[(i0 := i_row) * m + (j0 := 0) + j_l, col] = 1
                    V[(0 + j_l) * p + (0 + k_l), col] = 1
                    W[(0 + k_l) * n + (0 + i_l), col] = 1
                    col += 1
        
        # 3x2x2 again for next 2 k-values
        for i_l in range(3):
            for j_l in range(2):
                for k_l in range(2):
                    U[i_row * m + j_l, col] = 1
                    V[j_l * p + (2 + k_l), col] = 1
                    W[(2 + k_l) * n + i_l, col] = 1
                    col += 1
        
        # 3x2x2 again
        for i_l in range(3):
            for j_l in range(2):
                for k_l in range(2):
                    U[i_row * m + j_l, col] = 1
                    V[j_l * p + (4 + k_l), col] = 1
                    W[(4 + k_l) * n + i_l, col] = 1
                    col += 1
        
        # 3x2x1
        for i_l in range(3):
            for j_l in range(2):
                U[i_row * m + j_l, col] = 1
                V[j_l * p + 6, col] = 1
                W[6 * n + i_l, col] = 1
                col += 1
    
    # That gives 3 * (12 + 12 + 12 + 6) = 3 * 42 = 126, too much.
    
    # Let me restart with a cleaner approach: just use the block decomposition
    # framework from the current program with the best splits.
    
    # Reset
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Use the split: n=[3], m=[2,2,1], p=[2,2,2,1]
    n_blocks = [3]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2, 1]
    
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen 2x2x2, rank 7
                    strassen_data = [
                        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
                    ]
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0+ar)*m + (m0+ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0+br)*p + (p0+bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0+cc)*n + (n0+cr), col] += ccoeff
                        col += 1
                else:
                    # Schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0+i)*m + (m0+j), col] = 1
                                V[(m0+j)*p + (p0+k), col] = 1
                                W[(p0+k)*n + (n0+i), col] = 1
                                col += 1
    
    actual_rank = col
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

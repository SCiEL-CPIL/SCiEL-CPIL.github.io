import time
import numpy as np

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: the rank-66 block-Strassen decomposition
    # M(3,4,6) = 3 * [M(1,4,2) + M(2,4,2)] = 3 * [8 + 14] = 66
    U_fallback = [[0] * 66 for _ in range(n * m)]
    V_fallback = [[0] * 66 for _ in range(m * p)]
    W_fallback = [[0] * 66 for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1_fb(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= 66:
            return
        for idx, val in zip(u_indices, u_vals):
            U_fallback[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V_fallback[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W_fallback[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2_fb(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1_fb([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2_fb(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1_fb([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1_fb([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1_fb([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1_fb([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1_fb([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1_fb([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1_fb([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2_fb(i_off, j_off, k_off):
        add_2_2_2_fb(i_off, j_off, k_off)
        add_2_2_2_fb(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2_fb(0, 0, k_off)
        add_2_4_2_fb(1, 0, k_off)
    
    # Check if we have time for the 32-rank attempt
    if time.time() - START_TIME > 5.0:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    # Attempt 32-rank decomposition:
    # M(3,4,6) = M(3,3,6) + M(3,1,6) (splitting m=4 into 3+1)
    # M(3,3,6) = 2 * M(3,3,3) (splitting p=6 into 3+3)
    # M(3,1,6) = 2 * M(3,1,3) (splitting p=6 into 3+3)
    # M(3,3,3) rank 12 (Laderman)
    # M(3,1,3) rank 5 (Winograd-like or linear)
    # Total: 2*12 + 2*5 = 34. Wait, I calculated 32 before. Let's re-evaluate.
    # M(3,1,6) rank is n+p-1 = 3+6-1 = 8.
    # So M(3,4,6) = M(3,3,6) + M(3,1,6).
    # M(3,3,6) rank? 2 * M(3,3,3) = 24.
    # M(3,1,6) rank 8.
    # Total 24 + 8 = 32.
    
    # Let's construct the 32-rank solution.
    U_32 = [[0] * 32 for _ in range(n * m)]
    V_32 = [[0] * 32 for _ in range(m * p)]
    W_32 = [[0] * 32 for _ in range(p * n)]
    r = 0
    
    def add_rank1_32(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal r
        if r >= 32:
            return
        for idx, val in zip(u_indices, u_vals):
            U_32[idx][r] += val
        for idx, val in zip(v_indices, v_vals):
            V_32[idx][r] += val
        for idx, val in zip(w_indices, w_vals):
            W_32[idx][r] += val
        r += 1
    
    # Laderman's algorithm for 3x3x3
    # T[i,j,k] = 1 for i,j,k in {0,1,2}
    # U indices: i*3+k
    # V indices: k*3+j
    # W indices: j*3+i
    
    def add_laderman(i_off, j_off, k_off):
        # 12 terms
        # 1. U[i0,k0], V[k0,j0], W[j0,i0]
        add_rank1_32([(i_off+0)*m + (k_off+0)], [1], [(k_off+0)*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+0)], [1])
        # 2. U[i0,k1], V[k1,j0], W[j0,i0]
        add_rank1_32([(i_off+0)*m + (k_off+1)], [1], [(k_off+1)*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+0)], [1])
        # 3. U[i1,k0], V[k0,j1], W[j1,i1]
        add_rank1_32([(i_off+1)*m + (k_off+0)], [1], [(k_off+0)*p + (j_off+1)], [1], [(j_off+1)*n + (i_off+1)], [1])
        # 4. U[i1,k1], V[k1,j1], W[j1,i1]
        add_rank1_32([(i_off+1)*m + (k_off+1)], [1], [(k_off+1)*p + (j_off+1)], [1], [(j_off+1)*n + (i_off+1)], [1])
        # 5. U[i2,k2], V[k2,j2], W[j2,i2]
        add_rank1_32([(i_off+2)*m + (k_off+2)], [1], [(k_off+2)*p + (j_off+2)], [1], [(j_off+2)*n + (i_off+2)], [1])
        # 6. U[i0,k2], V[k2,j0], W[j0,i0]
        add_rank1_32([(i_off+0)*m + (k_off+2)], [1], [(k_off+2)*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+0)], [1])
        # 7. U[i2,k0], V[k0,j2], W[j2,i2]
        add_rank1_32([(i_off+2)*m + (k_off+0)], [1], [(k_off+0)*p + (j_off+2)], [1], [(j_off+2)*n + (i_off+2)], [1])
        # 8. U[i0,k0+k1], V[k0+k1, j0+j1], W[j0+j1, i0+i1]
        # This is getting complicated to list manually. Let's use a known Laderman decomposition.
        # Actually, let's just use the schoolbook for 3x3x3 which is rank 9? No, rank 12 is Laderman.
        # Schoolbook is 27. Laderman is 12.
        # Let's try to find a simpler way or just use the known Laderman terms.
        # Laderman terms for 3x3x3:
        # 1. (0,0) (0,0) (0,0)
        # 2. (0,1) (1,0) (0,0)
        # 3. (1,0) (0,1) (1,1)
        # 4. (1,1) (1,1) (1,1)
        # 5. (2,2) (2,2) (2,2)
        # 6. (0,2) (2,0) (0,0)
        # 7. (2,0) (0,2) (2,2)
        # 8. (0,0+1) (0+1, 0+1) (0+1, 0+1) -> (0,1) (1,1) (1,1) ?? No.
        
        # Let's use a verified Laderman implementation.
        # U:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (1,0)
        # 3: (1,1)
        # 4: (2,2)
        # 5: (0,2)
        # 6: (2,0)
        # 7: (0,0+1) = (0,1) ?? No, (0,0) and (0,1) are separate.
        # Laderman uses:
        # M1 = A00 B00
        # M2 = A01 B10
        # M3 = A10 B01
        # M4 = A11 B11
        # M5 = A22 B22
        # M6 = A02 B20
        # M7 = A20 B02
        # M8 = (A00+A01) (B00+B01) - M1 - M2 - M3 ... this is not rank 1.
        # Laderman's 12 terms:
        # 1. A00 B00
        # 2. A01 B10
        # 3. A10 B01
        # 4. A11 B11
        # 5. A22 B22
        # 6. A02 B20
        # 7. A20 B02
        # 8. (A00+A01) (B00+B01)
        # 9. (A10+A11) (B01+B11)
        # 10. (A00+A10) (B00+B10)
        # 11. (A01+A11) (B01+B11)
        # 12. (A00+A11+A22) (B00+B11+B22) ... this is not right.
        
        # Let's just use the schoolbook for 3x3x3 to be safe, rank 27.
        # Then total rank = 2*27 + 8 = 62. Still better than 66.
        # But I want 32.
        # Let's use the known Laderman decomposition from a reliable source.
        # Laderman's algorithm for 3x3x3:
        # U:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (1,0)
        # 3: (1,1)
        # 4: (2,2)
        # 5: (0,2)
        # 6: (2,0)
        # 7: (0,0+1) -> (0,1) is already there.
        # The 12 terms are:
        # 1. (0,0) (0,0) (0,0)
        # 2. (0,1) (1,0) (0,0)
        # 3. (1,0) (0,1) (1,1)
        # 4. (1,1) (1,1) (1,1)
        # 5. (2,2) (2,2) (2,2)
        # 6. (0,2) (2,0) (0,0)
        # 7. (2,0) (0,2) (2,2)
        # 8. (0,0+1) (0+1, 0+1) (0+1, 0+1) -> (0,1) (1,1) (1,1)
        # 9. (1,0+1) (0+1, 1+1) (1+1, 1+1) -> (1,1) (1,1) (1,1) duplicate.
        
        # I will use the schoolbook for 3x3x3 (rank 27) to ensure correctness.
        # Total rank = 2*27 + 8 = 62.
        
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    add_rank1_32([(i_off+i)*m + (k_off+k)], [1], [(k_off+k)*p + (j_off+j)], [1], [(j_off+j)*n + (i_off+i)], [1])
    
    # M(3,1,3) rank 5? No, rank 5 is for 2x2x2.
    # M(3,1,3) rank is 3+3-1 = 5.
    # Winograd-like for M(3,1,3):
    # T[i,j,0] = 1 for i in 0..2, j in 0..2.
    # U[i, r], V[j, r], W[j*3+i, r].
    # 5 terms.
    
    def add_3_1_3(i_off, j_off):
        # 1. (0,0)
        add_rank1_32([(i_off+0)*m + 3], [1], [3*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+0)], [1])
        # 2. (1,0)
        add_rank1_32([(i_off+1)*m + 3], [1], [3*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+1)], [1])
        # 3. (2,0)
        add_rank1_32([(i_off+2)*m + 3], [1], [3*p + (j_off+0)], [1], [(j_off+0)*n + (i_off+2)], [1])
        # 4. (0,1)
        add_rank1_32([(i_off+0)*m + 3], [1], [3*p + (j_off+1)], [1], [(j_off+1)*n + (i_off+0)], [1])
        # 5. (1,1)
        add_rank1_32([(i_off+1)*m + 3], [1], [3*p + (j_off+1)], [1], [(j_off+1)*n + (i_off+1)], [1])
        # 6. (2,1)
        add_rank1_32([(i_off+2)*m + 3], [1], [3*p + (j_off+1)], [1], [(j_off+1)*n + (i_off+2)], [1])
        # 7. (0,2)
        add_rank1_32([(i_off+0)*m + 3], [1], [3*p + (j_off+2)], [1], [(j_off+2)*n + (i_off+0)], [1])
        # 8. (1,2)
        add_rank1_32([(i_off+1)*m + 3], [1], [3*p + (j_off+2)], [1], [(j_off+2)*n + (i_off+1)], [1])
        # 9. (2,2)
        add_rank1_32([(i_off+2)*m + 3], [1], [3*p + (j_off+2)], [1], [(j_off+2)*n + (i_off+2)], [1])
        # This is rank 9. I need rank 5.
        # Winograd for M(3,1,3):
        # U: [1,1,1], [1,1,0], [1,0,0]
        # V: [1,1,1], [1,0,0], [0,1,0]
        # This is getting too complex to derive on the fly.
        # I will use the schoolbook for M(3,1,3) which is rank 9.
        # Total rank = 2*27 + 2*9 = 72. Worse than 66.
        
        # Let's stick with the 66-rank solution.
        pass
    
    # Since I cannot easily derive the 32-rank solution, I will return the 66-rank solution.
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

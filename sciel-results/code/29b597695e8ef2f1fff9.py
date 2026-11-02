"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Uses a structured decomposition based on Strassen's algorithm for 2x2x2 sub-blocks.
    Target rank: 68.
    """
    start_time = time.time()
    DEADLINE = 140
    
    def check_time():
        return time.time() - start_time < DEADLINE

    def verify_full(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook_full():
        """Return the schoolbook decomposition with rank n*m*p = 72."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen_2x2x2(offset_u, offset_v, offset_w, rank_start):
        """
        Adds the Strassen decomposition for a 2x2x2 block to the global arrays.
        Returns the next rank index.
        """
        # Strassen factors for 2x2x2
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # U rows: offset_u + i*2 + j
        # V rows: offset_v + j*2 + k
        # W rows: offset_w + k*2 + i
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to populate U, V, W such that sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c]
        
        # Let's define the contributions to each rank term.
        # We will add 7 columns to U, V, W.
        
        # Term 1: (A00+A11) * (B00+B11) -> C00, C11
        # U contributions:
        # A00: i=0, j=0. U[0,0] = 1
        # A11: i=1, j=1. U[1,1] = 1
        # V contributions:
        # B00: j=0, k=0. V[0,0] = 1
        # B11: j=1, k=1. V[1,1] = 1
        # W contributions:
        # C00: k=0, i=0. W[0,0] = 1
        # C11: k=1, i=1. W[1,1] = 1
        
        # This is getting complicated to do manually in code without a general framework.
        # Let's use the explicit list of (u_idx, v_idx, w_idx, coeff) for Strassen.
        
        # Strassen 2x2x2 decomposition:
        # 7 multiplications.
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We represent the tensor as sum of rank-1 terms.
        # Each rank-1 term is defined by a triple (u_vec, v_vec, w_vec).
        # For Strassen, we can define 7 such triples.
        
        # Let's define the vectors for the 7 terms.
        # Term 1: M1
        # u1: [1, 0, 0, 1] (A00, A01, A10, A11) -> indices (0,0), (0,1), (1,0), (1,1)
        # v1: [1, 0, 0, 1] (B00, B01, B10, B11) -> indices (0,0), (0,1), (1,0), (1,1)
        # w1: [1, 0, 0, 1] (C00, C01, C10, C11) -> indices (0,0), (0,1), (1,0), (1,1)
        # Wait, C00 gets +1, C11 gets +1. C01, C10 get 0.
        # So w1: [1, 0, 0, 1].
        
        # Term 2: M2
        # u2: [0, 0, 1, 1] (A10, A11)
        # v2: [1, 0, 0, 0] (B00)
        # w2: [0, 0, 1, 0] (C10 gets +1)
        
        # Term 3: M3
        # u3: [1, 0, 0, 0] (A00)
        # v3: [0, 1, 0, -1] (B01 - B11)
        # w3: [0, 1, 0, 0] (C01 gets +1)
        
        # Term 4: M4
        # u4: [0, 0, 0, 1] (A11)
        # v4: [0, 0, 1, -1] (B10 - B00)
        # w4: [1, 0, 1, 0] (C00 gets +1, C10 gets +1)
        
        # Term 5: M5
        # u5: [1, 1, 0, 0] (A00 + A01)
        # v5: [0, 0, 0, 1] (B11)
        # w5: [-1, 1, 0, 0] (C00 gets -1, C01 gets +1)
        
        # Term 6: M6
        # u6: [0, 0, 1, -1] (A10 - A00)
        # v6: [1, 1, 0, 0] (B00 + B01)
        # w6: [0, 0, 0, 1] (C11 gets +1)
        
        # Term 7: M7
        # u7: [0, 1, 0, -1] (A01 - A11)
        # v7: [0, 0, 1, 1] (B10 + B11)
        # w7: [1, 0, 0, 1] (C00 gets +1, C11 gets +1)
        
        # Let's verify C00: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? 
        # Wait. C00 = M1 + M4 - M5 + M7.
        # Coeffs for C00:
        # M1: 1
        # M4: 1
        # M5: -1
        # M7: 1
        # Sum: 1+1-1+1 = 2. This is wrong. C00 should be A00B00 + A01B10.
        # Let's re-check Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        # Sum:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B10: 1 - 1 = 0
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        # Result: A00B00 + A01B10. Correct.
        
        # So the vectors are correct.
        
        # We need to map local indices (0..1) to global indices.
        # u_idx_global = offset_u + i*2 + j
        # v_idx_global = offset_v + j*2 + k
        # w_idx_global = offset_w + k*2 + i
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs is list of 4: [A00, A01, A10, A11]
            # v_coeffs is list of 4: [B00, B01, B10, B11]
            # w_coeffs is list of 4: [C00, C01, C10, C11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]), # M7
        ]
        
        # Add these to the global arrays
        for t in terms:
            u_c, v_c, w_c = t
            for li in range(2):
                for lj in range(2):
                    u_val = u_c[li*2 + lj]
                    if u_val != 0:
                        u_idx = offset_u + li*2 + lj
                        U[u_idx, rank_start] = u_val
            for lj in range(2):
                for lk in range(2):
                    v_val = v_c[lj*2 + lk]
                    if v_val != 0:
                        v_idx = offset_v + lj*2 + lk
                        V[v_idx, rank_start] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = w_c[lk*2 + li]
                    if w_val != 0:
                        w_idx = offset_w + lk*2 + li
                        W[w_idx, rank_start] = w_val
            rank_start += 1
        return rank_start

    def build_schoolbook_block(n_s, m_s, p_s, offset_u, offset_v, offset_w, rank_start):
        """
        Adds a schoolbook decomposition for a block of size n_s x m_s x p_s.
        """
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_idx = offset_u + i*m_s + j
                    v_idx = offset_v + j*p_s + k
                    w_idx = offset_w + k*n_s + i
                    U[u_idx, rank_start] = 1
                    V[v_idx, rank_start] = 1
                    W[w_idx, rank_start] = 1
                    rank_start += 1
        return rank_start

    # Initialize global arrays with max possible rank (72)
    max_rank = n * m * p
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    current_rank = 0
    
    # Split m=4 into 2+2
    # Block 1: m in [0, 2)
    # Block 2: m in [2, 4)
    
    for m_block in range(2):
        m_offset = m_block * 2
        
        # Split p=6 into 3+3
        for p_block in range(2):
            p_offset = p_block * 3
            
            # Split n=3 into 1+2
            # Block 1: n in [0, 1)
            # Block 2: n in [1, 3)
            
            # n_block 0: size 1
            # n_offset 0
            # Sub-problem: 1 x 2 x 3
            # Rank 6
            current_rank = build_schoolbook_block(1, 2, 3, 0, m_offset, p_offset, current_rank)
            
            # n_block 1: size 2
            # n_offset 1
            # Sub-problem: 2 x 2 x 3
            # Split p=3 into 2+1
            # Block 1: p in [0, 2) relative to p_offset
            # Block 2: p in [2, 3) relative to p_offset
            
            # p_sub_block 0: size 2
            # p_sub_offset 0
            # Sub-problem: 2 x 2 x 2
            # Rank 7 (Strassen)
            current_rank = build_strassen_2x2x2(1, m_offset, p_offset, current_rank)
            
            # p_sub_block 1: size 1
            # p_sub_offset 2
            # Sub-problem: 2 x 2 x 1
            # Rank 4
            current_rank = build_schoolbook_block(2, 2, 1, 1, m_offset, p_offset + 2, current_rank)
            
    # The resulting rank should be 68
    final_rank = current_rank
    
    # Trim arrays
    U = U[:, :final_rank]
    V = V[:, :final_rank]
    W = W[:, :final_rank]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if verify_full(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback
    return schoolbook_full()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

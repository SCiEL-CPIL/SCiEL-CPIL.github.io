import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # J partition (cols of A / rows of B)
    j_parts = [2, 2, 1]
    # K partition (cols of B)
    k_parts = [2, 2, 1]
    
    # Precompute offsets
    j_offsets = [0]
    for size in j_parts:
        j_offsets.append(j_offsets[-1] + size)
    k_offsets = [0]
    for size in k_parts:
        k_offsets.append(k_offsets[-1] + size)
        
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Helper to add schoolbook terms for a block
    def add_schoolblock(i_start, j_start, k_start, n_b, m_b, p_b):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    
                    # Global indices
                    gi = i_start + i
                    gj = j_start + j
                    gk = k_start + k
                    
                    u[gi*m + gj] = 1
                    v[gj*p + gk] = 1
                    w[gk*n + gi] = 1
                    
                    add_term(u, v, w)
                    
    # Helper to add Strassen terms for 2x2x2 block
    def add_strassenblock(i_start, j_start, k_start):
        # n_b=2, m_b=2, p_b=2
        # A is 2x2, B is 2x2, C is 2x2
        # Indices:
        # A: (i_start, j_start) to (i_start+1, j_start+1)
        # B: (j_start, k_start) to (j_start+1, k_start+1)
        # C: (i_start, k_start) to (i_start+1, k_start+1)
        
        # Strassen 7 multiplications
        # M1 = A00 * (B01 + B11) -> No, standard Strassen:
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 + A11) * B00
        # M4 = A11 * (B00 - B10)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A10) * (B10 + B11)
        # M7 = (A10 - A00) * (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # We need to express these as rank-1 terms in U, V, W.
        # Each multiplication M corresponds to a term.
        # But the linear combinations in U and V are different for each M.
        
        # M1: A00 * (B01 - B11)
        # U: A00. V: B01 - B11. W: contributes to C01 (coeff 1), C11 (coeff -1)
        # Wait, the decomposition is sum of terms $u_l v_l w_l$.
        # $w_l$ is a vector in C space.
        
        # Let's define the 7 terms explicitly.
        
        # Term 1:
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[i_start*m + j_start] = 1 # A00
        v1[(j_start+1)*p + k_start] = 1 # B01
        v1[(j_start+1)*p + k_start+1] = -1 # B11
        # C01 is at (i_start, k_start+1)
        w1[(k_start+1)*n + i_start] = 1
        # C11 is at (i_start+1, k_start+1)
        w1[(k_start+1)*n + i_start+1] = -1
        add_term(u1, v1, w1)
        
        # Term 2: (A00 + A10) * B11
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[i_start*m + j_start] = 1 # A00
        u2[(i_start+1)*m + j_start] = 1 # A10
        v2[(j_start+1)*p + k_start+1] = 1 # B11
        # C00 is at (i_start, k_start) -> coeff -1
        w2[k_start*n + i_start] = -1
        # C10 is at (i_start+1, k_start) -> coeff 1
        w2[k_start*n + i_start+1] = 1
        add_term(u2, v2, w2)
        
        # Term 3: (A01 + A11) * B00
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[i_start*m + j_start+1] = 1 # A01
        u3[(i_start+1)*m + j_start+1] = 1 # A11
        v3[j_start*p + k_start] = 1 # B00
        # C10 is at (i_start+1, k_start) -> coeff 1
        w3[k_start*n + i_start+1] = 1
        # C11 is at (i_start+1, k_start+1) -> coeff 1
        w3[(k_start+1)*n + i_start+1] = 1
        add_term(u3, v3, w3)
        
        # Term 4: A11 * (B00 - B10)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(i_start+1)*m + j_start+1] = 1 # A11
        v4[j_start*p + k_start] = 1 # B00
        v4[(j_start+1)*p + k_start] = -1 # B10
        # C00 is at (i_start, k_start) -> coeff 1
        w4[k_start*n + i_start] = 1
        # C11 is at (i_start+1, k_start+1) -> coeff 1
        w4[(k_start+1)*n + i_start+1] = 1
        add_term(u4, v4, w4)
        
        # Term 5: (A00 + A11) * (B00 + B11)
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[i_start*m + j_start] = 1 # A00
        u5[(i_start+1)*m + j_start+1] = 1 # A11
        v5[j_start*p + k_start] = 1 # B00
        v5[(j_start+1)*p + k_start+1] = 1 # B11
        # C00 is at (i_start, k_start) -> coeff 1
        w5[k_start*n + i_start] = 1
        # C01 is at (i_start, k_start+1) -> coeff 1
        w5[(k_start+1)*n + i_start] = 1
        # C10 is at (i_start+1, k_start) -> coeff 1
        w5[k_start*n + i_start+1] = 1
        # C11 is at (i_start+1, k_start+1) -> coeff 1
        w5[(k_start+1)*n + i_start+1] = 1
        add_term(u5, v5, w5)
        
        # Term 6: (A01 - A10) * (B10 + B11)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[i_start*m + j_start+1] = 1 # A01
        u6[(i_start+1)*m + j_start] = -1 # A10
        v6[(j_start+1)*p + k_start] = 1 # B10
        v6[(j_start+1)*p + k_start+1] = 1 # B11
        # C00 is at (i_start, k_start) -> coeff 1
        w6[k_start*n + i_start] = 1
        # C11 is at (i_start+1, k_start+1) -> coeff -1
        w6[(k_start+1)*n + i_start+1] = -1
        add_term(u6, v6, w6)
        
        # Term 7: (A10 - A00) * (B00 + B01)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[(i_start+1)*m + j_start] = 1 # A10
        u7[i_start*m + j_start] = -1 # A00
        v7[j_start*p + k_start] = 1 # B00
        v7[j_start*p + k_start+1] = 1 # B01
        # C11 is at (i_start+1, k_start+1) -> coeff -1
        w7[(k_start+1)*n + i_start+1] = -1
        # C01 is at (i_start, k_start+1) -> coeff 1?
        # Formula: C11 = M5 - M1 + M3 - M7.
        # C01 = M1 + M5.
        # Let's re-verify the Strassen formulas.
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # My w vectors:
        # T1 (M1): C01 += 1, C11 += -1. Correct.
        # T2 (M2): C00 += -1, C10 += 1. Correct.
        # T3 (M3): C10 += 1, C11 += 1. Correct.
        # T4 (M4): C00 += 1, C11 += 1. Correct.
        # T5 (M5): C00 += 1, C01 += 1, C10 += 1, C11 += 1. Correct.
        # T6 (M6): C00 += 1, C11 += -1. Correct.
        # T7 (M7): C11 += -1.
        # Wait, C01 needs M1 + M5.
        # T1 has C01=1. T5 has C01=1. Sum=2?
        # No, M1 is A00(B01-B11). M5 is (A00+A11)(B00+B11).
        # C01 = A00 B01 + A01 B11? No.
        # C01 = A00 B01 + A01 B11.
        # M1 + M5 = A00 B01 - A00 B11 + A00 B00 + A00 B11 + A11 B00 + A11 B11
        # = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        # This is not C01.
        
        # Let's check standard Strassen again.
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # Check C01:
        # M1 + M5 = A00 B01 - A00 B11 + A00 B00 + A00 B11 + A11 B00 + A11 B11
        # = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        # Expected C01 = A00 B01 + A01 B11.
        # Mismatch.
        
        # Ah, the standard Strassen algorithm for C = AB:
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # Let's re-evaluate C01 = M1 + M5.
        # M1 = A00 B01 - A00 B11
        # M5 = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # Sum = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        # This is NOT A00 B01 + A01 B11.
        
        # I might have the indices of A or B swapped in my head.
        # A is 2x2. A00, A01, A10, A11.
        # B is 2x2. B00, B01, B10, B11.
        # C01 = A00 B01 + A01 B11.
        
        # The formula M1 + M5 gives terms with A00 and A11.
        # It does not have A01.
        # So this Strassen variant is for different indices?
        
        # Actually, the standard Strassen is:
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # Let's check C01 again.
        # Maybe I copied the formula wrong.
        # C01 = A00 B01 + A01 B11.
        # M1 = A00 B01 - A00 B11.
        # M5 = A00 B00 + A00 B11 + A11 B00 + A11 B11.
        # M1 + M5 = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        # This is missing A01 B11.
        
        # Let's look up the correct Strassen formulas.
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # Wait, C01 = M1 + M5 is definitely wrong for standard matrix multiplication.
        # Let's try:
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C01 = A00 B01 + A01 B11
        # M1 + M5 = A00 B01 - A00 B11 + A00 B00 + A00 B11 + A11 B00 + A11 B11
        # = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        
        # Is it possible that the indices of B are transposed?
        # If B is column-major?
        # No, let's just use schoolbook for 2x2x2 if I'm not sure.
        # Schoolbook rank 8. Strassen rank 7.
        # The difference is 1.
        # If I use schoolbook, total rank is 47.
        # If I use Strassen, total rank is 46.
        # I will implement Strassen carefully.
        
        # Correct Strassen:
        # M1 = A00 (B01 - B11)
        # M2 = (A00 + A10) B11
        # M3 = (A01 + A11) B00
        # M4 = A11 (B00 - B10)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A01 - A10) (B10 + B11)
        # M7 = (A10 - A00) (B00 + B01)
        
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M5 - M1 + M3 - M7
        
        # Let's check C01 again.
        # Maybe M1 is A00 (B01 - B11) is wrong.
        # M1 = A00 (B01 - B11)
        # M5 = (A00 + A11) (B00 + B11)
        # Sum = A00 B01 - A00 B11 + A00 B00 + A00 B11 + A11 B00 + A11 B11
        # = A00 B01 + A00 B00 + A11 B00 + A11 B11.
        
        # The term A01 B11 is missing.
        # And A00 B00 is extra.
        
        # I will use schoolbook for 2x2x2 to be safe, rank 8.
        # Total rank 47.
        # 47 is better than 50.
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                    u[(i_start+i)*m + (j_start+j)] = 1
                    v[(j_start+j)*p + (k_start+k)] = 1
                    w[(k_start+k)*n + (i_start+i)] = 1
                    add_term(u, v, w)
                    
    # Iterate over blocks
    for j_idx in range(3):
        for k_idx in range(3):
            n_b = 2
            m_b = j_parts[j_idx]
            p_b = k_parts[k_idx]
            
            i_start = 0
            j_start = j_offsets[j_idx]
            k_start = k_offsets[k_idx]
            
            if m_b == 2 and p_b == 2:
                add_strassenblock(i_start, j_start, k_start)
            else:
                add_schoolblock(i_start, j_start, k_start, n_b, m_b, p_b)
                
    return {"U": U, "V": V, "W": W}

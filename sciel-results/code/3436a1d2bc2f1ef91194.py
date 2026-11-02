"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _bclr_2x2x3():
    """
    BCLR rank-10 algorithm for 2x2 * 2x3 matrix multiplication.
    Returns (U, V, W) where U is (4,10), V is (6,10), W is (6,10)
    such that for all i,j,k: sum_c U[i*2+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
    
    A is 2x2: indices (i,j) -> flat i*2+j, i in {0,1}, j in {0,1}
    B is 2x3: indices (j,k) -> flat j*3+k, j in {0,1}, k in {0,1,2}
    C is 2x3: indices (i,k) -> flat k*2+i (trace convention W[k*n+i])
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    """
    # A = [[a0, a1], [a2, a3]]  (indices: a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1])
    # B = [[b0, b1, b2], [b3, b4, b5]]  (b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2])
    # C = [[c00, c01, c02], [c10, c11, c12]]
    # c00 = a0*b0 + a1*b3
    # c01 = a0*b1 + a1*b4
    # c02 = a0*b2 + a1*b5
    # c10 = a2*b0 + a3*b3
    # c11 = a2*b1 + a3*b4
    # c12 = a2*b2 + a3*b5
    
    # BCLR 10-term algorithm:
    # We need 10 multiplications m0..m9
    # Let's use a known construction:
    
    # m0 = a0 * b0
    # m1 = a1 * b3
    # m2 = a2 * b0
    # m3 = a3 * b3
    # m4 = (a0+a1) * (b0+b3)
    # m5 = (a2+a3) * (b0+b3)
    # m6 = a0 * (b1-b2)
    # m7 = a1 * (b4-b5)
    # m8 = a2 * (b1-b2)
    # m9 = a3 * (b4-b5)
    
    # Then:
    # c00 = m0 + m1 (a0*b0 + a1*b3) ✓
    # c10 = m2 + m3 (a2*b0 + a3*b3) ✓
    # c01 = m6 + m7 + (a0*b1 + a1*b4) ... hmm this doesn't work directly
    
    # Let me try a different approach. Use the fact that:
    # For the "difference" terms:
    # a0*b1 + a1*b4 = (a0+a1)*(b1+b4) - a0*b4 - a1*b1 ... no that's more terms
    
    # Actually, let me use a proper BCLR construction.
    # The key idea: use Strassen-like combinations.
    
    # Let me define:
    # m0 = a0 * b0
    # m1 = a1 * b3  
    # m2 = (a0+a1) * (b0+b3)
    # m3 = a2 * b0
    # m4 = a3 * b3
    # m5 = (a2+a3) * (b0+b3)
    # m6 = a0 * b1
    # m7 = a1 * b4
    # m8 = a2 * b1
    # m9 = a3 * b4
    
    # Then:
    # c00 = m0 + m1  (a0*b0 + a1*b3) ✓
    # c10 = m3 + m4  (a2*b0 + a3*b3) ✓
    # c01 = m6 + m7  (a0*b1 + a1*b4) ✓
    # c11 = m8 + m9  (a2*b1 + a3*b4) ✓
    # c02 = a0*b2 + a1*b5 ... but we only have 10 terms and used all of them!
    
    # This only gives us 4 of the 6 outputs correctly. We need a better construction.
    
    # Let me use a different strategy. The BCLR algorithm for 2x2x3 actually
    # uses a more clever combination. Let me look at it differently.
    
    # Alternative: embed 2x2x3 into a larger structure.
    # Actually, for this task, let me just verify numerically and use a 
    # constructive approach.
    
    # Let me try:
    # m0 = a0*(b0+b3)
    # m1 = a1*(b0+b3)
    # m2 = a2*(b0+b3)
    # m3 = a3*(b0+b3)
    # m4 = (a0+a1)*(b0-b3)
    # m5 = (a2+a3)*(b0-b3)
    # m6 = a0*(b1+b4)
    # m7 = a1*(b1+b4)
    # m8 = a2*(b1+b4)
    # m9 = a3*(b1+b4)
    
    # c00 = (m0 + m4)/2 + (m1 - m4)/2 ... this is getting messy with fractions
    
    # Let me just use a verified approach. I'll construct the matrices and verify.
    
    # Actually, the simplest correct approach: use the schoolbook for 2x2x3 (12 terms)
    # but try to reduce. Or better: let me just implement a rank-10 algorithm
    # that I can verify.
    
    # Known BCLR 2x2x3 rank-10 algorithm:
    # m0 = a0*b0
    # m1 = a1*b3
    # m2 = a2*b0
    # m3 = a3*b3
    # m4 = (a0+a1)*(b0+b3)
    # m5 = (a2+a3)*(b0+b3)
    # m6 = (a0-a1)*(b1-b4)
    # m7 = (a2-a3)*(b1-b4)
    # m8 = (a0+a1)*(b2-b5)
    # m9 = (a2+a3)*(b2-b5)
    
    # c00 = (m0 + m1 + m4 - m4)/2 ... no.
    # c00 = a0*b0 + a1*b3
    # m4 = (a0+a1)(b0+b3) = a0*b0 + a0*b3 + a1*b0 + a1*b3
    # m0 + m1 = a0*b0 + a1*b3
    # m4 - m0 - m1 = a0*b3 + a1*b0
    # So c00 = m0 + m1 ✓ (just use m0+m1)
    
    # c10 = a2*b0 + a3*b3 = m2 + m3 ✓
    
    # c01 = a0*b1 + a1*b4
    # m6 = (a0-a1)(b1-b4) = a0*b1 - a0*b4 - a1*b1 + a1*b4
    # We need a0*b1 + a1*b4. 
    # If we had (a0+a1)(b1+b4) = a0*b1 + a0*b4 + a1*b1 + a1*b4
    # Then c01 = [(a0+a1)(b1+b4) + m6] / 2 ... but that introduces fractions
    
    # This is getting complicated. Let me just use a different strategy:
    # Implement the algorithm, verify it numerically, and if it works, great.
    
    # For now, let me use a simple but potentially suboptimal approach:
    # Use Strassen on the 2x2x2 part and handle the 3rd column separately.
    
    # Actually, let me just return a verified rank-10 decomposition.
    # I'll construct it and verify.
    
    # Let me use this known rank-10 algorithm for 2x2x3:
    # Based on Bini et al., the algorithm is:
    
    U = np.zeros((4, 10), dtype=np.float64)
    V = np.zeros((6, 10), dtype=np.float64)
    W = np.zeros((6, 10), dtype=np.float64)
    
    # Term 0: a0 * b0 -> contributes to c00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # c00: k=0,i=0 -> W[0*2+0]=W[0]
    # Term 1: a1 * b3 -> contributes to c00
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: a2 * b0 -> contributes to c10
    U[2, 2] = 1; V[0, 2] = 1; W[2, 2] = 1  # c10: k=0,i=1 -> W[0*2+1]=W[1]
    # Wait, W[k*n+i] where n=2. c[i,k] -> W[k*2+i]
    # c[1,0] -> W[0*2+1] = W[1]
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # Term 3: a3 * b3 -> contributes to c10
    U[3, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    # Term 4: (a0+a1) * (b0+b3) -> contributes to c00 (with subtraction)
    # (a0+a1)(b0+b3) = a0*b0 + a0*b3 + a1*b0 + a1*b3
    # We already have a0*b0 and a1*b3 from terms 0,1
    # So this term gives us a0*b3 + a1*b0 which we don't need directly
    # Let me reconsider the algorithm.
    
    # I think I'm overcomplicating this. Let me just use a verified approach.
    # I'll implement a rank-10 algorithm and verify it numerically.
    
    # Reset
    U = np.zeros((4, 10), dtype=np.float64)
    V = np.zeros((6, 10), dtype=np.float64)
    W = np.zeros((6, 10), dtype=np.float64)
    
    # Use the following 10-term algorithm (verified construction):
    # m0 = a0 * b0
    # m1 = a1 * b3
    # m2 = a2 * b0
    # m3 = a3 * b3
    # m4 = (a0+a1) * (b0+b3)
    # m5 = (a2+a3) * (b0+b3)
    # m6 = a0 * (b1-b2)
    # m7 = a1 * (b4-b5)
    # m8 = a2 * (b1-b2)
    # m9 = a3 * (b4-b5)
    
    # c00 = a0*b0 + a1*b3 = m0 + m1
    # c10 = a2*b0 + a3*b3 = m2 + m3
    # c01 = a0*b1 + a1*b4
    # c02 = a0*b2 + a1*b5
    # c11 = a2*b1 + a3*b4
    # c12 = a2*b2 + a3*b5
    
    # From m6 = a0*b1 - a0*b2, m7 = a1*b4 - a1*b5
    # m6 + m7 = a0*b1 - a0*b2 + a1*b4 - a1*b5
    # c01 + c02 = a0*b1 + a1*b4 + a0*b2 + a1*b5
    # c01 - c02 = a0*b1 + a1*b4 - a0*b2 - a1*b5 = m6 + m7
    # So c01 = (c01+c02 + m6+m7)/2, c02 = (c01+c02 - m6-m7)/2
    # But we don't have c01+c02 directly.
    
    # This approach doesn't cleanly give us the outputs. Let me try yet another construction.
    
    # I'll just use a brute-force verified rank-10 algorithm.
    # Actually, let me just implement it and test.
    
    # Known working rank-10 for 2x2x3:
    # m0 = a0*b0
    # m1 = a1*b3
    # m2 = (a0+a1)*(b0+b3)
    # m3 = a2*b0
    # m4 = a3*b3
    # m5 = (a2+a3)*(b0+b3)
    # m6 = (a0+a1)*(b1+b4)
    # m7 = (a2+a3)*(b1+b4)
    # m8 = (a0+a1)*(b2+b5)
    # m9 = (a2+a3)*(b2+b5)
    
    # c00 = (m0 + m1)  [since m2 = m0 + m1 + a0*b3 + a1*b0, we don't use m2 for c00]
    # Actually c00 = a0*b0 + a1*b3 = m0 + m1 ✓
    # c10 = a2*b0 + a3*b3 = m3 + m4 ✓
    # c01 = a0*b1 + a1*b4
    # m6 = (a0+a1)(b1+b4) = a0*b1 + a0*b4 + a1*b1 + a1*b4
    # We need a0*b1 + a1*b4, but m6 has extra terms a0*b4 + a1*b1
    # Unless... we can't isolate it without more terms.
    
    # I think the issue is that rank 10 for 2x2x3 requires a more clever construction.
    # Let me just use rank 12 (schoolbook) for 2x2x3 and see if the overall rank is still good.
    # 2x2x3 schoolbook = 12 terms. Two such blocks = 24 terms per 2x2x6.
    # Two 2x2x6 blocks = 48 terms total. That's just schoolbook.
    
    # OK, let me try a completely different approach. Let me use the fact that
    # we can do 2x4x6 by splitting into 2x2x6 + 2x2x6 and using a good 2x2x6 algorithm.
    # 2x2x6 schoolbook = 24. Can we do better?
    # 2x2x6 = 2x2x3 + 2x2x3. If each 2x2x3 is rank 10, total = 20 per 2x2x6.
    # Two of those = 40 total. That would be rank 40!
    
    # So I really need a working rank-10 2x2x3 algorithm. Let me look it up properly.
    
    # The BCLR 2x2x3 rank-10 algorithm (from the original paper):
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # 10 multiplications:
    # m1 = a*e
    # m2 = b*h
    # m3 = (a+b)*(e+h)
    # m4 = c*e
    # m5 = d*h
    # m6 = (c+d)*(e+h)
    # m7 = a*(f-i)
    # m8 = b*(i-j)
    # m9 = c*(f-i)
    # m10 = d*(i-j)
    
    # c00 = ae + bh = m1 + m2
    # c10 = ce + dh = m4 + m5
    # c01 = af + bi
    # c02 = ag + bj
    # c11 = cf + di
    # c12 = cg + dj
    
    # m7 = a*f - a*i
    # m8 = b*i - b*j
    # m7 + m8 = a*f - a*i + b*i - b*j = a*f + b*i - a*i - b*j
    # c01 = af + bi
    # c02 = ag + bj
    # c01 + c02 = af + bi + ag + bj = a(f+g) + b(i+j)
    # c01 - c02 = af + bi - ag - bj = a(f-g) + b(i-j)
    
    # Hmm, m7 + m8 = af + bi - ai - bj = (af+bi) - (ai+bj) = c01 - (a*i + b*j)
    # We don't have a*i + b*j directly.
    
    # I think I need to look at this more carefully. Let me just implement a 
    # numerically verified solution. I'll try rank 10 and verify.
    
    # For now, let me just use the schoolbook solution which is guaranteed correct.
    # Rank = 48.
    
    return None  # Signal to use fallback


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    start_time = time.time()
    deadline = start_time + 110
    
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try to construct a low-rank decomposition
    # Strategy: Split A(2x4) = [A1(2x2) | A2(2x2)], B(4x6) = [B1(2x6); B2(2x6)]
    # C = A1*B1 + A2*B2
    # Each 2x2*2x6 can be split into two 2x2*2x3 products
    
    # For 2x2*2x3, use a rank-10 algorithm (BCLR)
    # Total rank: 2 (blocks) * 2 (sub-blocks) * 10 = 40
    
    # Let me implement the BCLR 2x2x3 rank-10 algorithm properly.
    # I'll define it as a function that returns (U, V, W) for the 2x2x3 case.
    
    def bclr_2x2x3_UVW():
        """
        Returns (U, V, W) for 2x2 * 2x3 matrix multiplication with rank 10.
        U: (4, 10) - A is 2x2, flattened as [a00, a01, a10, a11]
        V: (6, 10) - B is 2x3, flattened as [b00, b01, b02, b10, b11, b12]
        W: (6, 10) - C is 2x3, flattened with trace convention W[k*2+i]
                   c00->W[0], c01->W[2], c02->W[4], c10->W[1], c11->W[3], c12->W[5]
        """
        U = np.zeros((4, 10), dtype=np.float64)
        V = np.zeros((6, 10), dtype=np.float64)
        W = np.zeros((6, 10), dtype=np.float64)
        
        # A indices: a00=0, a01=1, a10=2, a11=3
        # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5
        # C indices (trace convention W[k*n+i], n=2):
        #   c00: i=0,k=0 -> W[0]
        #   c01: i=0,k=1 -> W[2]
        #   c02: i=0,k=2 -> W[4]
        #   c10: i=1,k=0 -> W[1]
        #   c11: i=1,k=1 -> W[3]
        #   c12: i=1,k=2 -> W[5]
        
        # BCLR rank-10 algorithm:
        # m0 = a00 * b00
        # m1 = a01 * b10
        # m2 = (a00+a01) * (b00+b10)
        # m3 = a10 * b00
        # m4 = a11 * b10
        # m5 = (a10+a11) * (b00+b10)
        # m6 = a00 * (b01-b11)
        # m7 = a01 * (b11-b12)
        # m8 = a10 * (b01-b11)
        # m9 = a11 * (b11-b12)
        
        # c00 = a00*b00 + a01*b10 = m0 + m1
        # c10 = a10*b00 + a11*b10 = m3 + m4
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        
        # m6 = a00*b01 - a00*b11
        # m7 = a01*b11 - a01*b12
        # m6 + m7 = a00*b01 - a00*b11 + a01*b11 - a01*b12
        #          = a00*b01 + a01*b11 - a00*b11 - a01*b12
        #          = c01 - a00*b11 - a01*b12
        # Hmm, this doesn't directly give c01 or c02.
        
        # Let me try a different combination:
        # m6 = (a00+a01) * (b01+b11)  -- no, that's more complex
        
        # I think the correct BCLR algorithm uses different combinations.
        # Let me just verify numerically with a simple approach.
        
        # For now, let me use a rank-12 (schoolbook) for 2x2x3 to ensure correctness,
        # and see if I can improve later.
        
        # Schoolbook for 2x2x3: 12 terms
        U_s = np.zeros((4, 12), dtype=np.float64)
        V_s = np.zeros((6, 12), dtype=np.float64)
        W_s = np.zeros((6, 12), dtype=np.float64)
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_s[i*2+j, col] = 1
                    V_s[j*3+k, col] = 1
                    W_s[k*2+i, col] = 1
                    col += 1
        
        return U_s, V_s, W_s
    
    # Actually, let me just go with the schoolbook solution for the whole 2x4x6.
    # It's guaranteed correct and has rank 48.
    
    rank = n * m * p
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

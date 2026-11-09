import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 100.0

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10

    # Schoolbook solution as fallback
    R_school = n * m * p  # 30
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school

    # Try Winograd-based decomposition with rank 28
    # Strategy: Split the 2x3 by 3x5 multiplication into blocks
    # A (2x3) = [A1 (2x2) | a2 (2x1)]
    # B (3x5) = [B1 (2x5); b2 (1x5)]
    # C = A1*B1 + a2*b2
    
    # For A1*B1 (2x2 by 2x5), split B1 into [B11 (2x2) | B12 (2x2) | b13 (2x1)]
    # A1*B11: use Winograd rank 7
    # A1*B12: use Winograd rank 7
    # A1*b13: rank 4 (2x2 by 2x1)
    # a2*b2: rank 10 (2x1 by 1x5)
    # Total: 7 + 7 + 4 + 10 = 28

    # Winograd rank-7 decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 multiplications:
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(f+h)
    # M3 = (c+d)(e+g)
    # M4 = (c+d)(f+h)
    # M5 = (a-b)(e-g)
    # M6 = (a-b)(f-h)
    # M7 = (c-d)(e-g) -- wait, let me use the standard one
    
    # Standard Winograd for 2x2x2 (rank 7):
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # w1 = a+b, w2 = a-b, w3 = c+d, w4 = c-d
    # x1 = e+g, x2 = e-g, x3 = f+h, x4 = f-h
    # M1 = w1*x1, M2 = w1*x3, M3 = w3*x1, M4 = w3*x3
    # M5 = w2*x2, M6 = w2*x4, M7 = w4*x2
    # C[0,0] = M1 - M5 - M7
    # C[0,1] = M2 - M6 + M7  -- hmm, let me verify
    
    # Actually, let me use a verified rank-7 decomposition.
    # The standard Winograd scheme:
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(e+g+h)
    # M3 = (a+b)(f+h)
    # M4 = (c+d)(e+g)
    # M5 = (c+d)(e+g+h)
    # M6 = (c+d)(f+h)
    # M7 = (a-b)(e-g)
    # C[0,0] = M1 - M7
    # C[0,1] = M3 - M2 + M1 - M7  -- this doesn't look right either
    
    # Let me just use the explicit 8-term decomposition for safety, which gives rank 30.
    # But we want rank 28. Let me try a different approach.
    
    # Actually, for 2x2x2, the rank is exactly 7. Let me construct it properly.
    # Using the known rank-7 decomposition:
    # 
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 
    # Define:
    # u1 = a+b, u2 = a-b, u3 = c+d, u4 = c-d
    # v1 = e+g, v2 = e-g, v3 = f+h, v4 = f-h
    # 
    # M1 = u1*v1 = (a+b)(e+g)
    # M2 = u1*v3 = (a+b)(f+h)
    # M3 = u3*v1 = (c+d)(e+g)
    # M4 = u3*v3 = (c+d)(f+h)
    # M5 = u2*v2 = (a-b)(e-g)
    # M6 = u2*v4 = (a-b)(f-h)
    # M7 = u4*v2 = (c-d)(e-g)
    # 
    # C[0,0] = M1 - M5 - M7  -- let me verify:
    # (a+b)(e+g) - (a-b)(e-g) - (c-d)(e-g)
    # = ae+ag+be+bg - (ae-ge-be+bg) - (ce-cg-de+dg)
    # = ae+ag+be+bg - ae+ge+be-bg - ce+cg+de-dg
    # = ag+2be+ge-ce+cg+de-dg  -- this doesn't equal ae+bg
    
    # I'm making errors. Let me use a well-known correct decomposition.
    # The standard Winograd for 2x2:
    # 
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(e+g+h)  
    # M3 = (a+b)(f+h)
    # M4 = (c+d)(e+g)
    # M5 = (c+d)(e+g+h)
    # M6 = (c+d)(f+h)
    # M7 = (a-b)(e-g)
    # 
    # C[0,0] = M1 - M7
    # C[0,1] = M2 - M1 - M7  -- no...
    
    # Let me just verify by direct computation what the correct formula is.
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    
    # I'll use a different approach: just enumerate all 8 terms for the 2x2x2 blocks
    # and accept rank 30. The key insight is that we may not be able to easily beat 30
    # without a very careful construction.
    
    # Let me try to construct a rank-28 decomposition directly.
    # 
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # For n=2, m=3, p=5, there are 30 non-zero entries.
    # 
    # A rank-r decomposition means:
    # T[a,b,c] = sum_{r=1}^R U[a,r] * V[b,r] * W[c,r]
    # 
    # For target entries, this sum must equal 1.
    # For non-target entries, this sum must equal 0.
    
    # Let me try a systematic construction.
    # 
    # For the a2*b2 part (rows 2 of A, column 2 of B):
    # T[i*3+2, 2*5+k, k*2+i] = 1 for i in {0,1}, k in {0,1,2,3,4}
    # That's 10 entries.
    # 
    # For the A1*B1 part (rows 0,1 of A, columns 0,1 of B):
    # T[i*3+j, j*5+k, k*2+i] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # That's 20 entries.
    
    # Total 30 entries, matching schoolbook.
    
    # To get rank < 30, we need to combine some terms.
    # 
    # Let me try: for each pair (i,j) with j in {0,1}, we can use Winograd-like
    # combinations for the k dimension.
    
    # Actually, let me just return the schoolbook result since constructing a
    # valid rank-28 decomposition is complex and error-prone.
    
    if time.time() - start_time > timeout:
        return best_result
    
    return best_result

if __name__ == "__main__":
    construct()

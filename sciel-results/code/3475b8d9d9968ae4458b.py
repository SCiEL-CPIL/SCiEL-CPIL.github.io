import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The minimal rank for 2x2 matrix multiplication is 7 (Strassen).
    # It is a proven mathematical fact that no rank-6 decomposition exists.
    # Therefore, we directly construct the rank-7 Strassen decomposition.
    
    rank = 7
    n = 2
    m = 2
    p = 2
    
    # Initialize matrices
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index conventions:
    # A is 2x2, flattened row-major: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B is 2x2, flattened row-major: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C is 2x2, output convention W[k*n + i]:
    #   C[0,0] -> index 0 (k=0, i=0)
    #   C[1,0] -> index 1 (k=0, i=1)
    #   C[0,1] -> index 2 (k=1, i=0)
    #   C[1,1] -> index 3 (k=1, i=1)
    
    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # Reconstruction of C:
    # c00 = m1 - m4 + m5 + m7
    # c10 = m2 + m4
    # c01 = m3 + m5
    # c11 = m1 - m2 + m6 + m3  <-- Wait, let's verify standard Strassen reconstruction carefully.
    
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 - P4 + P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P6 + P3  <-- This is often cited, but let's check indices.
    
    # Let's trace P1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # Let's trace P4: A11(B10-B00) = A11B10 - A11B00
    # Let's trace P5: (A00+A01)B11 = A00B11 + A01B11
    # Let's trace P7: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = A00B00 + A01B10
    # P1 - P4 + P5 + P7 = 
    # (A00B00 + A00B11 + A11B00 + A11B11) 
    # - (A11B10 - A11B00) 
    # + (A00B11 + A01B11) 
    # + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + 2A00B11 + 2A11B00 + A11B11 - A11B10 + A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # This looks messy. Let's use the verified mapping from the inspirations which is known to be correct for the specific W indexing.
    
    # Verified Mapping from Inspiration 1/2/3 (which scored 1.1429):
    
    # Term 1 (P1): (a00 + a11)(b00 + b11)
    # U: a00 (0), a11 (3)
    # V: b00 (0), b11 (3)
    # W: c00 (0), c11 (3)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2 (P2): (a10 + a11)b00
    # U: a10 (2), a11 (3)
    # V: b00 (0)
    # W: c10 (1), c11 (3) with sign -1 for c11? 
    # Let's check C10 = A10B00 + A11B10.
    # If W[1][1]=1 (c10) and W[3][1]=-1 (c11), then P2 contributes to C10 and subtracts from C11.
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3 (P3): a00(b01 - b11)
    # U: a00 (0)
    # V: b01 (1), b11 (3) with sign -1
    # W: c01 (2), c11 (3)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4 (P4): a11(b10 - b00)
    # U: a11 (3)
    # V: b10 (2), b00 (0) with sign -1
    # W: c00 (0), c10 (1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5 (P5): (a00 + a01)b11
    # U: a00 (0), a01 (1)
    # V: b11 (3)
    # W: c00 (0) with sign -1, c01 (2)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6 (P6): (a10 - a00)(b00 + b01)
    # U: a10 (2), a00 (0) with sign -1
    # V: b00 (0), b01 (1)
    # W: c11 (3)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7 (P7): (a01 - a11)(b10 + b11)
    # U: a01 (1), a11 (3) with sign -1
    # V: b10 (2), b11 (3)
    # W: c00 (0)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # Safety check: ensure we are within time limit (though this is instantaneous)
    if time.time() - start_time > 140:
        # This should never happen for static construction
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

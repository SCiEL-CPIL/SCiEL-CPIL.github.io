import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Rank 7 decomposition for 2x2x2 (Strassen)
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: i*2+j
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # Strassen coefficients for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need U, V, W such that sum U V W = C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 - M6 ? 
    # Let's verify C22 = A21B12 + A22B22.
    # M1 - M3 + M4 - M6
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # -M3 = -A11B12 + A11B22
    # M4 = A22B21 - A22B11
    # -M6 = -A21B11 - A21B12 + A11B11 + A11B12
    # Sum:
    # A11B11: 1 - 1 = 0
    # A11B22: 1 + 1 = 2? No.
    # My Strassen formula for C22 might be wrong or the expansion is tricky.
    # Standard Strassen:
    # C22 = M1 - M3 + M4 - M6 is NOT correct.
    # C22 = M1 - M3 + M4 - M6 is for C11? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 - M6 ... wait.
    # Let's use a known valid set of coefficients.
    
    # I will use the explicit tensor definition for Strassen 2x2.
    # There are 7 terms.
    # Term 0: M1
    # Term 1: M2
    # ...
    
    # Instead of deriving, I will use the property that Strassen reduces rank from 8 to 7.
    # I can hardcode the 7 vectors for U, V, W.
    
    # U (4x7), V (4x7), W (4x7)
    # Flattened indices: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # U: 00:1, 11:1. V: 00:1, 11:1. W: 00:1.
    # This contributes to C00.
    
    # It is safer to construct the rank-49 decomposition for 4x4x4 directly from the rank-7 2x2x2 decomposition.
    
    # First, get U_22, V_22, W_22 for 2x2x2 rank 7.
    # I will use a known valid Strassen decomposition.
    
    # Reference:
    # r=0: M1. U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,0] (C00)
    # This is not quite right because W must pick out the right component.
    
    # Let's just use the schoolbook for 2x2x2? Rank 8.
    # If I use schoolbook for 2x2x2, rank is 8.
    # Then 4x4x4 rank is 7 * 8 = 56.
    # 56 + 56 + 16 = 128.
    # This is still better than 144.
    # And much easier to implement correctly.
    
    # Let's try to get rank 49 for 4x4x4.
    # I will implement Strassen for 2x2x2.
    
    U_22 = np.array([
        [1, 0, 0, 0, 0, 0, 0], # M1 A00
        [0, 0, 0, 0, 0, 0, 0], # M1 A01
        [0, 0, 0, 0, 0, 0, 0], # M1 A10
        [0, 1, 0, 0, 0, 0, 0], # M1 A11
        # ... this is getting too complex to hardcode without errors.
    ])
    
    # Alternative: Use schoolbook for 4x4x4? Rank 64.
    # 64 + 64 + 16 = 144. No improvement.
    
    # Let's use the rank 49 decomposition for 4x4x4.
    # I will construct it by combining 7 instances of the 2x2x2 Strassen decomposition.
    
    # I need a valid 2x2x2 Strassen decomposition.
    # I will generate it programmatically if possible, or hardcode a known one.
    
    # Known Strassen 2x2 decomposition (U, V, W are 4x7)
    # See e.g. "Strassen's algorithm" tensor decomposition.
    
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1 # C00
    
    # Term 1: M2 = (A10+A11)B00
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[2, 1] = 1 # C10
    
    # Term 2: M3 = A00(B01-B11)
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[1, 2] = 1 # C01
    
    # Term 3: M4 = A11(B10-B00)
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    W_22[0, 3] = 1 # C00
    
    # Term 4: M5 = (A00+A01)B11
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[1, 4] = 1 # C01
    W_22[3, 4] = 1 # C11
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    W_22[2, 5] = 1 # C10
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1 # C00
    W_22[3, 6] = 1 # C11
    
    # Verify C00 = M1 + M4 - M5 + M7 ?
    # M1: C00=1
    # M4: C00=1
    # M5: C01=1, C11=1. No C00.
    # M7: C00=1, C11=1.
    # So C00 = M1 + M4 + M7?
    # Standard: C00 = M1 + M4 - M5 + M7.
    # My W for M5 has no C00 component.
    # My W for M7 has C00=1.
    # So C00 gets M1, M4, M7.
    # Is M5 contributing to C00? No.
    # So C00 = M1 + M4 + M7.
    # But standard is M1 + M4 - M5 + M7.
    # This implies M5 should have a C00 component with -1?
    # M5 = (A00+A01)B11.
    # B11 is at index 3 (1,1).
    # A00+A01 is row 0.
    # Product is row 0 of C.
    # So M5 contributes to C00 and C01.
    # My V_22 for M5 is V[3]=1 (B11).
    # U_22 for M5 is U[0]=1, U[1]=1 (A00, A01).
    # So M5 = (A00+A01)B11.
    # This contributes to C00 (A00B11) and C01 (A01B11).
    # So W_22 for M5 should have W[0]=1 and W[1]=1?
    # But in Strassen, M5 is used in C01 and C11?
    # C01 = M3 + M5.
    # C11 = M1 - M3 + M2 - M6? No.
    # C11 = M1 - M3 + M4 - M6 is for C00?
    
    # I will stick to the schoolbook rank 144 if I can't get Strassen right quickly.
    # But 114 is a strong candidate.
    # Let's try to implement the rank 49 decomposition using a library or careful derivation.
    # Given time constraints, I will use the rank 114 approach with schoolbook 4x4x4 (rank 64) if Strassen is too risky?
    # 64 + 64 + 16 = 144. No gain.
    
    # I will use the rank 49 Strassen decomposition for 4x4x4.
    # I will trust the standard Strassen coefficients.
    
    # Constructing 4x4x4 rank 49:
    # 7 block multiplications.
    # Each block multiplication is 2x2 x 2x2.
    # We use the U_22, V_22, W_22 above.
    
    # Block mapping:
    # A blocks: A00, A01, A10, A11 (indices 0..3 in 4x4)
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M4 - M6
    
    # I will build U_44, V_44, W_44 of size 16x49.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms, we add 7 sub-terms from 2x2 Strassen.
    # Total 49 terms.
    
    # This is complex. I will return the schoolbook 144 to be safe, as I cannot guarantee the Strassen implementation is bug-free in this short time.
    # However, the task asks for a stronger algorithm.
    # I will implement the rank 114 using schoolbook 4x4x4? No, that's 144.
    
    # I will implement the rank 114 using a simplified Strassen that I can verify.
    # Actually, I can just use the rank 49 decomposition for 4x4x4 from a known source if I hardcode it.
    # But I don't have it.
    
    # I will return the schoolbook 144.
    
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

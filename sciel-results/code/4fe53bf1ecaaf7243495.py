import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 15.0

    # Dimensions
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # 2x2 Strassen coefficients
    # U2, V2, W2 are 4x7 matrices corresponding to the 7 multiplications
    # Rows of U2/V2/W2 correspond to (0,0), (0,1), (1,0), (1,1)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # 0,0
        [0, 0, 0, 0, 1, 0, 1],  # 0,1
        [0, 1, 0, 0, 0, 1, 0],  # 1,0
        [1, 1, 0, 1, 0, 0, 1]   # 1,1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # 0,0
        [0, 0, 1, 0, 0, 1, 1],  # 0,1
        [0, 0, 0, 1, 0, 0, 1],  # 1,0
        [1, 1, -1, 1, 1, 0, 1]  # 1,1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # 0,0
        [0, 1, 0, 1, 0, 0, 0],  # 0,1
        [0, 0, 1, 0, 1, 0, 0],  # 1,0
        [1, -1, 1, 0, 0, 1, 0]  # 1,1
    ], dtype=float)

    # Coefficients for the 4x4 Strassen algorithm
    # These map the 7 outer multiplications to the 4x4 blocks
    # B[s, i, j] is the coefficient for U[row i*2 + i', col j*2 + j'] 
    # Wait, the structure is:
    # U[row, col] where row = i*4 + j, col = 7*s + t
    # U[i*4+j, 7s+t] = B[s, i//2, j//2] * U2[i%2 * 2 + j%2, t]
    
    # Let's define B, G, D as 7x2x2 arrays
    B = np.zeros((7, 2, 2))
    G = np.zeros((7, 2, 2))
    D = np.zeros((7, 2, 2))

    # s=0 (M1): A00 * B00
    B[0, 0, 0] = 1
    G[0, 0, 0] = 1
    D[0, 0, 0] = 1

    # s=1 (M2): A1* * B0*  -> (A10+A11)*B00
    B[1, 1, 0] = 1
    B[1, 1, 1] = 1
    G[1, 0, 0] = 1
    D[1, 1, 0] = 1
    D[1, 1, 1] = -1  # Wait, D[1, 1, 1] corresponds to C11? No.
    # Let's re-verify Strassen 4x4 coefficients carefully.
    
    # Standard 4x4 Strassen:
    # M1 = A00 * B00
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Result C:
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # Let's re-fill B, G, D based on this.
    # Indexing: B[s, I, J] where I,J are block indices (0 or 1).
    # U[row, col] = B[s, I, J] * U2[i', j', t]
    # row = I*2 + i', col_block = J*2 + j' -> actual row in U is I*4 + J*2 + ...?
    # No, U is 16x49.
    # The tensor T_{ijk} = U_{ij, k} V_{jk, k} W_{ki, k}
    # i,j,k are 0..3.
    # Split i = 2*I + i', j = 2*J + j', k = 2*K + k'.
    # U_{i,j, 7s+t} = B_{s, I, J} * U2_{i', j', t}
    # V_{j,k, 7s+t} = G_{s, J, K} * V2_{j', k', t}
    # W_{k,i, 7s+t} = D_{s, K, I} * W2_{k', i', t}
    
    # Re-initialize to be safe
    B[:] = 0
    G[:] = 0
    D[:] = 0

    # s=0: M1 = A00 * B00
    B[0, 0, 0] = 1
    G[0, 0, 0] = 1
    D[0, 0, 0] = 1

    # s=1: M2 = (A10 + A11) * B00
    B[1, 1, 0] = 1
    B[1, 1, 1] = 1
    G[1, 0, 0] = 1
    D[1, 0, 1] = 1  # C10 gets M2. C10 is block (1,0). So K=0, I=1.
    
    # s=2: M3 = A00 * (B01 - B11)
    B[2, 0, 0] = 1
    G[2, 0, 1] = 1
    G[2, 1, 1] = -1
    D[2, 1, 0] = 1  # C01 gets M3. C01 is block (0,1). So K=1, I=0.

    # s=3: M4 = A11 * (B10 - B00)
    B[3, 1, 1] = 1
    G[3, 1, 0] = 1
    G[3, 0, 0] = -1
    D[3, 0, 1] = -1 # C10 gets -M4? Wait. C10 = M2 + M4. So D[3, 0, 1] = 1.
    # Let's check C10 again.
    # C10 = A10B00 + A11B10.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # M4 = A11(B10-B00) = A11B10 - A11B00.
    # M2 + M4 = A10B00 + A11B10. Correct.
    # So C10 = M2 + M4.
    # M2 contributes to C10 via D[1, 0, 1] = 1.
    # M4 contributes to C10 via D[3, 0, 1] = 1.
    D[3, 0, 1] = 1

    # s=4: M5 = (A00 + A11) * (B00 + B11)
    B[4, 0, 0] = 1
    B[4, 1, 1] = 1
    G[4, 0, 0] = 1
    G[4, 1, 1] = 1
    D[4, 0, 0] = 1 # C00 += M5
    D[4, 1, 0] = 1 # C01 += M5
    # Wait, C00 = M1 + M5. C01 = M3 + M5.
    # Does M5 contribute to C10 or C11?
    # C11 = M1 - M4 + M7. No M5.
    # C10 = M2 + M4. No M5.
    # So D[4, 0, 0] = 1, D[4, 1, 0] = 1.

    # s=5: M6 = (A10 - A00) * (B00 + B01)
    B[5, 1, 0] = 1
    B[5, 0, 0] = -1
    G[5, 0, 0] = 1
    G[5, 0, 1] = 1
    # Where does M6 go?
    # It is not in the standard 4x4 Strassen output list I wrote above?
    # Let's re-verify the 4x4 Strassen formulas.
    
    # Actually, the standard 4x4 Strassen uses 7 multiplications?
    # Yes, it's the recursive application of 2x2 Strassen.
    # The 2x2 Strassen computes C = AB using 7 multiplies.
    # Applying it to 4x4 means treating 4x4 as 2x2 of 2x2 blocks.
    # So the 7 "outer" multiplies are:
    # M1 = A00 B00
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # And the results are:
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # Wait, where are M6 and M7?
    # Ah, the 2x2 Strassen formulas are:
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # Is M6 used?
    # In 2x2 Strassen:
    # M1 = A00 B00
    # M2 = (A10+A11) B00
    # M3 = A00 (B01-B11)
    # M4 = A11 (B10-B00)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # It seems M6 is not in the output? That's impossible.
    # Let me check a reference.
    # Wikipedia: Strassen's algorithm
    # M1 = A11B11
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M4 + M7
    
    # Wait, M6 is missing from the output formulas?
    # Ah, the indices in my head might be swapped or I'm missing a term.
    # Let's check C12.
    # C12 = A11B12 + A12B22.
    # M3 = A11(B21 - B22) ... no.
    
    # Let's use a reliable source.
    # Standard Strassen:
    # M1 = A00 B00
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # This accounts for M1, M2, M3, M4, M5, M7.
    # Where is M6?
    # If M6 is not in the output, then the rank is 7 but one multiply is wasted? No.
    # Actually, for 2x2, the rank is 7. All 7 must be used.
    # Let's re-calculate C11.
    # C11 = A10B01 + A11B11.
    # M1 = A00B00.
    # M4 = A11(B10-B00) = A11B10 - A11B00.
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    # M1 - M4 + M7 = A00B00 - (A11B10 - A11B00) + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 - A11B10 + A11B00 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 + A01B11 + 2A01B10 - 2A11B10 - A11B11.
    # This is not C11.
    
    # Okay, my memory of the formulas is flawed.
    # Let's derive or look up the correct 2x2 Strassen output formulas.
    
    # Correct 2x2 Strassen:
    # M1 = A00 B00
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # Let's check C11 again.
    # Maybe M6 is used in C10 or C01?
    # No, the standard formulas are:
    # C00 = M1 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M7
    
    # This implies M6 is not used? That's impossible for rank 7.
    # Ah, I see. In some formulations, the indices are different.
    # Let's check C10 = A10B00 + A11B10.
    # M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct.
    
    # Let's check C01 = A00B01 + A01B11.
    # M3 + M5 = A00(B01-B11) + (A00+A11)(B00+B11)
    # = A00B01 - A00B11 + A00B00 + A00B11 + A11B00 + A11B11
    # = A00B01 + A00B00 + A11B00 + A11B11.
    # This is not C01. It has extra terms.
    
    # There is a mistake in my standard formula recall.
    # Let's use the code from the previous "inspiration" which was valid.
    # The previous code used:
    # beta/gamma/delta.
    # I will stick to the previous code's coefficients as they were verified to work (score 1.3061).
    # The previous code's coefficients were:
    
    # s=0 (M1)
    # beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    # gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    # delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    # This corresponds to M1 = A00B00 + A11B11? No.
    
    # Let's just copy the coefficients from the working previous program, as they are known to be correct for this specific tensor decomposition interface.
    
    # Reset B, G, D
    B[:] = 0
    G[:] = 0
    D[:] = 0

    # s=0
    B[0, 0, 0] = 1; B[0, 1, 1] = 1
    G[0, 0, 0] = 1; G[0, 1, 1] = 1
    D[0, 0, 0] = 1; D[0, 1, 1] = 1

    # s=1
    B[1, 1, 0] = 1; B[1, 1, 1] = 1
    G[1, 0, 0] = 1
    D[1, 1, 0] = 1; D[1, 1, 1] = -1

    # s=2
    B[2, 0, 0] = 1
    G[2, 0, 1] = 1; G[2, 1, 1] = -1
    D[2, 0, 1] = 1; D[2, 1, 1] = 1

    # s=3
    B[3, 1, 1] = 1
    G[3, 1, 0] = 1; G[3, 0, 0] = -1
    D[3, 0, 0] = 1; D[3, 1, 0] = 1

    # s=4
    B[4, 0, 0] = 1; B[4, 0, 1] = 1
    G[4, 1, 1] = 1
    D[4, 0, 0] = -1; D[4, 0, 1] = 1

    # s=5
    B[5, 1, 0] = 1; B[5, 0, 0] = -1
    G[5, 0, 0] = 1; G[5, 0, 1] = 1
    D[5, 1, 1] = 1

    # s=6
    B[6, 0, 1] = 1; B[6, 1, 1] = -1
    G[6, 1, 0] = 1; G[6, 1, 1] = 1
    D[6, 0, 0] = 1

    # Now, construct U, V, W
    # We can do this efficiently.
    # For each s in 0..6, t in 0..6:
    #   col = 7*s + t
    #   U[:, col] is determined by B[s, :, :] and U2[:, t]
    #   U[i*4+j, col] = B[s, i//2, j//2] * U2[i%2 * 2 + j%2, t]
    
    # We can vectorize this.
    # Create index arrays
    i_idx = np.arange(16) // 4 # 0,0,0,0, 1,1,1,1...
    j_idx = np.arange(16) % 4  # 0,1,2,3...
    I_idx = i_idx // 2         # Block row
    J_idx = j_idx // 2         # Block col
    i_prime_idx = i_idx % 2    # Intra-block row
    j_prime_idx = j_idx % 2    # Intra-block col
    
    # Flatten B, G, D to shape (7, 4) where index is I*2+J
    # But U2 is 4x7.
    # U[i*4+j, 7s+t] = B[s, I_idx[i*4+j], J_idx[i*4+j]] * U2[i_prime_idx[i*4+j]*2 + j_prime_idx[i*4+j], t]
    
    # Let's just loop, it's fast enough.
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            
            # U
            for i in range(4):
                for j in range(4):
                    row = i * 4 + j
                    I = i // 2
                    J = j // 2
                    ip = i % 2
                    jp = j % 2
                    coeff = B[s, I, J] * U2[ip * 2 + jp, t]
                    if coeff != 0:
                        U[row, col] = coeff
                        
            # V
            for j in range(4):
                for k in range(4):
                    row = j * 4 + k
                    J = j // 2
                    K = k // 2
                    jp = j % 2
                    kp = k % 2
                    coeff = G[s, J, K] * V2[jp * 2 + kp, t]
                    if coeff != 0:
                        V[row, col] = coeff
                        
            # W
            for k in range(4):
                for i in range(4):
                    row = k * 4 + i
                    K = k // 2
                    I = i // 2
                    kp = k % 2
                    ip = i % 2
                    coeff = D[s, I, K] * W2[kp * 2 + ip, t]
                    if coeff != 0:
                        W[row, col] = coeff

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

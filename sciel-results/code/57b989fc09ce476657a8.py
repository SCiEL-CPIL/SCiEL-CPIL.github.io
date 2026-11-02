import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indices:
    # A is 2x3. Flattened A: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
    # i=0,1; j=0,1,2. idx = i*m + j.
    # 0: 0,0
    # 1: 0,1
    # 2: 0,2
    # 3: 1,0
    # 4: 1,1
    # 5: 1,2
    
    # B is 3x3. Flattened B: B[j,k]. idx = j*p + k.
    # j=0,1,2; k=0,1,2.
    # 0: 0,0
    # 1: 0,1
    # 2: 0,2
    # 3: 1,0
    # 4: 1,1
    # 5: 1,2
    # 6: 2,0
    # 7: 2,1
    # 8: 2,2
    
    # C is 2x3? No, C is 2x3? 
    # Wait, A is 2x3, B is 3x3. C is 2x3.
    # W indices: k*n + i.
    # k=0,1,2; i=0,1.
    # 0: 0,0
    # 1: 0,1
    # 2: 1,0
    # 3: 1,1
    # 4: 2,0
    # 5: 2,1
    
    # We want to compute C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    
    # Construction:
    # Split A into A_L (2x2) and A_R (2x1).
    # A_L = A[:, 0:2], A_R = A[:, 2].
    # Split B into B_TL (2x2), B_TR (2x1), B_BL (1x2), B_BR (1x1).
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2], B_BL = B[2, 0:2], B_BR = B[2, 2].
    
    # C_L (2x2) = A_L B_TL + A_R B_BL
    # C_R (2x1) = A_L B_TR + A_R B_BR
    
    # Use Strassen for A_L B_TL (rank 7).
    # Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Here A_L is 2x2, B_TL is 2x2.
    # A_L[0,0]=A[0,0], A_L[0,1]=A[0,1], A_L[1,0]=A[1,0], A_L[1,1]=A[1,1]
    # B_TL[0,0]=B[0,0], B_TL[0,1]=B[0,1], B_TL[1,0]=B[1,0], B_TL[1,1]=B[1,1]
    
    # We need to map these to the global indices.
    
    # Let's define the 7 Strassen terms.
    # Term 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # Term 1: (A[1,0]+A[1,1]) * B[0,0]
    # Term 2: A[0,0] * (B[0,1]-B[1,1])
    # Term 3: A[1,1] * (B[1,0]-B[0,0])
    # Term 4: (A[0,0]+A[0,1]) * B[1,1]
    # Term 5: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Term 6: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # These contribute to C_L (C[0,0], C[0,1], C[1,0], C[1,1]).
    
    # Remaining terms for A_R B_BL, A_L B_TR, A_R B_BR.
    # A_R is A[:, 2] = [A[0,2], A[1,2]].
    # B_BL is B[2, 0:2] = [B[2,0], B[2,1]].
    # A_R B_BL is 2x2.
    # C_L += A_R B_BL.
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    # This requires 4 multiplications.
    
    # A_L B_TR is 2x1.
    # B_TR is B[0:2, 2] = [B[0,2], B[1,2]].
    # C_R[0] = C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C_R[1] = C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    # This requires 4 multiplications?
    # Can we share with Strassen?
    # Strassen terms involve B[0,0], B[0,1], B[1,0], B[1,1].
    # B_TR involves B[0,2], B[1,2]. No overlap.
    # So 4 multiplications.
    
    # A_R B_BR is 2x1.
    # B_BR is B[2,2].
    # C_R[0] += A[0,2]B[2,2]
    # C_R[1] += A[1,2]B[2,2]
    # This requires 2 multiplications.
    
    # Total: 7 + 4 + 4 + 2 = 17.
    
    # Let's build the matrices.
    
    # Helper to set U, V, W for a term c:
    # U[row, c] = coeff_A
    # V[row, c] = coeff_B
    # W[row, c] = coeff_C
    
    def add_term(c, u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx, c] += u_val
        V[v_idx, c] += v_val
        W[w_idx, c] += w_val

    # Term 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # A part: A[0,0] (idx 0), A[1,1] (idx 4)
    # B part: B[0,0] (idx 0), B[1,1] (idx 4)
    # C part:
    # C[0,0] += M1 - M2 - M4 + M7
    # C[0,1] += M3 + M5
    # C[1,0] += M2 + M4
    # C[1,1] += M1 - M3 - M5 + M6
    # This is complex to assign directly.
    # It is easier to define the linear combinations explicitly.
    
    # I will use a different approach: define the 17 products explicitly as linear combinations of A and B.
    # Then solve for U, V, W.
    
    # Products:
    # P0 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # P1 = (A[1,0]+A[1,1]) * B[0,0]
    # P2 = A[0,0] * (B[0,1]-B[1,1])
    # P3 = A[1,1] * (B[1,0]-B[0,0])
    # P4 = (A[0,0]+A[0,1]) * B[1,1]
    # P5 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # P6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # P7 = A[0,2] * B[2,0]
    # P8 = A[0,2] * B[2,1]
    # P9 = A[1,2] * B[2,0]
    # P10 = A[1,2] * B[2,1]
    # P11 = A[0,0] * B[0,2]
    # P12 = A[0,1] * B[1,2]
    # P13 = A[1,0] * B[0,2]
    # P14 = A[1,1] * B[1,2]
    # P15 = A[0,2] * B[2,2]
    # P16 = A[1,2] * B[2,2]
    
    # Now, express C[i,k] as sum of P_c.
    # C[0,0] = P0 - P1 - P3 + P6 + P7
    # C[0,1] = P2 + P4 + P8
    # C[1,0] = P1 + P3 + P9
    # C[1,1] = P0 - P2 - P4 + P5 + P10
    # C[0,2] = P11 + P12 + P15
    # C[1,2] = P13 + P14 + P16
    
    # We need to verify these formulas.
    # Strassen:
    # C11 = M1 - M2 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # My P0..P6 correspond to M1..M7?
    # P0=M1, P1=M2, P2=M3, P3=M4, P4=M5, P5=M6, P6=M7.
    
    # C[0,0] (C11) = P0 - P1 - P3 + P6.
    # Plus A_R B_BL part: A[0,2]B[2,0] = P7.
    # So C[0,0] = P0 - P1 - P3 + P6 + P7. Correct.
    
    # C[0,1] (C12) = P2 + P4.
    # Plus A[0,2]B[2,1] = P8.
    # So C[0,1] = P2 + P4 + P8. Correct.
    
    # C[1,0] (C21) = P1 + P3.
    # Plus A[1,2]B[2,0] = P9.
    # So C[1,0] = P1 + P3 + P9. Correct.
    
    # C[1,1] (C22) = P0 - P2 - P4 + P5.
    # Plus A[1,2]B[2,1] = P10.
    # So C[1,1] = P0 - P2 - P4 + P5 + P10. Correct.
    
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # = P11 + P12 + P15. Correct.
    
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # = P13 + P14 + P16. Correct.
    
    # Now we need to map these to U, V, W.
    # For each P_c, we have linear combinations of A and B.
    # P_c = (sum_i u_{ic} A_i) * (sum_j v_{jc} B_j).
    # And C_k = sum_c w_{kc} P_c.
    
    # Let's fill U, V, W.
    
    # P0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # A idx: 0, 4. B idx: 0, 4.
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    # C contributions:
    # C[0,0] (idx 0) += 1
    # C[1,1] (idx 3) += 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # P1: (A[1,0]+A[1,1]) * B[0,0]
    # A idx: 3, 4. B idx: 0.
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    # C[0,0] -= 1
    # C[1,0] += 1
    W[0, 1] = -1
    W[2, 1] = 1
    
    # P2: A[0,0] * (B[0,1]-B[1,1])
    # A idx: 0. B idx: 1, 4.
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    # C[0,1] += 1
    # C[1,1] -= 1
    W[1, 2] = 1
    W[3, 2] = -1
    
    # P3: A[1,1] * (B[1,0]-B[0,0])
    # A idx: 4. B idx: 3, 0.
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    # C[0,0] -= 1
    # C[1,0] += 1
    W[0, 3] = -1
    W[2, 3] = 1
    
    # P4: (A[0,0]+A[0,1]) * B[1,1]
    # A idx: 0, 1. B idx: 4.
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    # C[0,1] += 1
    # C[1,1] -= 1
    W[1, 4] = 1
    W[3, 4] = -1
    
    # P5: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # A idx: 3, 0. B idx: 0, 1.
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C[1,1] += 1
    W[3, 5] = 1
    
    # P6: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # A idx: 1, 4. B idx: 3, 4.
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    # C[0,0] += 1
    W[0, 6] = 1
    
    # P7: A[0,2] * B[2,0]
    # A idx: 2. B idx: 6.
    U[2, 7] = 1
    V[6, 7] = 1
    # C[0,0] += 1
    W[0, 7] = 1
    
    # P8: A[0,2] * B[2,1]
    # A idx: 2. B idx: 7.
    U[2, 8] = 1
    V[7, 8] = 1
    # C[0,1] += 1
    W[1, 8] = 1
    
    # P9: A[1,2] * B[2,0]
    # A idx: 5. B idx: 6.
    U[5, 9] = 1
    V[6, 9] = 1
    # C[1,0] += 1
    W[2, 9] = 1
    
    # P10: A[1,2] * B[2,1]
    # A idx: 5. B idx: 7.
    U[5, 10] = 1
    V[7, 10] = 1
    # C[1,1] += 1
    W[3, 10] = 1
    
    # P11: A[0,0] * B[0,2]
    # A idx: 0. B idx: 2.
    U[0, 11] = 1
    V[2, 11] = 1
    # C[0,2] += 1
    W[4, 11] = 1
    
    # P12: A[0,1] * B[1,2]
    # A idx: 1. B idx: 5.
    U[1, 12] = 1
    V[5, 12] = 1
    # C[0,2] += 1
    W[4, 12] = 1
    
    # P13: A[1,0] * B[0,2]
    # A idx: 3. B idx: 2.
    U[3, 13] = 1
    V[2, 13] = 1
    # C[1,2] += 1
    W[5, 13] = 1
    
    # P14: A[1,1] * B[1,2]
    # A idx: 4. B idx: 5.
    U[4, 14] = 1
    V[5, 14] = 1
    # C[1,2] += 1
    W[5, 14] = 1
    
    # P15: A[0,2] * B[2,2]
    # A idx: 2. B idx: 8.
    U[2, 15] = 1
    V[8, 15] = 1
    # C[0,2] += 1
    W[4, 15] = 1
    
    # P16: A[1,2] * B[2,2]
    # A idx: 5. B idx: 8.
    U[5, 16] = 1
    V[8, 16] = 1
    # C[1,2] += 1
    W[5, 16] = 1
    
    return {"U": U, "V": V, "W": W}

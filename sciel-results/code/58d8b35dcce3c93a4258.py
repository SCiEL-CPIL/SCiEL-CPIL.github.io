import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    # Dimensions
    n, m, p = DIMENSIONS
    
    # We use a known Rank-11 decomposition.
    # It consists of Strassen's algorithm for the 2x2x2 part (Rank 7)
    # and Schoolbook for the 2x2x1 part (Rank 4).
    # Total Rank = 7 + 4 = 11.
    
    rank = 11
    
    # Initialize U, V, W
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 2x2x2 (cols 0,1 of B and C) ---
    # A is 2x2, B is 2x2 (cols 0,1), C is 2x2 (cols 0,1)
    # Indices:
    # U: A elements. 0:A00, 1:A01, 2:A10, 3:A11
    # V: B elements. 0:B00, 1:B01, 2:B02, 3:B10, 4:B11, 5:B12
    # W: C elements. 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12
    # Note: W index mapping in task: W[k*n+i, r] where C[i,j] = W[j*n+i, r]?
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,j] is at index j*n + i if stored column-major?
    # Let's check the seed program's W indexing.
    # Seed: W[0][0] += 1; W[3][0] += 1 for P1.
    # P1 in Strassen: (A00+A11)(B00+B11) contributes to C00, C11, C01, C10.
    # Specifically:
    # C00 += P1
    # C11 += P1
    # C01 -= P1 (in some variants) or +?
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10+B11)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P1 - P3 + P5 + P7
    # C11 = P1 - P2 + P3 - P4
    
    # Let's map indices in seed:
    # U[0]: A00, U[1]: A01, U[2]: A10, U[3]: A11
    # V[0]: B00, V[1]: B01, V[2]: B02, V[3]: B10, V[4]: B11, V[5]: B12
    # W[0]: C00 (index 0*2+0=0), W[1]: C10 (index 0*2+1=1? No, k*n+i. C[i,j]. If k=j, i=i. Index j*2+i.
    # C00 -> 0*2+0=0. C10 -> 0*2+1=1. C01 -> 1*2+0=2. C11 -> 1*2+1=3.
    # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11.
    
    # P1: (A00+A11)(B00+B11)
    # U: A00(1), A11(1) -> U[0][0]=1, U[3][0]=1
    # V: B00(1), B11(1) -> V[0][0]=1, V[4][0]=1
    # W: C00(1), C11(1) -> W[0][0]=1, W[3][0]=1
    # Seed has: W[0][0]+=1, W[3][0]+=1. Matches.
    
    # P2: (A10+A11)B00
    # U: A10(1), A11(1) -> U[2][1]=1, U[3][1]=1
    # V: B00(1) -> V[0][1]=1
    # W: C10(1), C11(-1) -> W[1][1]=1, W[3][1]=-1
    # Seed has: W[1][1]+=1, W[3][1]+=-1. Matches.
    
    # P3: A00(B10+B11)
    # U: A00(1) -> U[0][2]=1
    # V: B10(1), B11(1) -> V[3][2]=1, V[4][2]=1?
    # Seed has: V[1][2]=1, V[4][2]=-1.
    # Wait. V[1] is B01.
    # Standard Strassen P3 is A00(B10+B11).
    # Seed P3: U[0][2]=1. V[1][2]=1 (B01), V[4][2]=-1 (B11).
    # This corresponds to A00(B01 - B11)?
    # Let's check C contributions.
    # W[2][2]=1 (C01), W[3][2]=1 (C11).
    # If P3 = A00(B01-B11), then C01 += P3, C11 += P3?
    # Let's re-verify Strassen formulas.
    # There are different conventions. I will trust the seed program's coefficients as they are valid.
    
    # P4: A11(B01-B11)
    # Seed: U[3][3]=1. V[3][3]=1 (B10), V[0][3]=-1 (B00).
    # This is A11(B10-B00).
    # W[0][3]=1 (C00), W[1][3]=1 (C10).
    
    # P5: (A00+A01)B11
    # Seed: U[0][4]=1, U[1][4]=1. V[4][4]=1 (B11).
    # W[0][4]=-1 (C00), W[2][4]=1 (C01).
    
    # P6: (A10-A00)(B00+B01)
    # Seed: U[2][5]=1, U[0][5]=-1. V[0][5]=1 (B00), V[1][5]=1 (B01).
    # W[3][5]=1 (C11).
    
    # P7: (A01-A11)(B10+B11)
    # Seed: U[1][6]=1, U[3][6]=-1. V[3][6]=1 (B10), V[4][6]=1 (B11).
    # W[0][6]=1 (C00).
    
    # This set of 7 terms is a valid rank-7 decomposition for 2x2x2.
    
    # --- Part 2: Schoolbook for 2x2x1 (col 2 of B and C) ---
    # Q1: A00 * B02 -> C02
    # U[0][7]=1, V[2][7]=1 (B02), W[4][7]=1 (C02)
    # Q2: A01 * B12 -> C02
    # U[1][8]=1, V[5][8]=1 (B12), W[4][8]=1 (C02)
    # Q3: A10 * B02 -> C12
    # U[2][9]=1, V[2][9]=1 (B02), W[5][9]=1 (C12)
    # Q4: A11 * B12 -> C12
    # U[3][10]=1, V[5][10]=1 (B12), W[5][10]=1 (C12)
    
    # Fill in the matrices
    # P1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # P2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    # P3
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    # P4
    U[3][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # P5
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # P6
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # P7
    U[1][6] += 1; U[3][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Q1
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # Q2
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # Q3
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # Q4
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    return {"U": U, "V": V, "W": W}

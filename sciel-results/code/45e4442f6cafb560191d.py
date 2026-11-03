import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set U, V, W for a term
    # term: list of (U_idx, U_val), (V_idx, V_val), (W_idx, W_val)
    # Actually easier to just define the terms explicitly.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00 (0,0), A11 (1,1) -> idx 0, 4
    # V: B00 (0,0), B11 (1,1) -> idx 0, 4
    # W: C00 (0,0), C01 (1,0), C10 (0,1), C11 (1,1)
    # W idx: k*2+i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # A10 (1,0)->3, A11 (1,1)->4
    # B00 (0,0)->0
    # Contributes to P10, P11
    # P10 -> C10 (W[1]), P11 -> C11 (W[3])
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # Term 2: M3 = A00(B01-B11)
    # A00 (0,0)->0
    # B01 (0,1)->1, B11 (1,1)->4
    # Contributes to P01, P11
    # P01 -> C01 (W[2]), P11 -> C11 (W[3])
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    # A11 (1,1)->4
    # B10 (1,0)->3, B00 (0,0)->0
    # Contributes to P00, P10
    # P00 -> C00 (W[0]), P10 -> C10 (W[1])
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    # A00 (0,0)->0, A01 (0,1)->1
    # B11 (1,1)->4
    # Contributes to P00, P01
    # P00 -> C00 (W[0]), P01 -> C01 (W[2])
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # A10 (1,0)->3, A00 (0,0)->0
    # B00 (0,0)->0, B01 (0,1)->1
    # Contributes to P11
    # P11 -> C11 (W[3])
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # A01 (0,1)->1, A11 (1,1)->4
    # B10 (1,0)->3, B11 (1,1)->4
    # Contributes to P00, P11
    # P00 -> C00 (W[0]), P11 -> C11 (W[3])
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    # Term 7: M8 = A00*B02
    # A00 (0,0)->0
    # B02 (0,2)->2
    # Contributes to Q0 -> C02 (W[4])
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # Term 8: M9 = A01*B12
    # A01 (0,1)->1
    # B12 (1,2)->5
    # Contributes to Q0 -> C02 (W[4])
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # Term 9: M10 = A10*B02
    # A10 (1,0)->3
    # B02 (0,2)->2
    # Contributes to Q1 -> C12 (W[5])
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Term 10: M11 = A11*B12
    # A11 (1,1)->4
    # B12 (1,2)->5
    # Contributes to Q1 -> C12 (W[5])
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    # Term 11: M12 = A02*B20
    # A02 (0,2)->2
    # B20 (2,0)->6
    # Contributes to R00 -> C00 (W[0])
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1
    
    # Term 12: M13 = A02*B21
    # A02 (0,2)->2
    # B21 (2,1)->7
    # Contributes to R01 -> C01 (W[2])
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1
    
    # Term 13: M14 = A02*B22
    # A02 (0,2)->2
    # B22 (2,2)->8
    # Contributes to R02 -> C02 (W[4])
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Term 14: M15 = A12*B20
    # A12 (1,2)->5
    # B20 (2,0)->6
    # Contributes to R10 -> C10 (W[1])
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1
    
    # Term 15: M16 = A12*B21
    # A12 (1,2)->5
    # B21 (2,1)->7
    # Contributes to R11 -> C11 (W[3])
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1
    
    # Term 16: M17 = A12*B22
    # A12 (1,2)->5
    # B22 (2,2)->8
    # Contributes to R12 -> C12 (W[5])
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

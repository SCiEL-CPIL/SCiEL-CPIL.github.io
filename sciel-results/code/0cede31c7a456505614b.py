import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen-like terms for the 2x2 sub-block
    # M1 = (A00 + A11)(B00 + B11)
    # A indices: 00->0, 11->4
    # B indices: 00->0, 11->4
    # C indices: 00->0, 11->3
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # A indices: 10->3, 11->4
    # B indices: 00->0
    # C indices: 10->1, 11->3 (Note: The seed has -1 for C11. Let's verify if this is part of a valid identity)
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10 - B11)
    # A indices: 00->0
    # B indices: 10->3, 11->4
    # C indices: 01->2, 11->3
    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    # A indices: 11->4
    # B indices: 01->1, 00->0
    # C indices: 00->0, 01->2
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    # A indices: 00->0, 01->1
    # B indices: 11->4
    # C indices: 02->4, 00->0 (Seed has W[2,4]=1, W[0,4]=-1. 2->C01, 0->C00)
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A10 - A00)(B00 + B01)
    # A indices: 10->3, 00->0
    # B indices: 00->0, 01->1
    # C indices: 11->3
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # A indices: 01->1, 11->4
    # B indices: 10->3, 11->4
    # C indices: 00->0
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook terms for remaining interactions
    # A02*B20 -> C00
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # A02*B21 -> C01
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # A12*B20 -> C10
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # A12*B21 -> C11
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    # A00*B02 -> C02
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    # A01*B12 -> C02
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    # A10*B02 -> C12
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    # A11*B12 -> C12
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    # A02*B22 -> C02
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    # A12*B22 -> C12
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

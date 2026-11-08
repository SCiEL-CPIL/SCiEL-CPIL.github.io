"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-56 decomposition for 2x4x8 matrix multiplication
    using the Kronecker product of M_{2,2,2} (Strassen) and M_{1,2,4} (Naive).
    """
    # 1. Define M_{2,2,2} Strassen decomposition
    # A is 2x2, B is 2x2, C is 2x2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7).
    # Indices: A: 0,1,2,3 (row-major). B: 0,1,2,3. C: 0,1,2,3.
    
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # Strassen products P0..P6
    # P0 = (A00+A11)*(B00+B11)
    # P1 = (A10+A11)*B00
    # P2 = A00*(B01-B11)
    # P3 = A11*(B10-B00)
    # P4 = (A00+A01)*B11
    # P5 = (A10-A00)*(B00+B01)
    # P6 = (A01-A11)*(B10+B11)
    
    # A factors for P0..P6
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U1[0, 0] += 1; U1[3, 0] += 1  # P0: A00, A11
    U1[2, 1] += 1; U1[3, 1] += 1  # P1: A10, A11
    U1[0, 2] += 1                 # P2: A00
    U1[3, 3] += 1                 # P3: A11
    U1[0, 4] += 1; U1[1, 4] += 1  # P4: A00, A01
    U1[2, 5] += 1; U1[0, 5] -= 1  # P5: A10, -A00
    U1[1, 6] += 1; U1[3, 6] -= 1  # P6: A01, -A11
    
    # B factors for P0..P6
    V1[0, 0] += 1; V1[3, 0] += 1  # P0: B00, B11
    V1[0, 1] += 1                 # P1: B00
    V1[1, 2] += 1; V1[3, 2] -= 1  # P2: B01, -B11
    V1[2, 3] += 1; V1[0, 3] -= 1  # P3: B10, -B00
    V1[3, 4] += 1                 # P4: B11
    V1[0, 5] += 1; V1[1, 5] += 1  # P5: B00, B01
    V1[2, 6] += 1; V1[3, 6] += 1  # P6: B10, B11
    
    # C reconstruction
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    W1[0, 0] += 1; W1[0, 3] += 1; W1[0, 4] -= 1; W1[0, 6] += 1
    W1[1, 2] += 1; W1[1, 4] += 1
    W1[2, 1] += 1; W1[2, 3] += 1
    W1[3, 0] += 1; W1[3, 1] -= 1; W1[3, 2] += 1; W1[3, 5] += 1

    # 2. Define M_{1,2,4} Naive decomposition
    # A is 1x2, B is 2x4, C is 1x4.
    # Rank 8.
    # U2 shape (2, 8), V2 shape (8, 8), W2 shape (4, 8).
    
    U2 = np.zeros((2, 8), dtype=int)
    V2 = np.zeros((8, 8), dtype=int)
    W2 = np.zeros((4, 8), dtype=int)
    
    for j in range(2):
        for k in range(4):
            idx = j * 4 + k
            U2[j, idx] = 1
            V2[j * 4 + k, idx] = 1
            W2[k, idx] = 1

    # 3. Kronecker Product
    # U = U1 ⊗ U2
    # V = V1 ⊗ V2
    # W = W1 ⊗ W2
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

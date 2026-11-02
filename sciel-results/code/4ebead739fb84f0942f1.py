import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0:(a,e,c11), 1:(b,f,c12), 2:(c,g,c21), 3:(d,h,c22)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, -1] # d
    ])
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, -1, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 1] # h
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c11
        [0, 0, 1, 0, 1, 0, 0], # c12
        [0, 1, 0, 1, 0, 0, 0], # c21
        [1, -1, 1, 0, 0, 1, 0] # c22
    ])
    
    # Kronecker product
    # U4 = U2 (x) U2
    # The row index of U4 is (i1 * 2 + i2, j1 * 2 + j2) ?
    # No, the input to 4x4 is A (4x4).
    # A is partitioned into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # The Kronecker product of tensors corresponds to the Kronecker product of matrices if the indices are aligned correctly.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The shapes are 16x49.
    # We need to return lists of lists.
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
# EVOLVE-BLOCK-END

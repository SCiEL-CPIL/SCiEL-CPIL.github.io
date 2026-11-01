import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Rank-7 Strassen decomposition for 2x2 matrix multiplication.
    # Return factor matrices as numpy arrays with float64 entries.
    # U: (4,7) indexing A[i,j] as row i*2+j
    # V: (4,7) indexing B[j,k] as row j*2+k
    # W: (4,7) indexing C[i,k] output as row k*2+i (per task convention W[k*n+i,r])
    
    U = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],
        [ 0,  0,  0,  0,  1,  0,  1],
        [ 0,  1,  0,  0,  0,  1,  0],
        [ 1,  1,  0,  1,  0,  0, -1]
    ], dtype=np.float64)
    
    V = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],
        [ 0,  0,  1,  0,  0,  1,  0],
        [ 0,  0,  0,  1,  0,  0,  1],
        [ 1,  0, -1,  0,  1,  0,  1]
    ], dtype=np.float64)
    
    W = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],
        [ 0,  1,  0,  1,  0,  0,  0],
        [ 0,  0,  1,  0,  1,  0,  0],
        [ 1, -1,  1,  0,  0,  1,  0]
    ], dtype=np.float64)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

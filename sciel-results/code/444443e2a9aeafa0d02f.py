import numpy as np

def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # U: (4, 7) corresponds to A flattened (row-major)
    # V: (4, 7) corresponds to B flattened (row-major)
    # W: (4, 7) corresponds to C flattened (row-major)
    
    # U rows:
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[1,0]
    # 3: A[1,1]
    
    # V rows:
    # 0: B[0,0]
    # 1: B[0,1]
    # 2: B[1,0]
    # 3: B[1,1]
    
    # W rows:
    # 0: C[0,0]
    # 1: C[0,1]
    # 2: C[1,0]
    # 3: C[1,1]
    
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A[0,0]
        [0,  0,  0,  0,  1,  0,  1],  # A[0,1]
        [0,  1,  0,  0,  0,  1, -1],  # A[1,0]
        [1,  1,  0,  1,  0,  0,  0]   # A[1,1]
    ]
    
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B[0,0]
        [0,  0,  1,  0,  0,  1,  0],  # B[0,1]
        [0,  0,  0,  1,  0,  0,  1],  # B[1,0]
        [1,  0, -1, 0,  1,  0,  1]   # B[1,1]
    ]
    
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C[0,0]
        [0,  0,  1,  0,  1,  0,  0],  # C[0,1]
        [0,  1,  0,  1,  0,  0,  0],  # C[1,0]
        [1, -1,  1,  0,  0,  1,  0]   # C[1,1]
    ]
    
    return {"U": U, "V": V, "W": W}

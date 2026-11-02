import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal U, V, W
        # This function is not ideal because we need to add to existing columns.
        # We will just build the arrays directly.
        pass

    # We will build the 39 columns.
    # Columns 0-6: Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    # Columns 7-38: Schoolbook for the rest
    
    # Strassen Mapping
    # U indices: 0:(0,0), 1:(0,1), 4:(1,0), 5:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 5:(1,0), 6:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # P1
    U[0, 0] = 1; U[5, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2
    U[4, 1] = 1; U[5, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # P3
    U[0, 2] = 1
    V[1, 2] = 1; V[6, 2] = -1
    W[1, 2] = 1
    
    # P4
    U[5, 3] = 1
    V[5, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5
    U[0, 4] = 1; U[1, 4] = 1
    V[6, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6
    U[4, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1
    
    # P7
    U[1, 6] = 1; U[5, 6] = -1
    V[5, 6] = 1; V[6, 6] = 1
    W[1, 6] = -1; W[3, 6] = 1
    
    # Schoolbook for remaining 32 terms
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

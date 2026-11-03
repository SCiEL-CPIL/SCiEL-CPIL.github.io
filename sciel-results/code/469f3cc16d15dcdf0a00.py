import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 * 2x2
    U1 = [
        [1, 0, 1, 0, 1, -1, -1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook decomposition for 1x2 * 2x4
    U2 = [
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ]
    V2 = [
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1]
    ]
    
    U1_np = np.array(U1)
    V1_np = np.array(V1)
    W1_np = np.array(W1)
    
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    U_total = np.kron(U1_np, U2_np)
    V_total = np.kron(V1_np, V2_np)
    W_total = np.kron(W1_np, W2_np)
    
    # Convert to list of lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

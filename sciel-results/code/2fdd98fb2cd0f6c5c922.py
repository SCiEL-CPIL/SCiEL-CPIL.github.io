<<<<<<< SEARCH
    # Left block (columns 0-3 of output)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :] # Wait, W index is k*n+i. n=4.
            # W4 index is k*4+i.
            # So W[k*4+i, :] = W4[k*4+i, :]
            
    # Right block (columns 4-7 of output)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # Left block (columns 0-3 of output)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    # W for left block: W[k*4+i, 0:49] = W4[k*4+i, 0:49] for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Right block (columns 4-7 of output)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
    # W for right block: W[(4+k)*4+i, 49:98] = W4[k*4+i, 49:98] for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W4[k * 4 + i, 49:98]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

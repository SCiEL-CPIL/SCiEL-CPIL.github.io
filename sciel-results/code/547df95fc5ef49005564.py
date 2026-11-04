<<<<<<< SEARCH
def construct():
    # Rank-6 decomposition derived from the structure of the tensor
    # T[i, j, k] = 1 iff (i // 3 == k % 2) and (i % 3 == j // 3)
    # This tensor is the direct sum of 3 blocks (for b=0,1,2), each of rank 2.
    rank = 6
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))

    # Block b=0: i in {0,3}, j in {0,1,2}, k in {0,2,4} (for a=0) and {1,3,5} (for a=1)
    # Col 0: i=0, j in {0,1,2}, k in {0,2,4}
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[4, 0] = 1

    # Col 1: i=3, j in {0,1,2}, k in {1,3,5}
    U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1; V[2, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1; W[5, 1] = 1

    # Block b=1: i in {1,4}, j in {3,4,5}, k in {0,2,4} (for a=0) and {1,3,5} (for a=1)
    # Col 2: i=1, j in {3,4,5}, k in {0,2,4}
    U[1, 2] = 1
    V[3, 2] = 1; V[4, 2] = 1; V[5, 2] = 1
    W[0, 2] = 1; W[2, 2] = 1; W[4, 2] = 1

    # Col 3: i=4, j in {3,4,5}, k in {1,3,5}
    U[4, 3] = 1
    V[3, 3] = 1; V[4, 3] = 1; V[5, 3] = 1
    W[1, 3] = 1; W[3, 3] = 1; W[5, 3] = 1

    # Block b=2: i in {2,5}, j in {6,7,8}, k in {0,2,4} (for a=0) and {1,3,5} (for a=1)
    # Col 4: i=2, j in {6,7,8}, k in {0,2,4}
    U[2, 4] = 1
    V[6, 4] = 1; V[7, 4] = 1; V[8, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1; W[4, 4] = 1

    # Col 5: i=5, j in {6,7,8}, k in {1,3,5}
    U[5, 5] = 1
    V[6, 5] = 1; V[7, 5] = 1; V[8, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1; W[5, 5] = 1

    # Validate
    T = _build_tensor()
    if _validate(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-17 if validation fails (should not happen)
    U17, V17, W17 = _rank17()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
=======
def construct():
    # First, try to find a rank-16 decomposition with a bounded search
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit
    
    result16 = _try_rank16(deadline)
    if result16 is not None:
        U, V, W = result16
        T = _build_tensor()
        if _validate(U, V, W, T, atol=1e-7):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-17 Strassen decomposition
    U17, V17, W17 = _rank17()
    T = _build_tensor()
    if _validate(U17, V17, W17, T):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Final fallback: simple rank-18 schoolbook
    rank = 18
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            jb, kb = divmod(j, P)
            if ja == jb:
                k = kb * N + ia
                c = i * P + j  # unique column index
                U[i, c] = 1
                V[j, c] = 1
                W[k, c] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

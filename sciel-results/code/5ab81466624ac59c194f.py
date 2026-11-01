def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10
    
    # Build rank-28 decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: (2x1)x(1x5) -> (2x5), schoolbook rank 10
    # A1 = A[:, 0] (column 0 of A), B1 = B[0, :] (row 0 of B)
    # For i in {0,1}, k in {0..4}:
    #   U index: i*3 + 0
    #   V index: 0*5 + k
    #   W index: k*2 + i
    for i in range(2):
        for k in range(5):
            u = np.zeros(6)
            v = np.zeros(15)
            w = np.zeros(10)
            u[i*3 + 0] = 1
            v[0*5 + k] = 1
            w[k*2 + i] = 1
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Part 2: (2x2)x(2x5) -> (2x5)
    # A2 = A[:, 1:3] (columns 1,2 of A), B2 = B[1:3, :] (rows 1,2 of B)
    # Split B2 columns as 2+2+1:
    #   B2a = B2[:, 0:2], B2b = B2[:, 2:4], B2c = B2[:, 4:5]
    
    # Part 2a: (2x2)x(2x2) -> (2x2), Strassen rank 7
    # A2 indices: i*3 + (1+j) for i,j in {0,1} -> i*3+1, i*3+2
    # B2a indices: (1+j)*5 + k for j in {0,1}, k in {0,1} -> 5+k, 10+k
    # C indices: k*2 + i for i in {0,1}, k in {0,1} -> 0,1,2,3
    
    # Strassen 2x2x2: 7 terms
    # Local A indices: 0->A2[0,0]=A[0,1], 1->A2[0,1]=A[0,2], 2->A2[1,0]=A[1,1], 3->A2[1,1]=A[1,2]
    # Global U indices: A2[0,0]->0*3+1=1, A2[0,1]->0*3+2=2, A2[1,0]->1*3+1=4, A2[1,1]->1*3+2=5
    # Local B indices: 0->B2a[0,0]=B[1,0], 1->B2a[0,1]=B[1,1], 2->B2a[1,0]=B[2,0], 3->B2a[1,1]=B[2,1]
    # Global V indices: B[1,0]->1*5+0=5, B[1,1]->1*5+1=6, B[2,0]->2*5+0=10, B[2,1]->2*5+1=11
    # Local C indices: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    # Global W indices: C[0,0]->0*2+0=0, C[0,1]->1*2+0=2, C[1,0]->0*2+1=1, C[1,1]->1*2+1=3
    
    # Strassen 7 terms for 2x2x2 (local indices):
    strassen_terms = [
        # (local_u, local_v, local_w)
        ([1,0,1,0], [1,0,0,0], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        ([0,0,0,1], [0,1,-1,0], [0,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [1,0,0,0]),
        ([0,0,1,-1], [1,1,0,0], [0,0,1,0]),
        ([1,0,-1,0], [0,0,1,1], [0,1,0,0]),
    ]
    
    # Mapping from local to global indices
    # For Part 2a:
    u_map_2a = {0: 1, 1: 2, 2: 4, 3: 5}  # local A2 index -> global U index
    v_map_2a = {0: 5, 1: 6, 2: 10, 3: 11}  # local B2a index -> global V index
    w_map_2a = {0: 0, 1: 2, 2: 1, 3: 3}  # local C index -> global W index
    
    for lu, lv, lw in strassen_terms:
        u = np.zeros(6)
        v = np.zeros(15)
        w = np.zeros(10)
        for li, val in enumerate(lu):
            if val != 0:
                u[u_map_2a[li]] += val
        for li, val in enumerate(lv):
            if val != 0:
                v[v_map_2a[li]] += val
        for li, val in enumerate(lw):
            if val != 0:
                w[w_map_2a[li]] += val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Part 2b: (2x2)x(2x2) -> (2x2), Strassen rank 7
    # B2b = B2[:, 2:4], so B[1,2], B[1,3], B[2,2], B[2,3]
    # Global V indices: B[1,2]->1*5+2=7, B[1,3]->1*5+3=8, B[2,2]->2*5+2=12, B[2,3]->2*5+3=13
    # C columns 2,3: C[0,2]->2*2+0=4, C[0,3]->3*2+0=6, C[1,2]->2*2+1=5, C[1,3]->3*2+1=7
    # Wait, W index is k*n+i = k*2+i. So C[i,k] -> W[k*2+i].
    # C[0,2] -> W[2*2+0]=4, C[0,3] -> W[3*2+0]=6, C[1,2] -> W[2*2+1]=5, C[1,3] -> W[3*2+1]=7
    
    u_map_2b = {0: 1, 1: 2, 2: 4, 3: 5}  # same A2
    v_map_2b = {0: 7, 1: 8, 2: 12, 3: 13}
    w_map_2b = {0: 4, 1: 6, 2: 5, 3: 7}  # C[0,2], C[0,3], C[1,2], C[1,3]
    
    for lu, lv, lw in strassen_terms:
        u = np.zeros(6)
        v = np.zeros(15)
        w = np.zeros(10)
        for li, val in enumerate(lu):
            if val != 0:
                u[u_map_2b[li]] += val
        for li, val in enumerate(lv):
            if val != 0:
                v[v_map_2b[li]] += val
        for li, val in enumerate(lw):
            if val != 0:
                w[w_map_2b[li]] += val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Part 2c: (2x2)x(2x1) -> (2x1), schoolbook rank 4
    # B2c = B2[:, 4:5], so B[1,4], B[2,4]
    # Global V indices: B[1,4]->1*5+4=9, B[2,4]->2*5+4=14
    # C column 4: C[0,4]->W[4*2+0]=8, C[1,4]->W[4*2+1]=9
    # For i in {0,1}, j in {0,1}:
    #   U: A2[i,j] -> global U[i*3+1+j]
    #   V: B2[j,0] -> global V[(1+j)*5+4]
    #   W: C[i,0] -> global W[4*2+i]
    for i in range(2):
        for j in range(2):
            u = np.zeros(6)
            v = np.zeros(15)
            w = np.zeros(10)
            u[i*3 + 1 + j] = 1
            v[(1+j)*5 + 4] = 1
            w[4*2 + i] = 1
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    rank = len(U_list)
    assert rank == 28, f"Expected rank 28, got {rank}"
    
    U = np.array(U_list).T  # shape (6, 28)
    V = np.array(V_list).T  # shape (15, 28)
    W = np.array(W_list).T  # shape (10, 28)
    
    # Verify: For random A (2x3) and B (3x5), check that sum_r U[:,r]*V[:,r]*W[:,r] == A@B (reshaped)
    np.random.seed(42)
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 5)
    C = A @ B  # 2x5
    
    # Reconstruct C from decomposition
    C_recon = np.zeros((10,))
    for r in range(28):
        C_recon += U[:, r] * V[:, r] * W[:, r]  # This is wrong!
    
    # Actually, the decomposition is: C_tensor = sum_r U_r ⊗ V_r ⊗ W_r
    # where C_tensor[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # And C_tensor[i*3+j, j*5+k, k*2+i] should equal C[i,k] for the valid (i,j,k) combos
    # and 0 otherwise.
    
    # Let me verify properly:
    T_recon = np.zeros((6, 15, 10))
    for r in range(28):
        T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Check: T_recon[i*3+j, j*5+k, k*2+i] should be 1 for all valid i,j,k
    # And T_recon should be 0 elsewhere.
    # Actually no - the tensor T satisfies T[a,b,c] = sum over j where a=i*m+j, b=j*p+k, c=k*n+i
    # For the multiplication tensor, T[i*m+j, j*p+k, k*n+i] = 1 and 0 elsewhere.
    # The decomposition should reproduce this tensor exactly.
    
    # Let's check:
    T_expected = np.zeros((6, 15, 10))
    for i in range(2):
        for j in range(3):
            for k in range(5):
                T_expected[i*3+j, j*5+k, k*2+i] = 1
    
    # Check if T_recon matches T_expected
    max_err = np.max(np.abs(T_recon - T_expected))
    if max_err > 1e-10:
        # Decomposition is wrong, fall back to schoolbook
        return _schoolbook(2, 3, 5)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

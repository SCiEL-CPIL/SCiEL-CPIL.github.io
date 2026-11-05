import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # The multiplication tensor T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In the flattened index convention:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor is a 3-way tensor of size (nm, mp, pn) but it's sparse in the sense that
    # only specific (u_idx, v_idx, w_idx) triples are non-zero in the standard basis.
    
    # We want to find U, V, W of rank R such that:
    # sum_r U[u, r] * V[v, r] * W[w, r] = T[u, v, w]
    # where T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some valid i,j,k.
    
    # Let's build the target tensor T as a dense array of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Strategy: Start with a known valid decomposition (e.g., schoolbook rank 100)
    # and try to compress it.
    
    # Schoolbook decomposition:
    # U is identity-like on (i,j), V on (j,k), W on (i,k)
    # Actually, schoolbook is:
    # For each (i,j,k):
    #   u_vec has 1 at (i*m+j)
    #   v_vec has 1 at (j*p+k)
    #   w_vec has 1 at (k*n+i)
    
    # We can start with rank 100 and try to reduce.
    # However, a better starting point is a Strassen-based decomposition if we can construct one quickly,
    # or we can use a randomized SVD-like approach on the tensor.
    
    # Let's try a different approach: 
    # The rank of the 4x5x5 multiplication tensor is known to be less than 100.
    # Let's try to find a decomposition of rank R by solving a system of equations.
    # For a fixed rank R, we have 20*R + 25*R + 20*R = 65*R variables.
    # The constraints are T[u,v,w] = sum_r U[u,r]V[v,r]W[w,r] for all u,v,w.
    # This is a non-linear system.
    
    # Alternative: Use the fact that the tensor can be viewed as a matrix multiplication.
    # Let's try a greedy algorithm to build the decomposition term by term.
    # 1. Initialize residual R = T.
    # 2. Find the best rank-1 approximation of R.
    # 3. Subtract it from R.
    # 4. Repeat until R is close to zero.
    # 5. Try to merge or eliminate terms if possible.
    
    # Finding the best rank-1 approximation of a 3-way tensor is NP-hard in general,
    # but we can use a heuristic:
    # For each pair of modes, flatten and use SVD.
    
    # Let's try a simpler heuristic: 
    # We know that Strassen reduces the rank.
    # Let's try to construct a decomposition using a block structure that allows for Strassen-like savings.
    
    # Since hard-coding Strassen is complex, let's try to use a numerical search for a low-rank decomposition.
    # We'll start with a random initialization of U, V, W with a small rank (e.g., 50)
    # and optimize them to minimize ||T - U V W||_F.
    
    # However, optimizing 3-way tensor rank is difficult.
    
    # Let's go back to the structural idea but make it more systematic.
    # The current program uses two 2x2x2 Strassen blocks and schoolbook for the rest.
    # Rank = 7 + 7 + (100 - 8 - 8) = 14 + 84 = 98? No, the schoolbook part covers the remaining terms.
    # The current program's rank calculation:
    # Block 1: 7 terms
    # Block 2: 7 terms
    # Remaining: 100 - 8 (block 1) - 8 (block 2) = 84 terms?
    # Wait, the Strassen blocks cover 8 terms each in the tensor, but use 7 rank-1 terms.
    # So we save 1 rank per block.
    # Total rank = 100 - 2 = 98.
    
    # Can we do better?
    # What if we use a 4x4x4 Strassen block?
    # 4x4x4 Strassen has rank 49.
    # It covers 64 terms.
    # Remaining terms: 100 - 64 = 36.
    # Total rank: 49 + 36 = 85.
    
    # Let's implement the 4x4x4 Strassen block approach.
    # This is a known structural improvement.
    
    def idx_A(i, j): return i * m + j
    def idx_B(j, k): return j * p + k
    def idx_C(i, k): return k * n + i
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0

    def add_rank1(u_entries, v_entries, w_entries, coeff=1):
        nonlocal rank
        for ui, uv in u_entries.items():
            U_dict[(ui, rank)] = U_dict.get((ui, rank), 0) + uv * coeff
        for vi, vv in v_entries.items():
            V_dict[(vi, rank)] = V_dict.get((vi, rank), 0) + vv * coeff
        for wi, wv in w_entries.items():
            W_dict[(wi, rank)] = W_dict.get((wi, rank), 0) + wv * coeff
        rank += 1

    # 4x4x4 Strassen decomposition
    # This is a rank-49 decomposition for 4x4x4.
    # It is derived from two 2x2x2 Strassen blocks.
    # Let's implement it by calling the 2x2x2 Strassen on the four 2x2x2 sub-blocks
    # of the 4x4x4 tensor.
    
    def strassen_2x2(X_u, Y_v, Z_w, coeff=1):
        # M1: U: {X00:1, X11:1}, V: {Y00:1, Y11:1}, W: {Z00:1, Z11:1}
        add_rank1({X_u[0][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
        
        # M2: U: {X10:1, X11:1}, V: {Y00:1}, W: {Z10:1}
        add_rank1({X_u[1][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1},
                  {Z_w[1][0]: 1}, coeff)
        
        # M3: U: {X00:1}, V: {Y01:1, Y11:-1}, W: {Z01:1}
        add_rank1({X_u[0][0]: 1},
                  {Y_v[0][1]: 1, Y_v[1][1]: -1},
                  {Z_w[0][1]: 1}, coeff)
        
        # M4: U: {X11:1}, V: {Y10:1, Y00:-1}, W: {Z10:1}
        add_rank1({X_u[1][1]: 1},
                  {Y_v[1][0]: 1, Y_v[0][0]: -1},
                  {Z_w[1][0]: 1}, coeff)
        
        # M5: U: {X00:1, X01:1}, V: {Y11:1}, W: {Z01:1}
        add_rank1({X_u[0][0]: 1, X_u[0][1]: 1},
                  {Y_v[1][1]: 1},
                  {Z_w[0][1]: 1}, coeff)
        
        # M6: U: {X10:1, X00:-1}, V: {Y00:1, Y01:1}, W: {Z11:1}
        add_rank1({X_u[1][0]: 1, X_u[0][0]: -1},
                  {Y_v[0][0]: 1, Y_v[0][1]: 1},
                  {Z_w[1][1]: 1}, coeff)
        
        # M7: U: {X01:1, X11:-1}, V: {Y10:1, Y11:1}, W: {Z00:1, Z11:1}
        add_rank1({X_u[0][1]: 1, X_u[1][1]: -1},
                  {Y_v[1][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)

    # Apply Strassen to the four 2x2x2 sub-blocks of the 4x4x4 core
    # Sub-block 0: i in 0-1, j in 0-1, k in 0-1
    X0 = [[idx_A(0,0), idx_A(0,1)], [idx_A(1,0), idx_A(1,1)]]
    Y0 = [[idx_B(0,0), idx_B(0,1)], [idx_B(1,0), idx_B(1,1)]]
    Z0 = [[idx_C(0,0), idx_C(0,1)], [idx_C(1,0), idx_C(1,1)]]
    strassen_2x2(X0, Y0, Z0, 1)

    # Sub-block 1: i in 0-1, j in 0-1, k in 2-3
    X1 = [[idx_A(0,0), idx_A(0,1)], [idx_A(1,0), idx_A(1,1)]]
    Y1 = [[idx_B(0,2), idx_B(0,3)], [idx_B(1,2), idx_B(1,3)]]
    Z1 = [[idx_C(0,2), idx_C(0,3)], [idx_C(1,2), idx_C(1,3)]]
    strassen_2x2(X1, Y1, Z1, 1)

    # Sub-block 2: i in 0-1, j in 2-3, k in 0-1
    X2 = [[idx_A(0,2), idx_A(0,3)], [idx_A(1,2), idx_A(1,3)]]
    Y2 = [[idx_B(2,0), idx_B(2,1)], [idx_B(3,0), idx_B(3,1)]]
    Z2 = [[idx_C(0,0), idx_C(0,1)], [idx_C(1,0), idx_C(1,1)]]
    strassen_2x2(X2, Y2, Z2, 1)

    # Sub-block 3: i in 0-1, j in 2-3, k in 2-3
    X3 = [[idx_A(0,2), idx_A(0,3)], [idx_A(1,2), idx_A(1,3)]]
    Y3 = [[idx_B(2,2), idx_B(2,3)], [idx_B(3,2), idx_B(3,3)]]
    Z3 = [[idx_C(0,2), idx_C(0,3)], [idx_C(1,2), idx_C(1,3)]]
    strassen_2x2(X3, Y3, Z3, 1)

    # Now we have covered i in 0-1, j in 0-3, k in 0-3.
    # We still need to cover:
    # 1. i in 0-1, j in 4, k in 0-4
    # 2. i in 2-3, j in 0-4, k in 0-4
    
    # Let's use schoolbook for the remaining parts.
    
    # Part 1: i in 0-1, j=4, k in 0-4
    for i in range(2):
        for k in range(5):
            add_rank1({idx_A(i,4): 1}, {idx_B(4,k): 1}, {idx_C(i,k): 1}, 1)
            
    # Part 2: i in 2-3, j in 0-4, k in 0-4
    for i in range(2,4):
        for j in range(5):
            for k in range(5):
                add_rank1({idx_A(i,j): 1}, {idx_B(j,k): 1}, {idx_C(i,k): 1}, 1)

    # Convert to arrays
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for (row, col), val in U_dict.items():
        U[row, col] = val
    for (row, col), val in V_dict.items():
        V[row, col] = val
    for (row, col), val in W_dict.items():
        W[row, col] = val

    # Quick verification
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_recon[i, k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]

        if not np.allclose(C_recon, C_expected, atol=1e-10):
            # Fallback to schoolbook
            U_dict = {}
            V_dict = {}
            W_dict = {}
            rank = 0

            def add_term_fb(u_row, v_row, w_row, val=1):
                nonlocal rank
                U_dict[(u_row, rank)] = U_dict.get((u_row, rank), 0) + val
                V_dict[(v_row, rank)] = V_dict.get((v_row, rank), 0) + val
                W_dict[(w_row, rank)] = W_dict.get((w_row, rank), 0) + val
                rank += 1

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_term_fb(i * m + j, j * p + k, k * n + i, 1)

            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for (row, col), val in U_dict.items():
                U[row, col] = val
            for (row, col), val in V_dict.items():
                V[row, col] = val
            for (row, col), val in W_dict.items():
                W[row, col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

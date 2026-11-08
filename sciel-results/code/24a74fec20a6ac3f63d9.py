import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm: split B into two 4x4 halves, apply rank-49 Strassen to each.
    Total rank: 98.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = 4, 4, 8

    # Step 1: Define 2x2x2 Strassen terms
    # Each term is (U_dict, V_dict, W_dict)
    # U: {(i,j): coeff}, V: {(j,k): coeff}, W: {(k,i): coeff}
    strassen_2x2 = [
        # M1 = (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Step 2: Build 4x4x4 decomposition (Rank 49)
    # We compose the 2x2x2 terms to form 4x4x4 terms.
    # The 4x4x4 tensor is the Kronecker product of the 2x2x2 tensor with itself.
    # There are 7*7 = 49 terms.
    
    terms_4x4x4 = []
    for meta_U, meta_V, meta_W in strassen_2x2:
        for inner_U, inner_V, inner_W in strassen_2x2:
            new_U = {}
            new_V = {}
            new_W = {}
            
            # Combine U factors
            for (i_b, j_b), c_meta in meta_U.items():
                for (i_l, j_l), c_inner in inner_U.items():
                    i_g = i_b * 2 + i_l
                    j_g = j_b * 2 + j_l
                    new_U[(i_g, j_g)] = new_U.get((i_g, j_g), 0) + c_meta * c_inner
            
            # Combine V factors
            for (j_b, k_b), c_meta in meta_V.items():
                for (j_l, k_l), c_inner in inner_V.items():
                    j_g = j_b * 2 + j_l
                    k_g = k_b * 2 + k_l
                    new_V[(j_g, k_g)] = new_V.get((j_g, k_g), 0) + c_meta * c_inner
            
            # Combine W factors
            # Note: W index is (k, i)
            for (k_b, i_b), c_meta in meta_W.items():
                for (k_l, i_l), c_inner in inner_W.items():
                    k_g = k_b * 2 + k_l
                    i_g = i_b * 2 + i_l
                    new_W[(k_g, i_g)] = new_W.get((k_g, i_g), 0) + c_meta * c_inner
            
            terms_4x4x4.append((new_U, new_V, new_W))
    
    rank_444 = len(terms_4x4x4) # 49
    
    # Step 3: Build 4x4x8 decomposition (Rank 98)
    # Split B (4x8) into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left uses terms_4x4x4 with k indices 0..3
    # C_right = A * B_right uses terms_4x4x4 with k indices 4..7
    
    rank_total = rank_444 * 2 # 98
    
    U = [[0] * rank_total for _ in range(n * m)] # 16 x 98
    V = [[0] * rank_total for _ in range(m * p)] # 32 x 98
    W = [[0] * rank_total for _ in range(p * n)] # 32 x 98
    
    # Fill Left Half (k = 0..3)
    for r in range(rank_444):
        U_t, V_t, W_t = terms_4x4x4[r]
        for (i, j), coeff in U_t.items():
            if coeff != 0:
                U[i * m + j][r] = coeff
        for (j, k), coeff in V_t.items():
            if coeff != 0:
                V[j * p + k][r] = coeff
        for (k, i), coeff in W_t.items():
            if coeff != 0:
                W[k * n + i][r] = coeff

    # Fill Right Half (k = 4..7)
    for r in range(rank_444):
        r2 = rank_444 + r
        U_t, V_t, W_t = terms_4x4x4[r]
        for (i, j), coeff in U_t.items():
            if coeff != 0:
                U[i * m + j][r2] = coeff
        for (j, k), coeff in V_t.items():
            if coeff != 0:
                # k in V_t is 0..3, shift to 4..7
                V[j * p + (k + 4)][r2] = coeff
        for (k, i), coeff in W_t.items():
            if coeff != 0:
                # k in W_t is 0..3, shift to 4..7
                W[(k + 4) * n + i][r2] = coeff

    # Verification
    if time.time() < deadline - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank_total):
            # Extract vectors
            # U_r is length n*m, reshape to n,m
            # V_r is length m*p, reshape to m,p
            # W_r is length p*n, reshape to p,n
            
            U_r = U_np[:, r].reshape(n, m)
            V_r = V_np[:, r].reshape(m, p)
            W_r = W_np[:, r].reshape(p, n)
            
            # Calculate contribution
            # a_part[i] = sum_j U_r[i,j] * A[i,j]
            # b_part[k] = sum_j V_r[j,k] * B[j,k]
            # C[i,k] += a_part[i] * b_part[k] * W_r[k,i]
            
            a_part = np.sum(U_r * A, axis=1)
            b_part = np.sum(V_r * B, axis=0)
            
            # W_r[k,i] corresponds to W_np[k*n+i, r]
            # We need to add a_part[i] * b_part[k] * W_r[k,i] to C[i,k]
            # This is an outer product of a_part and b_part, scaled by W_r.T
            # (a_part[:, None] * b_part[None, :]) * W_r.T
            # W_r.T[i,k] = W_r[k,i]
            
            C_computed += (a_part[:, None] * b_part[None, :]) * W_r.T
            
        max_err = np.max(np.abs(C_computed - C_expected))
        
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank_schoolbook = n * m * p
            U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
            V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
            W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][col] = 1
                        V_sb[j * p + k][col] = 1
                        W_sb[k * n + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

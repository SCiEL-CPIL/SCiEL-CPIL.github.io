import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # 2. Schoolbook fallback (Rank 64)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # 3. Strassen Block Decomposition (Rank 56) - The current best known simple construction
    def get_strassen_block_decomp():
        # Strassen for 2x2 * 2x2 has rank 7.
        # We have 4 such blocks?
        # A is 2x4, B is 4x8.
        # Split A into A1 (2x2), A2 (2x2).
        # Split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x2).
        # C1 = A1 B1 + A2 B2
        # C2 = A1 B3 + A2 B4
        # C3 = A1 B5 + A2 B6 (wait, B is 4x8, so 4 blocks of 2x2)
        # C4 = A1 B7 + A2 B8 (wait, B has 4 columns of 2x2? No, B is 4x8.
        # B columns 0-1: B_0 (2x2), cols 2-3: B_1, cols 4-5: B_2, cols 6-7: B_3.
        # C = [C_0, C_1, C_2, C_3] where C_k is 2x2.
        # C_0 = A1 B_0 + A2 B_2 ? No.
        # A = [A1 A2], B = [B_0 B_1; B_2 B_3] ?
        # A (2x4) * B (4x8).
        # Row 0 of A: A[0,0], A[0,1], A[0,2], A[0,3]
        # Row 1 of A: A[1,0], A[1,1], A[1,2], A[1,3]
        # Let's partition A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3).
        # Let's partition B into two 2x4 blocks? Or four 2x2 blocks?
        # B rows 0,1 vs 2,3.
        # B = [B_TL B_TR; B_BL B_BR] where each is 2x4.
        # This doesn't simplify to 2x2 products directly without further splitting.
        
        # The seed program used 8 blocks of 2x2*2x2.
        # How?
        # A is 2x4. B is 4x8.
        # For each k in 0..7 (output col), and i in 0..1 (output row):
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The seed decomposed this into 8 independent 2x2*2x2 Strassen multiplications.
        # This implies it treated the problem as 8 separate rank-7 problems.
        # Total rank 56.
        
        # We will reconstruct the 56-rank solution to use as a baseline.
        U_56 = np.zeros((8, 56))
        V_56 = np.zeros((32, 56))
        W_56 = np.zeros((16, 56))
        
        # Strassen 2x2 matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map the 2x2 Strassen to the global indices.
        # The seed code did this:
        # 8 blocks.
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ?
        # Let's just use the seed's hardcoded 56-rank solution if we can't derive it faster.
        # Actually, I will copy the seed's 56-rank construction logic to ensure a valid baseline.
        
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7))
        
        U_22[0, 0] = 1; U_22[3, 0] = 1
        V_22[0, 0] = 1; V_22[3, 0] = 1
        W_22[0, 0] = 1; W_22[3, 0] = 1
        U_22[2, 1] = 1; U_22[3, 1] = 1
        V_22[0, 1] = 1
        W_22[2, 1] = 1; W_22[3, 1] = -1
        U_22[0, 2] = 1
        V_22[1, 2] = 1; V_22[3, 2] = -1
        W_22[1, 2] = 1; W_22[3, 2] = 1
        U_22[3, 3] = 1
        V_22[2, 3] = 1; V_22[0, 3] = -1
        W_22[0, 3] = 1; W_22[2, 3] = 1
        U_22[0, 4] = 1; U_22[1, 4] = 1
        V_22[3, 4] = 1
        W_22[0, 4] = -1; W_22[1, 4] = 1
        U_22[2, 5] = 1; U_22[0, 5] = -1
        V_22[0, 5] = 1; V_22[1, 5] = 1
        W_22[3, 5] = 1
        U_22[1, 6] = 1; U_22[3, 6] = -1
        V_22[2, 6] = 1; V_22[3, 6] = 1
        W_22[0, 6] = 1; W_22[1, 6] = -1

        def add_block(u_map, v_map, w_map, start_col):
            for c in range(7):
                for l in range(4):
                    if U_22[l, c] != 0:
                        U_56[u_map[l], start_col + c] += U_22[l, c]
                    if V_22[l, c] != 0:
                        V_56[v_map[l], start_col + c] += V_22[l, c]
                    if W_22[l, c] != 0:
                        W_56[w_map[l], start_col + c] += W_22[l, c]

        # Mapping for 8 blocks
        # A is 2x4. Indices 0,1,2,3.
        # B is 4x8. Indices 0..7.
        # C is 8x2? No, C is 2x8. Indices k*2+i.
        
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A rows 0,1. A cols 0,1.
        # B rows 0,1. B cols 0,1.
        # C rows 0,1. C cols 0,1.
        add_block([0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 2, 3], 0)
        
        # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        add_block([0, 1, 4, 5], [2, 3, 10, 11], [4, 5, 6, 7], 7)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        add_block([0, 1, 4, 5], [4, 5, 12, 13], [8, 9, 10, 11], 14)
        
        # Block 3: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
        add_block([0, 1, 4, 5], [6, 7, 14, 15], [12, 13, 14, 15], 21)
        
        # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        add_block([2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 2, 3], 28)
        
        # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        add_block([2, 3, 6, 7], [18, 19, 26, 27], [4, 5, 6, 7], 35)
        
        # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
        add_block([2, 3, 6, 7], [20, 21, 28, 29], [8, 9, 10, 11], 42)
        
        # Block 7: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
        add_block([2, 3, 6, 7], [22, 23, 30, 31], [12, 13, 14, 15], 49)
        
        return U_56, V_56, W_56

    U_base, V_base, W_base = get_strassen_block_decomp()
    
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros_like(T)
        for c in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_recon, T, atol=1e-8)

    if verify(U_base, V_base, W_base):
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 56
    else:
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = 64

    # 4. NEW STRATEGY: Local Search for Lower Rank
    # We try to find a decomposition with rank < best_rank.
    # We use a randomized restart approach with gradient descent.
    # Objective: Minimize || T - sum_{c=1}^r u_c v_c w_c ||_F^2
    
    def try_find_lower_rank(current_best_rank):
        if time.time() > deadline - 10.0:
            return None
            
        target_ranks = list(range(current_best_rank - 1, max(current_best_rank - 5, 40), -1))
        
        for r in target_ranks:
            if time.time() > deadline - 5.0:
                break
                
            # Try a few random restarts
            for restart in range(5):
                if time.time() > deadline - 5.0:
                    break
                    
                np.random.seed(42 + r * 100 + restart)
                
                # Initialize with random vectors
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
                
                # Scale to match the magnitude of the base solution
                U *= 1.0
                V *= 1.0
                W *= 1.0
                
                # Perform a few steps of gradient descent
                # Gradient of E = ||T - sum u v w||^2
                # dE/du_c = -2 sum_{i,j,k} (T[i,j,k] - sum u v w)[i,j,k] V[j,c] W[k,c]
                # This is expensive. We can use a simpler heuristic:
                # Randomly perturb one vector and check if error decreases.
                
                # Let's use a simple coordinate descent / random walk
                best_err = np.inf
                current_err = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', U, V, W))
                
                for step in range(200):
                    if time.time() > deadline - 5.0:
                        return None
                        
                    # Pick a random vector to update
                    vec_idx = np.random.choice(3)
                    col_idx = np.random.randint(r)
                    
                    # Generate a small perturbation
                    if vec_idx == 0:
                        delta = np.random.randn(n * m) * 0.1
                        new_U = U.copy()
                        new_U[:, col_idx] += delta
                        err = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', new_U, V, W))
                        if err < current_err:
                            U = new_U
                            current_err = err
                    elif vec_idx == 1:
                        delta = np.random.randn(m * p) * 0.1
                        new_V = V.copy()
                        new_V[:, col_idx] += delta
                        err = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', U, new_V, W))
                        if err < current_err:
                            V = new_V
                            current_err = err
                    else:
                        delta = np.random.randn(p * n) * 0.1
                        new_W = W.copy()
                        new_W[:, col_idx] += delta
                        err = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', U, V, new_W))
                        if err < current_err:
                            W = new_W
                            current_err = err
                            
                    if current_err < 1e-8:
                        if verify(U, V, W):
                            return U, V, W
                            
                if current_err < best_err:
                    best_err = current_err
                    
        return None

    # Attempt to improve
    if best_rank > 48: # Only try if we have a reasonable baseline
        improved = try_find_lower_rank(best_rank)
        if improved is not None:
            U_new, V_new, W_new = improved
            if verify(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = U_new.shape[1]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

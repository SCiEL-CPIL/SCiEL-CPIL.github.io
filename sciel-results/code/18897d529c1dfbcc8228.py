"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 96
    
    def is_valid_fast(U, V, W, rank):
        """Fast vectorized check if U, V, W form a valid tensor decomposition.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        """
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        # For each component c, compute the contribution to T
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient: for each c, form outer product of reshaped vectors
        # Uc.reshape(n,m) * Vc.reshape(m,p) * Wc.reshape(p,n) element-wise
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Wc[k,i] means we need to transpose: Wc.T[i,k] = Wc[k,i]
            T += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[np.newaxis, :, :]
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as baseline
    school = schoolbook()
    if is_valid_fast(school["U"], school["V"], school["W"], best_rank):
        best_result = school
        # print(f"Schoolbook rank {best_rank} is valid")
    
    # Strategy: Try structured decompositions
    # 4x4 x 4x6 can be seen as 2x2 blocks of 2x2 times 2x3 blocks of 2x2
    # Each 2x2 x 2x2 -> 2x2 has Strassen rank 7
    # The block structure: 
    #   A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    #   B = [[B11, B12, B13], [B21, B22, B23]] where each Bj is 2x2
    #   C = A*B = 2x3 block matrix of 2x2 blocks
    
    # For the 2x2 block matrix multiplication (2x2 blocks times 2x3 blocks),
    # we need to compute 6 products of 2x2 matrices.
    # Using Strassen for each pair: 7 multiplications per pair, but we can share.
    
    # Actually, let's think of it as: we have a 2x2 "super-matrix" A and 2x3 "super-matrix" B
    # C[i_block, k_block] = sum_{j_block} A[i_block, j_block] * B[j_block, k_block]
    # This is like multiplying a 2x2 matrix by a 2x3 matrix at the block level.
    
    # The block-level multiplication 2x2 * 2x3 -> 2x3 has tensor rank...
    # It's similar to 2x2*2x2 but with an extra column.
    
    # Let me try a direct approach: use the known result that 
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,2,3} - what's its rank?
    
    # Alternative: just do a local search from a reasonable starting point
    # Start with a decomposition that's close to valid
    
    def generate_initial_candidate(rank):
        """Generate an initial candidate using a structured approach."""
        # Start with schoolbook and try to merge columns
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Initialize with random small values
        for c in range(rank):
            for idx in range(n * m):
                U[idx, c] = random.uniform(-1, 1) * 0.5
            for idx in range(m * p):
                V[idx, c] = random.uniform(-1, 1) * 0.5
            for idx in range(p * n):
                W[idx, c] = random.uniform(-1, 1) * 0.5
        return U, V, W
    
    def local_search(U, V, W, rank, max_iter=5000, lr=0.01):
        """Local search to minimize ||T - 1||^2 where T is the reconstructed tensor."""
        U_arr = U.copy()
        V_arr = V.copy()
        W_arr = W.copy()
        
        def compute_T(Ua, Va, Wa):
            T = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = Ua[:, c].reshape(n, m)
                Vc = Va[:, c].reshape(m, p)
                Wc = Wa[:, c].reshape(p, n)
                T += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[np.newaxis, :, :]
            return T
        
        best_loss = float('inf')
        U_best = U_arr.copy()
        V_best = V_arr.copy()
        W_best = W_arr.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if it % 100 == 0 and time.time() > deadline:
                break
            
            T = compute_T(U_arr, V_arr, W_arr)
            loss = np.sum((T - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                U_best = U_arr.copy()
                V_best = V_arr.copy()
                W_best = W_arr.copy()
                if best_loss < 1e-8:
                    break
            
            # Compute gradients numerically (simplified)
            # For efficiency, only do a few random perturbations per iteration
            improved = False
            for _ in range(3):
                # Pick a random matrix and random entry to perturb
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    mat = U_arr
                    size = n * m
                elif mat_choice == 1:
                    mat = V_arr
                    size = m * p
                else:
                    mat = W_arr
                    size = p * n
                
                idx = random.randint(0, size - 1)
                col = random.randint(0, rank - 1)
                delta = random.uniform(-lr, lr)
                
                old_val = mat[idx, col]
                mat[idx, col] += delta
                
                T_new = compute_T(U_arr, V_arr, W_arr)
                loss_new = np.sum((T_new - 1.0) ** 2)
                
                if loss_new < loss:
                    loss = loss_new
                    improved = True
                else:
                    mat[idx, col] = old_val
            
            if not improved and it > 100:
                # Try larger steps occasionally
                lr = max(0.001, lr * 0.5)
        
        # Convert back to lists with rounding to half-integers
        U_list = [[round(U_best[i, c] * 2) / 2 for c in range(rank)] for i in range(n * m)]
        V_list = [[round(V_best[i, c] * 2) / 2 for c in range(rank)] for i in range(m * p)]
        W_list = [[round(W_best[i, c] * 2) / 2 for c in range(rank)] for i in range(p * n)]
        
        return U_list, V_list, W_list, best_loss
    
    # Try various ranks with local search
    # Known lower bounds suggest rank should be significantly less than 96
    # For 4x4x4, the best known is 49 (Strassen-based). For 4x4x6, expect around 70-80
    
    ranks_to_try = [84, 80, 78, 76, 74, 72, 70, 68, 66, 64]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple initializations
        for init_trial in range(3):
            if time.time() > deadline:
                break
            
            U_init, V_init, W_init = generate_initial_candidate(target_rank)
            U_ls, V_ls, W_ls, final_loss = local_search(
                np.array(U_init), np.array(V_init), np.array(W_init), 
                target_rank, max_iter=2000, lr=0.05
            )
            
            if is_valid_fast(U_ls, V_ls, W_ls, target_rank):
                best_rank = target_rank
                best_result = {"U": U_ls, "V": V_ls, "W": W_ls}
                # print(f"Found valid rank {target_rank}!")
                break
        
        if time.time() > deadline:
            break
    
    # Also try a structured Strassen-based approach
    # Split into 2x2 blocks
    def strassen_block_decomposition():
        """Use Strassen at the 2x2 block level."""
        # A is 4x4 = 2x2 blocks of 2x2
        # B is 4x6 = 2x3 blocks of 2x2  
        # C is 4x6 = 2x3 blocks of 2x2
        
        # Block indices: A[i,j] for i,j in {0,1}, B[j,k] for j in {0,1}, k in {0,1,2}
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        # Use Strassen-like formulas for the 2x2 block matrix mult:
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,0])
        # M2 = (A[1,0]+A[1,1])B[0,0]
        # M3 = A[0,0](B[0,1]-B[1,1])
        # M4 = A[1,1](B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])B[1,1]
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        # C[0,0] = M1+M4-M5+M7
        # C[0,1] = M3+M5
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]  -- need extra terms for 3rd column
        # C[1,0] = M2+M4
        # C[1,1] = M1-M2+M3+M6
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]  -- need extra terms for 3rd column
        
        # For the 3rd column, we need additional products.
        # This gets complex. Let me just return a candidate and let local search refine.
        
        # For now, create a structured initial guess
        rank = 84  # 7 Strassen mults * 12 (for 2x2x2x2 schoolbook) + some extra
        # Actually each 2x2x2x2 mult needs rank 7 (Strassen), and we have ~12 such mults
        # But with sharing, maybe 7*6 + 6*4 = 42+24 = 66? Unclear.
        
        U = np.zeros((16, rank), dtype=np.float64)
        V = np.zeros((24, rank), dtype=np.float64)
        W = np.zeros((24, rank), dtype=np.float64)
        
        # Initialize with random values for local search
        for c in range(rank):
            U[:, c] = random.uniform(-0.5, 0.5)
            V[:, c] = random.uniform(-0.5, 0.5)
            W[:, c] = random.uniform(-0.5, 0.5)
        
        return U, V, W, rank
    
    # Try structured initialization with local search
    if time.time() < deadline:
        U_str, V_str, W_str, str_rank = strassen_block_decomposition()
        for target_rank in [str_rank, str_rank - 2, str_rank - 4]:
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            U_ls, V_ls, W_ls, final_loss = local_search(
                U_str.copy(), V_str.copy(), W_str.copy(),
                target_rank, max_iter=3000, lr=0.05
            )
            
            if is_valid_fast(U_ls, V_ls, W_ls, target_rank):
                best_rank = target_rank
                best_result = {"U": U_ls, "V": V_ls, "W": W_ls}
                break
    
    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END


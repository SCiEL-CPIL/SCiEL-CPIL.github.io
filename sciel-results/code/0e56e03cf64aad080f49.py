"""Matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 140  # 140s deadline (20s safety margin)
    
    def schoolbook():
        """Return schoolbook decomposition (rank n*m*p)."""
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
        return U, V, W, rank
    
    def validate_fast(U, V, W):
        """Fast validation using numpy."""
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def validate_exact(U, V, W):
        """Exact validation using Fraction arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        u_val = Fraction(U[i*m+j][c])
                        v_val = Fraction(V[j*p+k][c])
                        w_val = Fraction(W[k*n+i][c])
                        total += u_val * v_val * w_val
                    if total != Fraction(1):
                        return False
        return True
    
    # Winograd F(m, n) transform approach for matrix multiplication
    # For multiplying A (4x4) by B (4x9), split B into B1 (4x4) and B2 (4x5)
    # Use Winograd F(4,4) for the 4x4x4 part, schoolbook for 4x4x5
    
    # Winograd F(4,4) transform:
    # We want to compute C = A * B where A is 4x4, B is 4x4
    # Using a transform-based approach:
    # 1. Transform A rows: A_t = T_A * A (T_A is 4x4)
    # 2. Transform B cols: B_t = B * T_B (T_B is 4x4)  
    # 3. Element-wise multiply: P = A_t * B_t (4x4)
    # 4. Transform result: C = T_C * P
    # This gives rank 16 (same as schoolbook) unless transforms are sparse
    
    # Better: use a proper tensor decomposition search
    # Let's implement a random search for lower-rank decompositions
    
    # Strategy: Start with schoolbook, then try to find linear dependencies
    # among the rank-1 terms to reduce the total rank
    
    # For a more structured approach, let's try:
    # 1. Block decomposition: split 9 = 4 + 4 + 1
    # 2. For each 4x4 block, try to use a rank < 16 decomposition
    # 3. For the 4x1 block, use rank 4
    
    # Known: 2x2x2 matrix mult has rank 7 (Strassen)
    # 4x4x4 can be viewed as 4 blocks of 2x2x2, each rank 7 = rank 28
    # But this is worse than schoolbook 16!
    
    # Actually for 4x4x4, the best known is rank 14 (Bürgisser) or similar
    # Let's just do a direct random search
    
    # Direct search: represent the multiplication tensor as sum of rank-1 tensors
    # T[i,j,k] = 1 for all i,j,k in valid range
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Random search approach:
    # 1. Initialize with random rank-r decomposition
    # 2. Use gradient descent or random perturbation to minimize error
    
    def random_search(target_rank, num_iters=5000, seed=42):
        """Try to find a rank-target_rank decomposition via random search."""
        rng = random.Random(seed)
        
        # Initialize with random values
        U = [[float(rng.choice([-1, 1])) * rng.random() for _ in range(target_rank)] for _ in range(n*m)]
        V = [[float(rng.choice([-1, 1])) * rng.random() for _ in range(target_rank)] for _ in range(m*p)]
        W = [[float(rng.choice([-1, 1])) * rng.random() for _ in range(target_rank)] for _ in range(p*n)]
        
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # Compute current error
        def compute_error(U_mat, V_mat, W_mat):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                        err += (s - 1.0) ** 2
            return err
        
        best_err = compute_error(U_mat, V_mat, W_mat)
        best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        
        lr = 0.01
        for it in range(num_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute gradients (approximate)
            # For each (i,j,k), d/dU[i*m+j,c] = (s_ijk - 1) * V[j*p+k,c] * W[k*n+i,c]
            # This is expensive; use stochastic gradient descent
            
            # Pick random (i,j,k)
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            
            # Compute s_ijk
            s = 0.0
            for c in range(target_rank):
                s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
            
            err_ijk = s - 1.0
            if abs(err_ijk) < 1e-10:
                continue
            
            # Update U[i*m+j, :]
            for c in range(target_rank):
                grad_u = err_ijk * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                U_mat[i*m+j, c] -= lr * grad_u
                
                grad_v = err_ijk * U_mat[i*m+j, c] * W_mat[k*n+i, c]
                V_mat[j*p+k, c] -= lr * grad_v
                
                grad_w = err_ijk * U_mat[i*m+j, c] * V_mat[j*p+k, c]
                W_mat[k*n+i, c] -= lr * grad_w
            
            # Occasionally check error
            if it % 500 == 0:
                cur_err = compute_error(U_mat, V_mat, W_mat)
                if cur_err < best_err:
                    best_err = cur_err
                    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        
        # Check if we got close enough
        final_err = compute_error(best_U, best_V, best_W)
        if final_err < 1e-6:
            # Convert to list of lists with rounded values
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return U_list, V_list, W_list, target_rank
        return None
    
    # Try different target ranks
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find better decompositions
    for target_rank in range(143, 30, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 25:
                break
            
            result = random_search(target_rank, num_iters=3000, seed=seed)
            if result is not None:
                U, V, W, rank = result
                if validate_fast(U, V, W):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    break
        else:
            continue
        break
    
    # Final validation
    if not validate_fast(best_U, best_V, best_W):
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

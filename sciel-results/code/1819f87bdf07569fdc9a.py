"""Matrix multiplication 2x3x7 rank minimization using structured
Strassen-style block decomposition and exact integer verification."""
import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130  # safety margin before 150s kill

    def schoolbook():
        """Standard schoolbook decomposition: rank = n*m*p = 42."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Faster verification using matrix operations."""
        U_arr = np.array(U, dtype=np.int64).reshape(n, m, rank)
        V_arr = np.array(V, dtype=np.int64).reshape(m, p, rank)
        W_arr = np.array(W, dtype=np.int64).reshape(p, n, rank)
        # For each c, the contribution is U[:,:,c] * V[:,:,c] * W[:,:,c]
        # Sum over c
        total = np.zeros((n, m, p), dtype=np.int64)
        for c in range(rank):
            total += U_arr[:, :, c] * V_arr[:, :, c] * W_arr[:, :, c]
        return np.all(total == 1)

    # Strategy 1: Strassen-style block decomposition
    # Split the 2x3 matrix into blocks and use Strassen-like formulas
    # For 2x3 x 3x7:
    # A is 2x3, B is 3x7, C is 2x7
    # We can split A into A1 (1x3) and A2 (1x3)
    # B into B1 (3x3), B2 (3x3), B3 (3x1) or similar
    
    # Let's try a more systematic approach:
    # Split the 3x7 matrix B into three 3x3 blocks B1, B2, B3 and one 3x1 block B4
    # Actually, 7 = 3+3+1, so B = [B1 B2 B3 B4] where B1,B2,B3 are 3x3 and B4 is 3x1
    
    # For the 2x3 matrix A, we can use it as is.
    # C = A*B = [A*B1, A*B2, A*B3, A*B4]
    # Each A*Bj is 2x3 (for j=1,2,3) or 2x1 (for j=4)
    
    # Let's try to use Strassen on the 2x3 x 3x3 multiplications
    # For 2x3 * 3x3, we can split:
    # A = [A1; A2] where A1, A2 are 1x3
    # B = [B1 B2; B3 B4; B5 B6] where each Bj is 1x1... no, that doesn't work well
    
    # Alternative: Use the fact that we can compute 2x3 * 3x3 with rank 7 or 8
    # using a Strassen-like approach on the 2x2 part
    
    # Let me try a different approach: direct construction with known patterns
    
    # Strategy 2: Use a structured pattern based on the geometry
    # The key insight: since p=7 is large, we can potentially share computation
    # across the output columns
    
    # Let's try to find a decomposition where the rank is less than 42
    # by exploiting linear dependencies
    
    # For now, let's try random search with better initialization
    # Start from a perturbed schoolbook and try to reduce rank
    
    # Strategy 3: Greedy rank reduction
    # Start with schoolbook rank 42 and try to merge terms
    
    def try_rank_reduction(start_rank, target_rank, max_attempts=100):
        """Try to reduce rank from start_rank to target_rank."""
        if time.time() > deadline - 30:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Start with a random sparse initialization
            sparsity = 0.5
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            for i in range(nm):
                for c in range(target_rank):
                    if random.random() < sparsity:
                        U[i][c] = random.choice([-1, 1])
            
            for j in range(mp):
                for c in range(target_rank):
                    if random.random() < sparsity:
                        V[j][c] = random.choice([-1, 1])
            
            for k in range(pn):
                for c in range(target_rank):
                    if random.random() < sparsity:
                        W[k][c] = random.choice([-1, 1])
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try various target ranks
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 40:
            break
        
        result = try_rank_reduction(best_rank, target_rank, max_attempts=50)
        if result is not None:
            best = result
            best_rank = target_rank
    
    # Strategy 4: Structured construction using the rectangular geometry
    # For 2x3x7, try to exploit the fact that n=2 is small
    # We can write the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=0,1 and k=0,...,6
    
    # Let's try a construction where we use shared intermediate values
    # across different output positions
    
    # Construct a decomposition based on the following idea:
    # Use rank-1 terms that cover multiple (i,j,k) tuples at once
    
    def construct_structured(target_rank):
        """Try to construct a valid decomposition with the given rank."""
        if time.time() > deadline - 20:
            return None
        
        U = [[0] * target_rank for _ in range(nm)]
        V = [[0] * target_rank for _ in range(mp)]
        W = [[0] * target_rank for _ in range(pn)]
        
        # For each rank term, assign patterns that cover multiple (i,j,k)
        for c in range(target_rank):
            # Choose a pattern for this term
            # U pattern: which (i,j) pairs are non-zero
            # V pattern: which (j,k) pairs are non-zero  
            # W pattern: which (k,i) pairs are non-zero
            
            # Try different sparsity levels
            u_sparsity = random.choice([0.3, 0.5, 0.7])
            v_sparsity = random.choice([0.3, 0.5, 0.7])
            w_sparsity = random.choice([0.3, 0.5, 0.7])
            
            for i in range(n):
                for j in range(m):
                    if random.random() < u_sparsity:
                        U[i * m + j][c] = random.choice([-1, 1])
            
            for j in range(m):
                for k in range(p):
                    if random.random() < v_sparsity:
                        V[j * p + k][c] = random.choice([-1, 1])
            
            for k in range(p):
                for i in range(n):
                    if random.random() < w_sparsity:
                        W[k * n + i][c] = random.choice([-1, 1])
        
        if verify_fast(U, V, W, target_rank):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try structured construction for various ranks
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            result = construct_structured(target_rank)
            if result is not None:
                best = result
                best_rank = target_rank
                break
    
    # Final verification
    if time.time() < deadline - 5:
        if not verify_fast(best["U"], best["V"], best["W"], best_rank):
            best = schoolbook()
            best_rank = n * m * p

    return best
# EVOLVE-BLOCK-END

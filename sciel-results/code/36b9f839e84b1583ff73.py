"""Program to find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 30
    W_dim = p * n  # 12

    deadline = time.time() + 110  # 110s out of 120s, leave safety margin
    best_rank = n * m * p  # 60
    best_solution = None

    # Initialize best solution with schoolbook
    U_sb = [[0.0] * best_rank for _ in range(U_dim)]
    V_sb = [[0.0] * best_rank for _ in range(V_dim)]
    W_sb = [[0.0] * best_rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    best_solution = (U_sb, V_sb, W_sb)

    def verify(U, V, W, rank):
        """Verify the decomposition is exact. Returns True if valid."""
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check that T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # equals 1 if k==k (always) and 0 otherwise... wait, no.
        # The matrix multiplication tensor is: T_{i,j,k} = 1 if the output position (i,k) 
        # receives contribution from A_{i,j} * B_{j,k}. So T_{i,j,k} = 1 for all i,j,k.
        # Wait, that's not right either. Let me reconsider.
        
        # For matrix multiplication C = A * B where A is n x m and B is m x p:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor decomposition satisfies:
        # sum_c U[i,j,c] A_{i,j} * V[j,k,c] B_{j,k} = C_{i,k}
        # This requires: sum_c U[i,j,c] V[j,k,c] = delta_{i,k}? No...
        # Actually, we need: sum_c U[i,j,c] * V[j,k,c] = W[k,i,c] for all c? No.
        # The condition is: for all c, U[i,j,c] * V[j,k,c] * W[k,i,c] should combine
        # to give the right tensor. The tensor T_{i,j,k} = 1 (since each A_{i,j}B_{j,k} 
        # contributes to C_{i,k} with coefficient 1). So we need:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
        
        # But wait, W has dimension p*n with index k*n+i. Let me re-check the convention.
        # W[k*n+i, c] is the entry for output position (i, k) at decomposition step c.
        # The standard convention for matrix multiplication tensor decomposition is:
        # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And we want T_{i,j,k} = 1 for all valid i, j, k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank, max_iter=500):
        """Try to find a valid decomposition of given rank using iterative refinement."""
        import numpy as np
        np.random.seed(rank)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize randomly
            U = np.random.uniform(-1, 1, (U_dim, rank))
            V = np.random.uniform(-1, 1, (V_dim, rank))
            W = np.random.uniform(-1, 1, (W_dim, rank))
            
            # Scale to avoid overflow
            U /= np.sqrt(np.mean(U**2)) if np.mean(U**2) > 0 else 1
            V /= np.sqrt(np.mean(V**2)) if np.mean(V**2) > 0 else 1
            W /= np.sqrt(np.mean(W**2)) if np.mean(W**2) > 0 else 1
            
            # For the 2x5x6 case, let's use a structured approach
            # The key insight: if we split B into blocks, we can use block decomposition.
            # Specifically, we can decompose the multiplication into:
            # - For each output (i,k), we need sum_j A_{i,j} B_{j,k}
            # 
            # One approach: use the fact that rank(2x5x6) <= rank(2x5x1) * 6 + ... 
            # Actually, let's try a direct low-rank search.
            
            # Use alternating least squares
            for iteration in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U: for each i,j,c, minimize sum_{k} (target - ...) 
                # Actually, use gradient-free approach with random perturbations
                
                # Try a different initialization strategy: exploit structure
                break  # Skip to next attempt with different init
            
            # Alternative: use known low-rank constructions
            # For 2 x m x p, we can use rank <= 2 * max(m, p) in some cases?
            # Actually, the Strassen-style algorithms don't directly apply here.
            
            # Let's use a greedy approach: start with rank 1 blocks
            # The minimum rank for 2x5x6... let's estimate.
            # Lower bound: max(n*m, m*p, p*n, n*p) = max(10, 30, 12, 12) = 30
            # Upper bound: n*m*p = 60
            # So rank is between 30 and 60.
            
            # Try to construct using the "outer product" structure
            # If we could make U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
            # with fewer terms...
            
            # A simple construction: for each (i,k) pair, create terms that cover all j
            # Number of (i,k) pairs: n*p = 12
            # For each (i,k), we need sum_j A_{i,j} B_{j,k} = C_{i,k}
            # We can do this with rank m for each (i,k) independently: rank = n*p*m = 12*5 = 60
            # That's the schoolbook.
            
            # Better: share terms across (i,k) pairs.
            # Idea: For each j, the contribution is A_{*,j} * B_{j,*}
            # A_{*,j} is a column of A (size n=2), B_{j,*} is a row of B (size p=6)
            # The outer product A_{*,j} B_{j,*} is a 2x6 matrix.
            # We need to represent the sum over j of these outer products.
            # Each outer product is rank-1 as a tensor? No, it's a matrix.
            # But as part of the 3-index tensor, each term A_{i,j} B_{j,k} contributes
            # to T[i,j,k] with coefficient... hmm.
            
            # Let me just try random search with verification
            U = np.random.randn(U_dim, rank) * 0.5
            V = np.random.randn(V_dim, rank) * 0.5
            W = np.random.randn(W_dim, rank) * 0.5
            
            # Normalize
            for c in range(rank):
                norm_u = np.sqrt(np.sum(U[:, c]**2))
                norm_v = np.sqrt(np.sum(V[:, c]**2))
                norm_w = np.sqrt(np.sum(W[:, c]**2))
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    # Scale so that typical product is O(1)
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    if scale > 1e-12:
                        U[:, c] /= scale ** (1/3) if scale ** (1/3) > 0 else 1
                        V[:, c] /= scale ** (1/3) if scale ** (1/3) > 0 else 1
                        W[:, c] /= scale ** (1/3) if scale ** (1/3) > 0 else 1
            
            # Simple verification
            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    # Try ranks from lower bound up
    # Lower bound for tensor rank of matrix mult: known to be at least max(n,m,p)^2 roughly
    # For 2x5x6, the known lower bound is not trivial, but >= min(n*m, m*p, p*n) = 10
    # More carefully: the border rank is at least the geometric rank etc.
    # Let's try from rank 30 (which is m*p, the largest "face" dimension)
    # Actually, let's be pragmatic: try a few promising ranks
    
    # The current best is 60. Let's see if we can do better.
    # Try ranks: 48, 45, 42, 40, 38, 36, 35, 34, 33, 32, 31, 30
    for rank in range(50, 30, -1):
        if time.time() > deadline:
            break
        
        print(f"Trying rank {rank}...", file=sys.stderr)
        
        # Use a structured initialization based on known techniques
        # For matrix multiplication, one known construction uses the
        # "block diagonal" or "tensor product" of smaller mults.
        
        # Actually, let's use a randomized search with more iterations
        np.random.seed(rank * 7 + 13)
        
        success = False
        for trial in range(200):
            if time.time() > deadline:
                break
            
            U = np.zeros((U_dim, rank))
            V = np.zeros((V_dim, rank))
            W = np.zeros((W_dim, rank))
            
            # Try a structured pattern: assign random signs
            for c in range(rank):
                U[:, c] = np.random.choice([-1, 1], U_dim)
                V[:, c] = np.random.choice([-1, 1], V_dim)
                W[:, c] = np.random.choice([-1, 1], W_dim)
            
            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                success = True
                best_rank = rank
                best_solution = (U.tolist(), V.tolist(), W.tolist())
                break
            
            # Also try with some continuous values
            U = np.random.randn(U_dim, rank)
            V = np.random.randn(V_dim, rank)
            W = np.random.randn(W_dim, rank)
            
            # Project to make it work... this is hard in general.
            
            # Try a heuristic: for each c, pick a (i0, j0, k0) and set
            # U[i0*m+j0, c] = 1, V[j0*p+k0, c] = 1, W[k0*n+i0, c] = 1
            # This gives rank n*m*p. Not helpful for reduction.
        
        if success:
            break
    
    # Use the best solution found
    U_final, V_final, W_final = best_solution
    
    # Convert to simple format (ensure integer/half-integer if possible)
    result_U = [[0] * best_rank for _ in range(U_dim)]
    result_V = [[0] * best_rank for _ in range(V_dim)]
    result_W = [[0] * best_rank for _ in range(W_dim)]
    
    for i in range(U_dim):
        for c in range(best_rank):
            val = U_final[i][c]
            # Round to nearest integer if close
            if abs(val - round(val)) < 1e-6:
                result_U[i][c] = int(round(val))
            elif abs(val - round(val*2)/2) < 1e-6:
                result_U[i][c] = round(val*2)/2
            else:
                result_U[i][c] = val
    for j in range(V_dim):
        for c in range(best_rank):
            val = V_final[j][c]
            if abs(val - round(val)) < 1e-6:
                result_V[j][c] = int(round(val))
            elif abs(val - round(val*2)/2) < 1e-6:
                result_V[j][c] = round(val*2)/2
            else:
                result_V[j][c] = val
    for k in range(W_dim):
        for c in range(best_rank):
            val = W_final[k][c]
            if abs(val - round(val)) < 1e-6:
                result_W[k][c] = int(round(val))
            elif abs(val - round(val*2)/2) < 1e-6:
                result_W[k][c] = round(val*2)/2
            else:
                result_W[k][c] = val
    
    # Final verification
    final_valid = verify(result_U, result_V, result_W, best_rank)
    if not final_valid:
        # Fall back to schoolbook
        result_U, result_V, result_W = U_sb, V_sb, W_sb
        best_rank = 60
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END

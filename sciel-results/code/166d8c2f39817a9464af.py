import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Dimensions of U, V, W
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Target Rank: 33
    # If we find rank 33, score = 36/33 = 1.0909
    # If we find rank 32, score = 36/32 = 1.125
    # We will try rank 33 first. If it fails, we return rank 34 (base).
    
    def get_rank34_solution():
        # Reusing the known rank-34 solution as fallback
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U_base, V_base, W_base = get_rank34_solution()
    
    def verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        # Check a subset of indices for speed, then full if needed? 
        # For final verification, we must be exact.
        # But for search, we can use approximate checks.
        # This function is for FINAL validation.
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                                if not np.isclose(s, target, atol=1e-9):
                                    return False
        return True

    def compute_residual(U, V, W, n, m, p):
        """
        Computes the Frobenius norm of the difference between the decomposed tensor and the target tensor.
        Target T[a,b,c,d,i,k] = 1 if a==i, c==k, b==d else 0.
        Decomposed D[a,b,c,d,i,k] = sum_l U[a*m+b, l] V[c*p+d, l] W[k*n+i, l]
        """
        # We can compute the residual by iterating over the "active" parts of the target tensor.
        # The target tensor is sparse: 1 only when a==i, c==k, b==d.
        # Number of 1s: 
        # a in 0..1, i=a (2 choices)
        # b in 0..2, d=b (3 choices)
        # c in 0..2, k=c (3 choices)
        # Total 18 non-zero entries in T.
        # Wait, T is a 6D tensor.
        # Indices: a,b,c,d,i,k
        # Constraint: a==i, c==k, b==d.
        # So for each (a,b,c), there is exactly one (i,k,d) such that T=1.
        # Specifically, i=a, k=c, d=b.
        # So T has 2*3*3 = 18 ones.
        
        # The decomposed tensor D has entries for ALL indices.
        # We need || D - T ||_F^2 = sum_{all indices} (D[idx] - T[idx])^2
        # = sum_{idx where T=1} (D[idx] - 1)^2 + sum_{idx where T=0} D[idx]^2
        
        # This is computationally expensive if done naively over all 1296 indices.
        # However, we can exploit linearity.
        # Let's just compute the error on the 18 "active" indices and a random sample of "inactive" indices?
        # No, for optimization, we need a good gradient-like signal.
        # Let's compute the error on all 18 active indices and 100 random inactive indices.
        
        error = 0.0
        
        # Active indices: T=1
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    i = a
                    k = c
                    d = b
                    # D[idx]
                    val = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                    error += (val - 1.0) ** 2
                    
        # Inactive indices: T=0
        # Sample 100 random indices where T=0
        # To ensure they are inactive, we can just pick random indices and check if they are active.
        # Or just pick random indices and subtract the target (which is 0).
        # If we pick an active index by chance, we subtract 1, which is wrong for the "inactive" sum.
        # Better: pick random indices, compute D, and if it's an active index, we already counted it in the first loop.
        # To avoid double counting or confusion, let's just sample random indices and check if they are active.
        
        for _ in range(100):
            a = random.randint(0, n - 1)
            b = random.randint(0, m - 1)
            c = random.randint(0, m - 1)
            d = random.randint(0, p - 1)
            i = random.randint(0, n - 1)
            k = random.randint(0, p - 1)
            
            is_active = (a == i and c == k and b == d)
            if is_active:
                continue # Skip to avoid double counting or bias
                
            val = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
            error += (val - 0.0) ** 2
            
        return error

    def try_find_rank33():
        if time.time() > deadline:
            return None
            
        rank = 33
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        
        # Optimization Loop: Coordinate Descent with Snap
        # We will perform 5000 iterations.
        # In each iteration, we pick a random matrix (U, V, or W) and a random column,
        # and try to update it to reduce the residual.
        # Since it's non-convex, we use a "trial move" approach:
        # 1. Save current state.
        # 2. Perturb one entry.
        # 3. If residual decreases, keep. Else revert.
        # This is slow. 
        # Alternative: Use a simple gradient approximation or just random restarts with snapping.
        
        # Given the time limit, let's do 20 random restarts.
        # Each restart:
        # 1. Init random.
        # 2. Run 500 steps of "Local Search" (try small perturbations).
        # 3. Snap to half-integers.
        # 4. Check exact validity.
        
        best_sol = None
        
        for restart in range(30):
            if time.time() > deadline:
                break
                
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Scale down to avoid overflow
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            current_err = compute_residual(U, V, W, n, m, p)
            
            # Local Search: 100 steps
            for step in range(100):
                if time.time() > deadline:
                    return best_sol
                    
                # Pick a random matrix and entry
                mat_type = random.randint(0, 2)
                if mat_type == 0:
                    mat = U
                    dim = dim_u
                elif mat_type == 1:
                    mat = V
                    dim = dim_v
                else:
                    mat = W
                    dim = dim_w
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, rank - 1)
                
                # Try a small perturbation
                old_val = mat[row, col]
                # Adaptive step size: start with 0.1, shrink if no progress?
                # For simplicity, fixed small step.
                delta = random.uniform(-0.2, 0.2)
                mat[row, col] = old_val + delta
                
                new_err = compute_residual(U, V, W, n, m, p)
                
                if new_err < current_err:
                    current_err = new_err
                    # If error is very small, snap and check
                    if current_err < 1e-4:
                        # Snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                            return U_snap, V_snap, W_snap
                else:
                    mat[row, col] = old_val
                    
            # End of local search for this restart
            # Final snap check
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
                
        return None

    sol = try_find_rank33()
    
    if sol is not None:
        U, V, W = sol
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to rank 34
    return {
        "U": U_base.tolist(),
        "V": V_base.tolist(),
        "W": W_base.tolist()
    }

if __name__ == "__main__":
    construct()

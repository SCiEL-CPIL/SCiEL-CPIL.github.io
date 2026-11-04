"""Matrix multiplication 4x5x6 rank-minimal decomposition search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a lower rank decomposition using random initialization and 
    # iterative refinement with alternating least squares
    # The tensor can be written as sum_r U_r V_r^T W_r^T (in mode-3 matricization terms)
    
    # Actually, let's think about this differently.
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,c] * V[j,c] * W[k,c]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # We can use ALS (Alternating Least Squares) to find such a decomposition.
    # For each rank r, we initialize randomly and refine.
    
    def try_als(rank, max_iters=200, tol=1e-10, seed=42):
        """Try ALS decomposition for given rank. Returns (U, V, W) or None if failed."""
        rng = random.Random(seed)
        
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each c, U[:,c] = argmin ||T - sum U V W||
            # Fix V, W. Residual R = T - sum_{c'} U[:,c'] * (V*W)[:,:,c']
            # For updating U[:,c], we need: U[:,c] = (V[c,:] * W[c,:])^T T / ((V[c,:] * W[c,:])^T (V[c,:] * W[c,:]))
            
            # More efficiently: compute the matricization
            # T_{i,(j,k)} = sum_c U[i,c] * V[j,c] * W[k,c]
            # Group (j,k) into index q = j*p+k... wait, V has index j*p+k and W has index k*n+i
            # This is tricky because V and W have different index structures.
            
            # Let me re-think. The tensor T has indices (i*m+j, j*p+k, k*n+i)
            # So the three modes are:
            # Mode 1: i*m+j (n*m values)
            # Mode 2: j*p+k (m*p values)  
            # Mode 3: k*n+i (p*n values)
            
            # For ALS, fix V and W, update U:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b,c: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is a linear system: for each (b,c) pair, we have an equation in U[a,:]
            # But we need it for all a simultaneously.
            
            # Better: For each column r, define M_r = V[:,r] * W[:,r] (element-wise product, shape (m*p, p*n))
            # Then T[a,b,c] = sum_r U[a,r] * M_r[b,c]
            # Residual: R = T - sum_r U[a,r] * M_r[b,c]
            # To update U[:,c0]: 
            #   R_without_c0 = T - sum_{r != c0} U[:,r] * M_r
            #   U[:,c0] = argmin ||R_without_c0 - U[:,c0] * M_{c0}||_F^2
            #   This gives: U[:,c0] = (M_{c0}^{(2)T} R_without_c0^{(1)}) / (M_{c0}^{(2)T} M_{c0}^{(2)})
            #   where M^{(2)} is the matricization of M_r by flattening (b,c) into one index.
            
            # Let me precompute M_r for all r
            M_list = [V[:,r] * W[:,r] for r in range(rank)]
            # M_list[r] has shape (m*p, p*n)
            
            # Flatten M_r to vector of length (m*p)*(p*n)
            M_flat = [M.reshape(-1) for M in M_list]  # each shape ((m*p)*(p*n),)
            
            # T flattened: shape (n*m, (m*p)*(p*n))
            T_flat = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
            
            # For each c0, compute residual without c0
            # This is O(rank^2 * size) which might be too slow
            # Let's use a more efficient approach
            
            # Alternative: use the standard CP ALS formulation
            # For mode-1 matricization: T_1 = U * (V ⊙ W)^T
            # where V ⊙ W is the Khatri-Rao product (column-wise element-wise product)
            # V ⊙ W has shape (m*p*p*n, rank) with each column being V[:,r] * W[:,r]
            
            # T_1 (shape n*m x m*p*p*n) = U (n*m x rank) * (V⊙W)^T (rank x m*p*p*n)
            
            # To update U: U = T_1 * (V⊙W) * ((V⊙W)^T (V⊙W))^{-1}
            # To update V: need mode-2 matricization
            # To update W: need mode-3 matricization
            
            # Let me compute the Khatri-Rao products
            # V⊙W: shape (m*p*p*n, rank)
            # This is expensive but let's try for small ranks
            
            # Actually m*p*p*n = 5*6*6*4 = 720, and rank could be up to ~50
            # So the matrix is 720 x rank, which is manageable
            
            # Compute mode-1 matricization T_1: (n*m, m*p*p*n)
            # T_1[a, b*c] where a = i*m+j, b = j*p+k, c = k*n+i
            # But b and c are not independent! b = j*p+k, c = k*n+i
            # So not all (b,c) pairs are valid.
            
            # Hmm, this is the issue. The tensor is not a full CP tensor over independent modes.
            # The modes are correlated through the index constraints.
            
            # Let me reconsider. The tensor T is defined on the set of valid triples (a,b,c)
            # where a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
            # There are only n*m*p = 120 such triples, not (n*m)*(m*p)*(p*n) = 20*30*24 = 14400.
            
            # So T is a sparse tensor with only 120 non-zero entries.
            
            # This changes things. We can work directly with these 120 equations.
            # T[a,b,c] = 1 for the 120 valid (a,b,c) triples, and 0 otherwise (but we don't care about zeros).
            
            # So we have 120 equations:
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a system of 120 nonlinear equations in rank*20 + rank*30 + rank*24 variables.
            # For rank < 120, this is overdetermined and may not have an exact solution.
            
            # Let's use a different approach: iterate over the 120 terms and use 
            # a gradient-based or combinatorial search.
            
            # For now, let's just try random rank-r solutions and check if they work.
            # For small r, we can try to solve using least squares.
            
            # Actually, let me just return the schoolbook solution since finding a 
            # better one within time is very challenging.
            return None
        
        return None
    
    # Try a few ranks with random restarts
    for rank in [40, 50, 60, 70, 80, 90, 100, 110]:
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with ALS-style refinement
        # For each rank, try to find U, V, W such that the 120 equations are satisfied
        
        # Use a simple iterative approach: randomly initialize, then try to minimize
        # the error sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        def minimize_error(rank, iters=100, lr=0.01, seed=0):
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, rank) * 0.1
            V = rng.randn(m*p, rank) * 0.1
            W = rng.randn(p*n, rank) * 0.1
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute error and gradients
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = 0.0
                            for r in range(rank):
                                val += U[a,r] * V[b,r] * W[c,r]
                            diff = val - 1.0
                            err += diff * diff
                            for r in range(rank):
                                grad_U[a,r] += 2 * diff * V[b,r] * W[c,r]
                                grad_V[b,r] += 2 * diff * U[a,r] * W[c,r]
                                grad_W[c,r] += 2 * diff * U[a,r] * V[b,r]
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    break
                
                # Gradient step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.95
            
            if best_err < 1e-10:
                # Convert to Fraction for exactness
                U_frac = [[Fraction(int(round(U_frac_val * 2**20))) / 2**20 for U_frac_val in row] for row in best_U]
                V_frac = [[Fraction(int(round(V_frac_val * 2**20))) / 2**20 for V_frac_val in row] for row in best_V]
                W_frac = [[Fraction(int(round(W_frac_val * 2**20))) / 2**20 for W_frac_val in row] for row in best_W]
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            s = Fraction(0)
                            for r in range(rank):
                                s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                            if s != 1:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    continue
                
                return {"U": U_frac, "V": V_frac, "W": W_frac, "rank": rank}
            
            return None
        
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = minimize_error(rank, iters=50, lr=0.05, seed=seed)
            if result is not None and result["rank"] < best_result["rank"]:
                best_result = result
                break
    
    if time.time() > deadline - 2:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    # Return the best result found
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END

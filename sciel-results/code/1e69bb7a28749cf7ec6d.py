import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin within 120s limit
    
    # Helper: compute cost of permutation
    def compute_cost(pos):
        return int(np.sum(a * b[np.ix_(pos, pos)]))
    
    # Helper: compute delta for swapping positions i and j
    def swap_delta(pos, i, j):
        x = pos[i]
        y = pos[j]
        diff1 = b[y, pos] - b[x, pos]
        diff2 = b[x, pos] - b[y, pos]
        diff3 = b[pos, y] - b[pos, x]
        diff4 = b[pos, x] - b[pos, y]
        return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)
    
    # Helper: Compute the gradient of the QAP objective at a doubly stochastic matrix X
    # f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # Gradient G[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    # For symmetric a and b: G = 2 * (A @ X @ B)
    def compute_gradient(X):
        return 2.0 * (a @ X @ b)
    
    # Frank-Wolfe on Birkhoff polytope
    def frank_wolfe_init():
        """
        Use Frank-Wolfe method on the Birkhoff polytope to find a good
        doubly stochastic matrix, then extract a permutation.
        """
        rng = np.random.default_rng(123)
        
        # Start with a random permutation as initial point
        init_perm = rng.permutation(n)
        X = np.zeros((n, n))
        X[init_perm, np.arange(n)] = 1.0
        
        # Frank-Wolfe iterations
        best_X = X.copy()
        best_cost = compute_cost(init_perm)
        
        for it in range(100):
            if time.time() > deadline - 60.0:
                break
            
            # Compute gradient
            G = compute_gradient(X)
            
            # Find linear minimizer: min_{Y in Birkhoff} <G, Y>
            # This is the linear assignment problem: min sum G[i,k] Y[i,k]
            row_idx, col_idx = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_idx, col_idx] = 1.0
            
            # Line search for step size gamma in [0, 1]
            best_gamma = 0.0
            best_perm_cost = best_cost
            best_X_new = X.copy()
            
            # Try a few step sizes
            gammas = [1.0, 0.5, 0.25, 2.0 / (it + 2.0)]
            
            for gamma in gammas:
                X_new = (1 - gamma) * X + gamma * Y
                
                # Extract permutation from X_new
                r_idx, c_idx = linear_sum_assignment(-X_new)
                perm = c_idx
                
                cost = compute_cost(perm)
                
                if cost < best_perm_cost:
                    best_perm_cost = cost
                    best_gamma = gamma
                    best_X_new = X_new
            
            # Update X
            if best_gamma > 0:
                X = best_X_new
                if best_perm_cost < best_cost:
                    best_cost = best_perm_cost
                    best_X = X.copy()
            else:
                break

        # Final extraction
        row_idx, col_idx = linear_sum_assignment(-X)
        return col_idx, best_cost
    
    # Local search with 2-opt and Hungarian moves
    def local_search(pos, max_iter=5000):
        """
        Hybrid local search:
        1. Full sweep 2-opt (first improvement)
        2. Periodic Hungarian move (global reassignment based on gradient)
        """
        pos = pos.copy()
        current_cost = compute_cost(pos)
        rng = np.random.default_rng(456)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Phase 1: 2-opt sweep
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                x = pos[i]
                
                for j in range(i + 1, n):
                    delta = swap_delta(pos, i, j)
                    if delta < 0:
                        pos[i], pos[j] = pos[j], pos[i]
                        current_cost += delta
                        improved = True
            
            if not improved:
                # If 2-opt is stuck, try a Hungarian move
                X = np.zeros((n, n))
                X[np.arange(n), pos] = 1.0
                
                G = compute_gradient(X)
                
                # Solve LASSO for G
                r_idx, c_idx = linear_sum_assignment(G)
                new_pos = c_idx.copy()
                
                new_cost = compute_cost(new_pos)
                
                if new_cost < current_cost:
                    pos = new_pos
                    current_cost = new_cost
                    improved = True
                else:
                    # No improvement, break
                    break
        
        return pos, current_cost
    
    # Phase 1: Frank-Wolfe initialization
    pos_fw, cost_fw = frank_wolfe_init()
    
    # Phase 2: Local search from FW solution
    pos_ls, cost_ls = local_search(pos_fw, max_iter=20)
    
    best_pos = pos_ls
    best_cost = cost_ls
    
    # Phase 3: Iterated local search with perturbations
    rng = np.random.default_rng(789)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb best solution
        pos = best_pos.copy()
        # Stronger perturbation: 3-opt or more swaps
        n_swaps = n // 4
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            pos[i], pos[j] = pos[j], pos[i]
        
        # Local search
        pos, cost = local_search(pos, max_iter=20)
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    # Final polish
    best_pos, best_cost = local_search(best_pos, max_iter=10)
    
    return best_pos.tolist()

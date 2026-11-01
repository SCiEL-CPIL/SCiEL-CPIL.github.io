"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Candidate values for entries - restricted to small integers for feasibility
    # Using a small set of values to make grid search possible
    candidate_vals = np.array([-2, -1, 0, 1, 2], dtype=float)
    
    found_lower = False
    
    # Try ranks slightly lower than schoolbook
    # Schoolbook rank is 42. Try 41, 40, 39...
    # We use a systematic search with pruning.
    # Since the space is huge, we restrict the search to "sparse" structures
    # or use a very tight time limit per rank.
    
    # For this specific task, the rank is likely 42 (schoolbook) or very close.
    # A full grid search over 42 columns is impossible.
    # However, we can try to find a rank 40 solution by assuming most columns are "standard"
    # and only a few are modified.
    
    # Alternative: Just return schoolbook if we don't have a better heuristic.
    # The prompt asks for a DIFFERENT strategy.
    # Let's try a "Perturbation" strategy: 
    # Take schoolbook, and try to merge two terms into one by adjusting coefficients.
    # This is equivalent to searching for a rank 41 solution.
    
    # Strategy: 
    # 1. Start with schoolbook (rank 42).
    # 2. Try to find a rank 41 decomposition by searching for a linear dependency 
    #    or a specific pattern of coefficients.
    #    Given the difficulty, we will implement a bounded random search 
    #    but with a much better initialization and a "greedy" cleanup.
    
    # Actually, let's stick to the "Systematic Grid Search with Pruning" for a SMALL subset of columns.
    # We assume that if a lower rank exists, it can be found by modifying a small number of columns 
    # from the schoolbook decomposition.
    
    # Let's try rank 41.
    # We will search for U, V, W of shape (6, 41), (21, 41), (14, 41).
    # This is still too large for brute force.
    
    # Revised Strategy: 
    # Use the fact that the tensor is sparse. 
    # We will perform a random search but with a "repair" step that tries to fix errors 
    # by adjusting single entries, using a more efficient error calculation.
    
    # To be materially different from the current "random local search", 
    # let's use a "Simulated Annealing" approach with a specific cooling schedule 
    # and a different move set (swapping columns, permuting rows).
    
    # However, given the 120s limit and the difficulty of finding an exact integer solution 
    # for rank < 42 in this specific matrix multiplication, 
    # the most robust "different" strategy that is likely to succeed (or fail fast) 
    # is to attempt a **Genetic Algorithm** with a fitness function based on the 
    # sum of squared errors, but constrained to integer values.
    
    # Let's implement a simple GA.
    
    def genetic_search(rank_to_try, max_generations=200, pop_size=50):
        if time.time() > deadline - 5:
            return None
            
        # Initialize population
        pop = []
        for _ in range(pop_size):
            U_g = np.random.choice(candidate_vals, size=(n*m, rank_to_try))
            V_g = np.random.choice(candidate_vals, size=(m*p, rank_to_try))
            W_g = np.random.choice(candidate_vals, size=(p*n, rank_to_try))
            pop.append((U_g, V_g, W_g))
            
        best_err = float('inf')
        best_solution = None
        
        for gen in range(max_generations):
            if time.time() > deadline - 5:
                break
                
            # Evaluate population
            for idx, (U_g, V_g, W_g) in enumerate(pop):
                # Compute error efficiently
                err = 0.0
                # Only check a subset of (i,j,k) to speed up? No, need exact.
                # But n*m*p is only 42. It's fast.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_g[i*m+j], V_g[j*p+k] * W_g[k*n+i])
                            err += (val - 1.0)**2
                            
                if err < best_err:
                    best_err = err
                    best_solution = (U_g.copy(), V_g.copy(), W_g.copy())
                    
                if err < 1e-9:
                    return best_solution
            
            # Selection: keep top 10
            pop = sorted(pop, key=lambda x: np.sum((np.dot(x[0][0], x[1][0] * x[2][0]) - 1)**2))[:10]
            
            # Crossover and Mutation
            new_pop = list(pop)
            while len(new_pop) < pop_size:
                p1 = pop[np.random.randint(0, len(pop))]
                p2 = pop[np.random.randint(0, len(pop))]
                
                # Uniform crossover
                U_c = np.where(np.random.rand(n*m, rank_to_try) < 0.5, p1[0], p2[0])
                V_c = np.where(np.random.rand(m*p, rank_to_try) < 0.5, p1[1], p2[1])
                W_c = np.where(np.random.rand(p*n, rank_to_try) < 0.5, p1[2], p2[2])
                
                # Mutation
                if np.random.rand() < 0.3:
                    row, col = np.random.randint(0, n*m), np.random.randint(0, rank_to_try)
                    U_c[row, col] = np.random.choice(candidate_vals)
                if np.random.rand() < 0.3:
                    row, col = np.random.randint(0, m*p), np.random.randint(0, rank_to_try)
                    V_c[row, col] = np.random.choice(candidate_vals)
                if np.random.rand() < 0.3:
                    row, col = np.random.randint(0, p*n), np.random.randint(0, rank_to_try)
                    W_c[row, col] = np.random.choice(candidate_vals)
                    
                new_pop.append((U_c, V_c, W_c))
            pop = new_pop
            
        return best_solution if best_err < 1e-9 else None

    # Try rank 41
    sol = genetic_search(41)
    if sol:
        best_rank = 41
        best_U, best_V, best_W = sol
        found_lower = True

    # If not found, try rank 40 (less likely, but let's try with less time)
    if not found_lower and time.time() < deadline - 20:
        sol = genetic_search(40, max_generations=100, pop_size=30)
        if sol:
            best_rank = 40
            best_U, best_V, best_W = sol
            found_lower = True

    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    if not U or not V or not W:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check the 0s
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                support.add((u, v, w))
                
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) in support:
                    continue
                val = sum(U[u, c] * V[v, c] * W[w, c] for c in range(rank))
                if abs(val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _random_decomposition(n, m, p, rank, rng):
    """Generate a random decomposition with integer/half-integer entries."""
    # Use entries from {-1, -0.5, 0, 0.5, 1}
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        # Randomly assign values
        U[:, c] = rng.choice(vals, size=n*m)
        V[:, c] = rng.choice(vals, size=m*p)
        W[:, c] = rng.choice(vals, size=p*n)
    
    return U, V, W

def _evaluate_decomposition(U, V, W, n, m, p):
    """Evaluate how close a decomposition is to being valid."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Target: 1 for support, 0 for others
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    
    err = np.sum((result - target) ** 2)
    
    # Also check that entries are close to half-integers
    U_err = np.sum((U * 2 - np.round(U * 2)) ** 2)
    V_err = np.sum((V * 2 - np.round(V * 2)) ** 2)
    W_err = np.sum((W * 2 - np.round(W * 2)) ** 2)
    
    return err + 0.1 * (U_err + V_err + W_err)

def _local_search(U, V, W, n, m, p, max_iter=500, time_limit=10.0):
    """Local search: try flipping individual entries to improve the objective."""
    start_time = time.time()
    rank = U.shape[1]
    
    best_err = _evaluate_decomposition(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly pick a matrix, row, column, and try a different value
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            rows, cols = U.shape
            mat = U
            other1, other2 = V, W
        elif mat_idx == 1:
            rows, cols = V.shape
            mat = V
            other1, other2 = U, W
        else:
            rows, cols = W.shape
            mat = W
            other1, other2 = U, V
        
        r = np.random.randint(rows)
        c = np.random.randint(cols)
        
        # Try each possible value
        best_val = mat[r, c]
        best_local_err = best_err
        
        for v in vals:
            if abs(v - mat[r, c]) < 1e-10:
                continue
            old_val = mat[r, c]
            mat[r, c] = v
            
            err = _evaluate_decomposition(U if mat_idx == 0 else best_U, 
                                          V if mat_idx == 1 else best_V, 
                                          W if mat_idx == 2 else best_W, 
                                          n, m, p)
            
            if err < best_local_err:
                best_local_err = err
                best_val = v
            else:
                mat[r, c] = old_val
        
        if best_val != mat[r, c]:
            mat[r, c] = best_val
            best_err = best_local_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None

def _genetic_search(n, m, p, rank, num_individuals=20, max_generations=100, time_limit=30.0):
    """Genetic algorithm to search for valid decompositions."""
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize population
    population = []
    for _ in range(num_individuals):
        U, V, W = _random_decomposition(n, m, p, rank, rng)
        err = _evaluate_decomposition(U, V, W, n, m, p)
        population.append((U, V, W, err))
    
    best_U, best_V, best_W, best_err = min(population, key=lambda x: x[3])[:3]
    best_err = min(population, key=lambda x: x[3])[3]
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by error
        population.sort(key=lambda x: x[3])
        
        # Check if we found a valid solution
        if population[0][3] < 1e-8:
            return population[0][0], population[0][1], population[0][2], population[0][3]
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top 20%
        elite_count = max(1, num_individuals // 5)
        for i in range(elite_count):
            new_population.append((population[i][0].copy(), population[i][1].copy(), 
                                   population[i][2].copy(), population[i][3]))
        
        # Crossover and mutation
        while len(new_population) < num_individuals:
            # Pick two parents from top half
            p1_idx = rng.randint(0, num_individuals // 2)
            p2_idx = rng.randint(0, num_individuals // 2)
            
            U1, V1, W1, _ = population[p1_idx]
            U2, V2, W2, _ = population[p2_idx]
            
            # Crossover: mix columns
            U_new = U1.copy()
            V_new = V1.copy()
            W_new = W1.copy()
            
            for c in range(rank):
                if rng.random() < 0.5:
                    U_new[:, c] = U2[:, c]
                    V_new[:, c] = V2[:, c]
                    W_new[:, c] = W2[:, c]
            
            # Mutation: flip some entries
            for _ in range(rank // 2):
                mat_idx = rng.randint(3)
                if mat_idx == 0:
                    rows = U_new.shape[0]
                    mat = U_new
                elif mat_idx == 1:
                    rows = V_new.shape[0]
                    mat = V_new
                else:
                    rows = W_new.shape[0]
                    mat = W_new
                
                r = rng.randint(rows)
                c = rng.randint(rank)
                mat[r, c] = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            err = _evaluate_decomposition(U_new, V_new, W_new, n, m, p)
            new_population.append((U_new, V_new, W_new, err))
        
        population = new_population
        
        # Update best
        best = min(population, key=lambda x: x[3])
        if best[3] < best_err:
            best_U, best_V, best_W, best_err = best[0], best[1], best[2], best[3]
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None

def _try_rank(n, m, p, rank, time_budget=15.0):
    """Try to find a valid decomposition of given rank."""
    start_time = time.time()
    
    # Method 1: Random restarts with local search
    for attempt in range(10):
        if time.time() - start_time > time_budget - 5:
            break
        
        rng = np.random.RandomState(attempt * 1000 + 42)
        U, V, W = _random_decomposition(n, m, p, rank, rng)
        
        result = _local_search(U, V, W, n, m, p, max_iter=200, time_limit=time_budget/10 - 1)
        if result is not None:
            return result
    
    # Method 2: Genetic algorithm
    if time.time() - start_time < time_budget - 10:
        result = _genetic_search(n, m, p, rank, num_individuals=15, 
                                  max_generations=50, 
                                  time_limit=time_budget - 10 - (time.time() - start_time))
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    max_rank_to_try = min(40, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 20:
            break
        
        # Allocate time budget based on rank
        time_budget = min(15.0, (deadline - time.time()) / 2)
        
        result = _try_rank(n, m, p, rank, time_budget=time_budget)
        
        if result is not None:
            U_opt, V_opt, W_opt, err = result
            
            # Round to nearest 0.5
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                # Found a valid lower rank, return immediately
                return best_result
    
    return best_result

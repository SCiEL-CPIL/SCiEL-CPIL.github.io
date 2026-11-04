import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 3]
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """
    Verify that the decomposition U, V, W correctly decomposes the 3x3 matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _check_half_integers(U, V, W, tol=1e-9):
    for M in [U, V, W]:
        for row in M:
            for x in row:
                if not _is_half_integer(x, tol):
                    return False
    return True

def _get_schoolbook(n=3, m=3, p=3):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _round_to_half_ints(M, tol=1e-6):
    """Round a matrix to nearest half-integers and check if close enough."""
    M_rounded = [[round(x * 2) / 2 for x in row] for row in M]
    M_np = np.array(M, dtype=float)
    M_round_np = np.array(M_rounded, dtype=float)
    if np.max(np.abs(M_np - M_round_np)) < tol:
        return M_rounded
    return None

def _optimize_rank(r, n=3, m=3, p=3, time_limit=30.0, n_restarts=5, seed=42):
    """
    Use genetic algorithm to find a rank-r decomposition.
    Returns (U, V, W) as lists of lists, or None if failed.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Precompute index mappings for efficiency
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    def compute_loss(U, V, W):
        loss = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            loss += (val - 1.0) ** 2
        return loss
    
    def crossover(U1, V1, W1, U2, V2, W2):
        # Uniform crossover
        mask = np.random.rand(n*m, r) < 0.5
        U_c = np.where(mask, U1, U2)
        mask = np.random.rand(m*p, r) < 0.5
        V_c = np.where(mask, V1, V2)
        mask = np.random.rand(p*n, r) < 0.5
        W_c = np.where(mask, W1, W2)
        return U_c, V_c, W_c
    
    def mutate(U, V, W, rate=0.1, scale=0.1):
        U_m = U.copy()
        V_m = V.copy()
        W_m = W.copy()
        
        mask = np.random.rand(*U.shape) < rate
        U_m[mask] += np.random.randn(*U.shape)[mask] * scale
        
        mask = np.random.rand(*V.shape) < rate
        V_m[mask] += np.random.randn(*V.shape)[mask] * scale
        
        mask = np.random.rand(*W.shape) < rate
        W_m[mask] += np.random.randn(*W.shape)[mask] * scale
        
        return U_m, V_m, W_m
    
    # Genetic algorithm
    pop_size = 20
    generations = 100
    elite_size = 2
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        population.append((U, V, W))
    
    best_loss = float('inf')
    best_solution = None
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate fitness
        losses = []
        for U, V, W in population:
            loss = compute_loss(U, V, W)
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
        
        # Sort by loss
        sorted_indices = np.argsort(losses)
        population = [population[i] for i in sorted_indices]
        
        if best_loss < 1e-10:
            break
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
            
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if losses[sorted_indices[idx1]] < losses[sorted_indices[idx2]]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if losses[sorted_indices[idx1]] < losses[sorted_indices[idx2]]:
                parent2 = population[idx1]
            else:
                parent2 = population[idx2]
            
            # Crossover
            U_c, V_c, W_c = crossover(parent1[0], parent1[1], parent1[2],
                                       parent2[0], parent2[1], parent2[2])
            
            # Mutation
            U_m, V_m, W_m = mutate(U_c, V_c, W_c)
            
            new_population.append((U_m, V_m, W_m))
        
        population = new_population
    
    # After optimization, round to half-integers
    if best_solution is not None:
        U, V, W = best_solution
        U_r = _round_to_half_ints(U.tolist())
        V_r = _round_to_half_ints(V.tolist())
        W_r = _round_to_half_ints(W.tolist())
        
        if U_r is not None and V_r is not None and W_r is not None:
            if _verify_decomposition(U_r, V_r, W_r):
                if _check_half_integers(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _find_decomposition(n=3, m=3, p=3, max_rank=27, time_limit=100.0):
    """
    Attempt to find a rank-minimal decomposition.
    Try ranks from 23 down to 19, then fall back to schoolbook.
    """
    start_time = time.time()
    
    # Try to find solutions with decreasing rank
    for r in [23, 22, 21, 20, 19]:
        if time.time() - start_time > time_limit - 5:
            break
        
        time_for_this_rank = min(25.0, time_limit - (time.time() - start_time))
        result = _optimize_rank(r, n, m, p, time_limit=time_for_this_rank, n_restarts=1)
        
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return _get_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, max_rank=27, time_limit=DEADLINE - SAFETY_MARGIN)
    
    # Safety check: ensure we didn't exceed time
    if time.time() > deadline:
        return _get_schoolbook(n, m, p)
        
    return result

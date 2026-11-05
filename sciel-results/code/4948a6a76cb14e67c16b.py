import itertools
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    # New Approach: Integer Linear Programming (ILP)
    # This is a fundamentally different approach from the heuristic local search in the seed.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: Simple Greedy (kept for safety, but task specifies q=3, n=7, r=1)
        return _greedy_construct_fallback(q, n, radius)

    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute ball indices for each codeword candidate
    # ball_indices[i] is a list of indices of words covered by candidate i
    ball_indices = []
    for i in range(N):
        w = words[i]
        covered = set()
        covered.add(i)
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    covered.add(word_to_idx[tuple(new_w)])
        ball_indices.append(covered)
        
    # Formulate ILP
    # Variables: x_i for i in 0..N-1, binary
    # Objective: Minimize sum(x_i)
    c = np.ones(N)
    
    # Constraints: For each word w_j, sum(x_i for i in ball_indices[j]) >= 1
    # We build the constraint matrix A such that A @ x >= 1
    # A[j, i] = 1 if i covers j, else 0
    # Note: scipy.optimize.milp expects A @ x <= ub and A @ x >= lb
    # So we define A where rows correspond to words (constraints) and cols to candidates (variables)
    # A[j, i] = 1 if candidate i covers word j
    
    # Efficient construction of A
    # A is sparse. We can use scipy.sparse for construction if needed, but milp accepts dense or sparse?
    # scipy.optimize.milp accepts sparse arrays for A.
    from scipy.sparse import coo_matrix
    
    rows = []
    cols = []
    data = []
    
    for i in range(N):
        for j in ball_indices[i]:
            rows.append(j)
            cols.append(i)
            data.append(1)
            
    A = coo_matrix((data, (rows, cols)), shape=(N, N))
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(N), ub=np.inf)
    constraints = LinearConstraint(A, lb=np.ones(N), ub=np.inf)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 for all variables (binary)
    integrality = np.ones(N)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x = res.x
            code_indices = [i for i in range(N) if x[i] > 0.5]
            return [words[i] for i in code_indices]
        else:
            # If ILP fails or times out, fall back to greedy
            # But milp usually returns a feasible solution if status is not 0? 
            # Status 0: Optimal
            # Status 1: Infeasible (should not happen)
            # Status 2: Unbounded (should not happen)
            # Status 3: Infeasible or unbounded
            # Status 4: Numerical error
            # Status 5: Iteration limit reached (but may have feasible solution)
            # Status 6: Time limit reached (but may have feasible solution)
            
            if res.x is not None:
                x = res.x
                # Check if feasible
                # We need to verify coverage
                cover_count = np.zeros(N, dtype=np.int32)
                code_indices = [i for i in range(N) if x[i] > 0.5]
                for i in code_indices:
                    for j in ball_indices[i]:
                        cover_count[j] += 1
                if np.all(cover_count >= 1):
                    return [words[i] for i in code_indices]
                
            # Fallback to greedy if ILP didn't provide a valid solution
            return _greedy_construct_fallback(q, n, radius)

    except Exception as e:
        # In case of any error, fallback to greedy
        return _greedy_construct_fallback(q, n, radius)

def _greedy_construct_fallback(q, n, radius):
    # Simple Greedy for fallback
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute balls for greedy
    balls = [set() for _ in range(N)]
    for i in range(N):
        w = words[i]
        balls[i].add(i)
        for pos in range(n):
            for delta in range(-radius, radius + 1):
                if delta == 0: continue
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    balls[i].add(word_to_idx[tuple(new_w)])
                    
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            cov = len(balls[i] & uncovered)
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        uncovered -= balls[best_i]
        
    return [words[i] for i in code]

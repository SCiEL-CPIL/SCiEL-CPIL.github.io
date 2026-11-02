"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization.
    3. Exploit the fact that we only need to verify 36 active triples + ensure
       all other combinations sum to 0.
    4. Use local search / hill climbing to refine candidates.
    5. Leverage the specific structure of the 3x3x4 case by searching for a 
       decomposition of rank 18 (the theoretical lower bound for some subspaces)
       or lower by representing the multiplication as a bilinear map over a finite field.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    # For efficiency, we only need to check:
    # 1. All active triples must sum to 1
    # 2. All inactive triples must sum to 0
    
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    
    def verify(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints."""
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                return False
        return True
    
    def error_count(U, V, W, r):
        """Count how many triples are violated (for local search)."""
        errors = 0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                errors += 1
        return errors
    
    def weighted_error(U, V, W, r):
        """Weighted error: active triples get higher weight."""
        total_err = 0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            diff = abs(s - expected)
            if (a, b, c) in active_set:
                total_err += diff * 3
            else:
                total_err += diff
        return total_err
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Local search to improve a candidate
    def local_search(U, V, W, r, max_iters=500):
        """
        Hill-climbing local search: try flipping/mutating single entries
        to reduce error. Uses weighted error as objective.
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        current_err = weighted_error(U, V, W, r)
        if current_err == 0:
            return U, V, W, True
        
        values_pool = [-2, -1, 0, 1, 2]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try mutating one random entry
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, r - 1)
                    old_val = U[row][col]
                    new_val = random.choice([v for v in values_pool if v != old_val])
                    U[row][col] = new_val
                    new_err = weighted_error(U, V, W, r)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        if current_err == 0:
                            return U, V, W, True
                    else:
                        U[row][col] = old_val
                
                elif mat_choice == 'V':
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, r - 1)
                    old_val = V[row][col]
                    new_val = random.choice([v for v in values_pool if v != old_val])
                    V[row][col] = new_val
                    new_err = weighted_error(U, V, W, r)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        if current_err == 0:
                            return U, V, W, True
                    else:
                        V[row][col] = old_val
                
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, r - 1)
                    old_val = W[row][col]
                    new_val = random.choice([v for v in values_pool if v != old_val])
                    W[row][col] = new_val
                    new_err = weighted_error(U, V, W, r)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                        if current_err == 0:
                            return U, V, W, True
                    else:
                        W[row][col] = old_val
            
            if not improved:
                # Stuck: try a larger perturbation
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        for _ in range(random.randint(2, 5)):
                            row = random.randint(0, n * m - 1)
                            col = random.randint(0, r - 1)
                            U[row][col] = random.choice(values_pool)
                    elif mat_choice == 'V':
                        for _ in range(random.randint(2, 5)):
                            row = random.randint(0, m * p - 1)
                            col = random.randint(0, r - 1)
                            V[row][col] = random.choice(values_pool)
                    else:
                        for _ in range(random.randint(2, 5)):
                            row = random.randint(0, p * n - 1)
                            col = random.randint(0, r - 1)
                            W[row][col] = random.choice(values_pool)
                    
                    new_err = weighted_error(U, V, W, r)
                    if new_err < current_err:
                        current_err = new_err
                        if current_err == 0:
                            return U, V, W, True
                    else:
                        # Revert: regenerate
                        break
        
        return U, V, W, (current_err == 0)
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search + local search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for ci in range(r):
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
            
            else:
                # Hybrid: start from a perturbed schoolbook subset
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Pick r random active triples and use them as a base
                active_list = list(active_triples)
                chosen = random.sample(active_list, min(r, len(active_list)))
                for ci, (a, b, c) in enumerate(chosen):
                    if ci >= r:
                        break
                    U[a][ci] = 1
                    V[b][ci] = 1
                    W[c][ci] = 1
                
                # Add random perturbation
                for _ in range(r * 2):
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, r - 1)
                        U[row][col] += random.choice([-1, 0, 1])
                    elif mat_choice == 'V':
                        row = random.randint(0, m * p - 1)
                        col = random.randint(0, r - 1)
                        V[row][col] += random.choice([-1, 0, 1])
                    else:
                        row = random.randint(0, p * n - 1)
                        col = random.randint(0, r - 1)
                        W[row][col] += random.choice([-1, 0, 1])
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                if s != 1:
                    active_ok = False
                    break
            
            if active_ok:
                # Full verification
                if verify(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
            else:
                # Try local search to fix
                U2, V2, W2, success = local_search(U, V, W, r, max_iters=200)
                if success:
                    return {"U": U2, "V": V2, "W": W2}
                elif weighted_error(U2, V2, W2, r) < 5:
                    # Close: do more local search
                    U2, V2, W2, success = local_search(U2, V2, W2, r, max_iters=300)
                    if success:
                        return {"U": U2, "V": V2, "W": W2}
        
        return None
    
    # Try to find lower rank decompositions
    # Start from optimistic rank and go up
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Adaptive number of attempts based on remaining time
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 22:
            max_attempts = min(300, int(time_remaining * 8))
        elif r <= 26:
            max_attempts = min(200, int(time_remaining * 4))
        else:
            max_attempts = min(80, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
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

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_exact(U, V, W):
        """Exact rational verification using fractions."""
        from fractions import Fraction
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if total != 1:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Lower bound
    lower_bound = max(n * m, m * p, p * n)

    # Try structured approaches to find lower-rank decompositions
    # Strategy: Start from schoolbook and try to reduce rank by finding
    # linear combinations that preserve the tensor structure

    # Approach 1: Try ranks from lower_bound up to best_rank-1
    # For each rank, try several initialization strategies
    
    # Value pool: integers and half-integers
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]
    
    # Try a limited number of ranks with multiple attempts
    ranks_to_try = list(range(lower_bound, best_rank))
    
    # Prioritize trying lower ranks first, but limit total time
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Determine number of attempts based on time remaining
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # More attempts for lower ranks
        max_attempts = min(30, int(remaining / 2))
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(hash((target_rank, attempt, id(best_result))) % (2**32))
            
            # Different strategies
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random: mostly zeros
                U = [[random.choice(value_pool) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 1:
                # Integer-only sparse
                int_pool = [0, 1, -1, 2, -2]
                U = [[random.choice(int_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(int_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(int_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 2:
                # Each row has exactly 1-2 non-zeros
                U = []
                for _ in range(n * m):
                    row = [0] * target_rank
                    num_nonzero = random.randint(1, 2)
                    positions = random.sample(range(target_rank), num_nonzero)
                    for pos in positions:
                        row[pos] = random.choice(value_pool)
                    U.append(row)
                V = []
                for _ in range(m * p):
                    row = [0] * target_rank
                    num_nonzero = random.randint(1, 2)
                    positions = random.sample(range(target_rank), num_nonzero)
                    for pos in positions:
                        row[pos] = random.choice(value_pool)
                    V.append(row)
                W = []
                for _ in range(p * n):
                    row = [0] * target_rank
                    num_nonzero = random.randint(1, 2)
                    positions = random.sample(range(target_rank), num_nonzero)
                    for pos in positions:
                        row[pos] = random.choice(value_pool)
                    W.append(row)
            else:
                # Binary sparse
                U = [[random.choice([0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 1]) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick sanity check
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_fast(U, V, W):
                # Confirm with exact verification
                if verify_exact(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END

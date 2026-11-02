"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
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

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # Use einsum: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Try to find lower-rank decompositions using structured random search
    # Strategy: try various rank values with different initialization strategies

    # Small value pool for half-integer entries
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]

    # Try ranks from lower bound up to current best
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break

        # Limit attempts per rank to stay within time budget
        max_attempts = 50 if target_rank < 40 else 20 if target_rank < 60 else 10

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Different initialization strategies based on attempt number
            strategy = attempt % 5

            if strategy == 0:
                # Pure random from value pool
                U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 1:
                # Sparse random: mostly zeros with a few non-zeros
                U = [[random.choice(value_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) if random.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 2:
                # Structured: each row has exactly 2 non-zeros
                U = []
                for _ in range(n * m):
                    row = [0] * target_rank
                    non_zero_positions = random.sample(range(target_rank), min(2, target_rank))
                    for pos in non_zero_positions:
                        row[pos] = random.choice(value_pool)
                    U.append(row)
                V = []
                for _ in range(m * p):
                    row = [0] * target_rank
                    non_zero_positions = random.sample(range(target_rank), min(2, target_rank))
                    for pos in non_zero_positions:
                        row[pos] = random.choice(value_pool)
                    V.append(row)
                W = []
                for _ in range(p * n):
                    row = [0] * target_rank
                    non_zero_positions = random.sample(range(target_rank), min(2, target_rank))
                    for pos in non_zero_positions:
                        row[pos] = random.choice(value_pool)
                    W.append(row)
            elif strategy == 3:
                # Integer-only random
                int_pool = [0, 1, -1, 2, -2]
                U = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(p * n)]
            else:
                # Binary random (0 or 1)
                U = [[random.choice([0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 1]) for _ in range(target_rank)] for _ in range(p * n)]

            # Quick check: verify each factor has at least one non-zero row
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue

            if verify_fast(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break

    return best_result
# EVOLVE-BLOCK-END

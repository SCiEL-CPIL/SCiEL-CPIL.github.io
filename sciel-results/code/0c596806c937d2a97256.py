"""Matrix multiplication 3,3,7 - rank minimization with verification and time guard."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 150s limit

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, num_samples=15):
        """Verify tensor decomposition correctness using random integer matrices."""
        rng = random.Random(42)
        rank = len(U[0]) if U else 0
        for _ in range(num_samples):
            A = [[rng.randint(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-2, 2) for _ in range(p)] for _ in range(m)]
            # True product
            C_true = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_true[i][k] = s
            # Product via factorization: C[i,k] = sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            C_fact = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        uc = U[i * m + j][c]
                        if uc == 0:
                            continue
                        for k in range(p):
                            vc = V[j * p + k][c]
                            if vc == 0:
                                continue
                            wc = W[k * n + i][c]
                            if wc == 0:
                                continue
                            C_fact[i][k] += uc * vc * wc
            for i in range(n):
                for k in range(p):
                    if C_true[i][k] != C_fact[i][k]:
                        return False
        return True

    # Try to find a lower rank decomposition using structured random search
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank

    # Try several target ranks with denser random initializations
    target_ranks = [62, 60, 58, 55, 50]
    max_attempts_per_rank = 4

    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            rng = random.Random(10000 + target_rank * 100 + attempt)
            # Use denser random matrices with entries in {-1, 0, 1}
            U = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]

            if verify(U, V, W, num_samples=15):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break

    # Return best found or schoolbook
    if best_result is not None:
        return best_result
    return schoolbook()
# EVOLVE-BLOCK-END

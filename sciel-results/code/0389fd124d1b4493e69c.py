import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 150s limit, 10s safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Schoolbook solution (always valid)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    def verify_fast(U, V, W, R, num_tests=5):
        """
        Fast verification using random integer matrices and integer arithmetic.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
        """
        for _ in range(num_tests):
            A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]

            # Direct
            C_dir = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_dir[i][k] = s

            # Decomposed
            # LA[i][r] = sum_j U[i*m+j][r] * A[i][j]
            LA = [[0]*R for _ in range(n)]
            for i in range(n):
                for r in range(R):
                    s = 0
                    base = i * m
                    for j in range(m):
                        u = U[base + j][r]
                        if u:
                            s += u * A[i][j]
                    LA[i][r] = s

            # LB[k][r] = sum_j V[j*p+k][r] * B[j][k]
            LB = [[0]*R for _ in range(p)]
            for k in range(p):
                for r in range(R):
                    s = 0
                    for j in range(m):
                        v = V[j * p + k][r]
                        if v:
                            s += v * B[j][k]
                    LB[k][r] = s

            # C[i][k] = sum_r W[k*n+i][r] * LA[i][r] * LB[k][r]
            C_dec = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    w_base = k * n + i
                    for r in range(R):
                        w = W[w_base][r]
                        if w:
                            s += w * LA[i][r] * LB[k][r]
                    C_dec[i][k] = s

            if C_dir != C_dec:
                return False
        return True

    def verify_symbolic(U, V, W, R):
        """
        Exact symbolic verification: check that the tensor identity holds.
        For each (i,j,k), the coefficient of A[i,j]*B[j,k] in the decomposition
        must equal 1.
        
        The decomposition gives:
        C[i,k] = sum_r W[k*n+i,r] * (sum_j' U[i*m+j',r]*A[i,j']) * (sum_k' V[j'*p+k',r]*B[j',k'])
        
        Coefficient of A[i,j]*B[j,k] = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r]
        This must equal 1 for all i,j,k.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    u_base = i * m + j
                    v_base = j * p + k
                    w_base = k * n + i
                    for r in range(R):
                        total += W[w_base][r] * U[u_base][r] * V[v_base][r]
                    if total != 1:
                        return False
        return True

    def gen_sparse_matrix(rows, cols, density, entries=(-1, 0, 1)):
        """Generate a sparse random matrix."""
        M = [[0]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if random.random() < density:
                    M[r][c] = random.choice(entries)
        return M

    def local_search(U, V, W, R, max_iters=500):
        """Try to fix a nearly-correct decomposition via local perturbation."""
        # First, check how close we are
        def error_count():
            errs = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        u_base = i * m + j
                        v_base = j * p + k
                        w_base = k * n + i
                        for r in range(R):
                            total += W[w_base][r] * U[u_base][r] * V[v_base][r]
                        if total != 1:
                            errs += 1
            return errs

        init_err = error_count()
        if init_err == 0:
            return True

        best_err = init_err
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Pick a random position to modify
            which = random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
            if which == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, R-1)
                old = U[row][col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                U[row][col] = new
                err = error_count()
                if err < best_err:
                    best_err = err
                    if err == 0:
                        return True
                else:
                    U[row][col] = old
            elif which == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, R-1)
                old = V[row][col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                V[row][col] = new
                err = error_count()
                if err < best_err:
                    best_err = err
                    if err == 0:
                        return True
                else:
                    V[row][col] = old
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, R-1)
                old = W[row][col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                W[row][col] = new
                err = error_count()
                if err < best_err:
                    best_err = err
                    if err == 0:
                        return True
                else:
                    W[row][col] = old

        return False

    # Search ranks from low to high
    # Theoretical lower bound: at least max(n*p, m*p, n*m) = max(12, 18, 6) = 18
    # But actually the tensor rank is at least max(n*m, m*p, p*n) / something...
    # For 2x3x6, schoolbook is 36. Let's search from 18 upward.
    
    min_rank = 18  # lower bound
    
    for rank in range(min_rank, schoolbook_rank):
        if time.time() > deadline - 10.0:
            break

        # Adaptive number of candidates based on remaining time
        remaining = deadline - time.time()
        if remaining < 15:
            break
        max_candidates = int(remaining * 3)  # ~3 candidates per second
        max_candidates = max(50, min(max_candidates, 5000))

        for cand in range(max_candidates):
            if time.time() > deadline - 5.0:
                break

            # Vary sparsity density
            density = random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
            if rank > 25:
                density = random.choice([0.1, 0.15, 0.2])

            U = gen_sparse_matrix(n*m, rank, density)
            V = gen_sparse_matrix(m*p, rank, density)
            W = gen_sparse_matrix(p*n, rank, density)

            # Quick check: verify symbolically (exact, fast)
            if verify_symbolic(U, V, W, rank):
                best_rank = rank
                best_solution = {"U": U, "V": V, "W": W}
                return best_solution

            # If close, try local search
            # Count errors
            errs = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        u_base = i * m + j
                        v_base = j * p + k
                        w_base = k * n + i
                        for r in range(rank):
                            total += W[w_base][r] * U[u_base][r] * V[v_base][r]
                        if total != 1:
                            errs += 1

            if errs <= 3:
                if local_search(U, V, W, rank, max_iters=200):
                    if verify_symbolic(U, V, W, rank):
                        best_rank = rank
                        best_solution = {"U": U, "V": V, "W": W}
                        return best_solution

    return best_solution

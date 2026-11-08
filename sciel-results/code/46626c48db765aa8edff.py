import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build the list of (a, b, c) triples with T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    num_eq = len(triples)
    num_rows_U = n * m  # 6
    num_rows_V = m * p  # 21
    num_rows_W = p * n  # 14

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W (as lists of lists) exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for (a, b, c) in triples:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        return True

    # Rank 39 fallback construction (Strassen-style blocks)
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1

            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1

            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1

            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1

            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1

            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1

            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1

            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])

        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])

        return {"U": U, "V": V, "W": W}

    fallback_result = construct_39()

    # Precompute tensor structure for fast error evaluation
    triple_a = np.array([t[0] for t in triples])
    triple_b = np.array([t[1] for t in triples])
    triple_c = np.array([t[2] for t in triples])

    def compute_error(U, V, W):
        """Compute sum of squared errors efficiently."""
        vals = np.zeros(len(triples))
        for r in range(U.shape[1]):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            vals += u_col[triple_a] * v_col[triple_b] * w_col[triple_c]
        return np.sum((vals - 1.0) ** 2)

    def try_local_search(R, time_budget=20):
        """Try to find a rank-R decomposition via local search from rank-39."""
        t_start = time.time()
        
        U39 = np.array(fallback_result['U'], dtype=float)
        V39 = np.array(fallback_result['V'], dtype=float)
        W39 = np.array(fallback_result['W'], dtype=float)
        
        np.random.seed(42)
        best_err = np.inf
        best_solution = None
        
        for attempt in range(15):
            if time.time() - t_start > time_budget - 2:
                break
            
            # Initialize: take first R columns of rank-39 solution
            U = U39[:, :R].copy()
            V = V39[:, :R].copy()
            W = W39[:, :R].copy()
            
            # Randomly perturb some entries to escape local minima
            for _ in range(20):
                if time.time() - t_start > time_budget - 1:
                    break
                choice = np.random.random()
                if choice < 0.2:
                    row = np.random.randint(num_rows_U)
                    mat = U
                elif choice < 0.6:
                    row = np.random.randint(num_rows_V)
                    mat = V
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                col = np.random.randint(R)
                mat[row, col] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            err = compute_error(U, V, W)
            if err < 1e-9:
                if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Local search iterations
            max_iters = 500
            for it in range(max_iters):
                if time.time() - t_start > time_budget - 1:
                    break
                
                if err < 1e-9:
                    if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                # Pick a random entry to modify
                choice = np.random.random()
                if choice < 0.15:
                    row = np.random.randint(num_rows_U)
                    mat = U
                elif choice < 0.55:
                    row = np.random.randint(num_rows_V)
                    mat = V
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                
                col = np.random.randint(R)
                old_val = mat[row, col]
                
                # Try candidate values
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue
                
                mat[row, col] = new_val
                err_new = compute_error(U, V, W)
                
                if err_new < err:
                    err = err_new
                else:
                    mat[row, col] = old_val
        
        # Final polish on best solution found
        if best_solution is not None and best_err < 1e-4:
            U, V, W = best_solution
            # High-precision polish
            for _ in range(200):
                if time.time() - t_start > time_budget - 0.5:
                    break
                err = compute_error(U, V, W)
                if err < 1e-12:
                    break
                
                choice = np.random.random()
                if choice < 0.15:
                    row = np.random.randint(num_rows_U)
                    mat = U
                elif choice < 0.55:
                    row = np.random.randint(num_rows_V)
                    mat = V
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                
                col = np.random.randint(R)
                old_val = mat[row, col]
                
                # Try integer and half-integer values
                for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    if new_val == old_val:
                        continue
                    mat[row, col] = new_val
                    err_new = compute_error(U, V, W)
                    if err_new < err:
                        err = err_new
                        break
                    mat[row, col] = old_val
            
            if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try ranks from 38 down to 34
    for R in range(38, 33, -1):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        
        result = try_local_search(R, time_budget=min(remaining, 25))
        if result is not None:
            return result

    return fallback_result
# EVOLVE-BLOCK-END

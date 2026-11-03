"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, V: V, "W": W}

    def verify_numpy(U, V, W, atol=1e-9):
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=atol):
                return False
            return True
        except Exception:
            return False

    def check_half_integer(values):
        for v in values:
            if not np.isfinite(v):
                return False
            if abs(v * 2 - round(v * 2)) > 1e-9:
                return False
        return True

    def is_valid_decomposition(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                if not check_half_integer(row):
                    return False
        return verify_numpy(U, V, W)

    def compute_error(U, V, W):
        """Compute the max absolute error of the tensor decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T - 1.0))

    def try_remove_column(U, V, W, col_idx):
        """Try removing a column and see if we can fix the error by adjusting remaining entries."""
        rank = U.shape[1]
        if rank <= 1:
            return None
        
        # Remove column
        U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
        V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
        W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
        
        new_rank = rank - 1
        U_np = np.array(U_new, dtype=float)
        V_np = np.array(V_new, dtype=float)
        W_np = np.array(W_new, dtype=float)
        
        U_r = U_np.reshape(n, m, new_rank)
        V_r = V_np.reshape(m, p, new_rank)
        W_r = W_np.reshape(p, n, new_rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = T - 1.0
        max_err = np.max(np.abs(err))
        
        if max_err < 1e-9:
            return (U_new, V_new, W_new)
        
        # Try to fix by adjusting entries - use the column that was removed as a guide
        # The removed column had specific values; try to redistribute
        removed_U = [U[row][col_idx] for row in range(n*m)]
        removed_V = [V[row][col_idx] for row in range(m*p)]
        removed_W = [W[row][col_idx] for row in range(p*n)]
        
        # For each non-zero entry in the removed column, try to absorb it
        # into other columns
        for c in range(new_rank):
            # Try adjusting U entries
            for i in range(n):
                for j in range(m):
                    delta = removed_U[i*m+j]
                    if abs(delta) < 1e-9:
                        continue
                    U_test = U_np.copy()
                    U_test[i, j, c] += delta
                    U_r_test = U_test.reshape(n, m, new_rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r_test, V_r, W_r)
                    err_test = T_test - 1.0
                    if np.max(np.abs(err_test)) < 1e-9:
                        U_test_list = U_test.tolist()
                        V_test_list = V_np.tolist()
                        W_test_list = W_np.tolist()
                        # Round to half-integers
                        U_rounded = [[round(v * 2) / 2 for v in row] for row in U_test_list]
                        V_rounded = [[round(v * 2) / 2 for v in row] for row in V_test_list]
                        W_rounded = [[round(v * 2) / 2 for v in row] for row in W_test_list]
                        if is_valid_decomposition(U_rounded, V_rounded, W_rounded):
                            return (U_rounded, V_rounded, W_rounded)
            
            # Try adjusting V entries
            for j in range(m):
                for k in range(p):
                    delta = removed_V[j*p+k]
                    if abs(delta) < 1e-9:
                        continue
                    V_test = V_np.copy()
                    V_test[j, k, c] += delta
                    V_r_test = V_test.reshape(m, p, new_rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r_test, W_r)
                    err_test = T_test - 1.0
                    if np.max(np.abs(err_test)) < 1e-9:
                        U_test_list = U_np.tolist()
                        V_test_list = V_test.tolist()
                        W_test_list = W_np.tolist()
                        U_rounded = [[round(v * 2) / 2 for v in row] for row in U_test_list]
                        V_rounded = [[round(v * 2) / 2 for v in row] for row in V_test_list]
                        W_rounded = [[round(v * 2) / 2 for v in row] for row in W_test_list]
                        if is_valid_decomposition(U_rounded, V_rounded, W_rounded):
                            return (U_rounded, V_rounded, W_rounded)
            
            # Try adjusting W entries
            for k in range(p):
                for i in range(n):
                    delta = removed_W[k*n+i]
                    if abs(delta) < 1e-9:
                        continue
                    W_test = W_np.copy()
                    W_test[k, i, c] += delta
                    W_r_test = W_test.reshape(p, n, new_rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r_test)
                    err_test = T_test - 1.0
                    if np.max(np.abs(err_test)) < 1e-9:
                        U_test_list = U_np.tolist()
                        V_test_list = V_np.tolist()
                        W_test_list = W_test.tolist()
                        U_rounded = [[round(v * 2) / 2 for v in row] for row in U_test_list]
                        V_rounded = [[round(v * 2) / 2 for v in row] for row in V_test_list]
                        W_rounded = [[round(v * 2) / 2 for v in row] for row in W_test_list]
                        if is_valid_decomposition(U_rounded, V_rounded, W_rounded):
                            return (U_rounded, V_rounded, W_rounded)
        
        return None

    def greedy_reduce(U, V, W, max_time=10):
        """Greedy rank reduction by trying to remove columns."""
        start = time.time()
        current_U, current_V, current_W = U[:], V[:], W[:]
        current_rank = current_U[0].__len__() if current_U else 0
        
        while time.time() - start < max_time:
            rank = current_U[0].__len__() if current_U else 0
            if rank <= 1:
                break
            
            found_removal = False
            # Try each column in random order
            cols = list(range(rank))
            random.shuffle(cols)
            
            for col_idx in cols:
                if time.time() - start > max_time:
                    break
                
                result = try_remove_column(current_U, current_V, current_W, col_idx)
                if result is not None:
                    current_U, current_V, current_W = result
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        return current_U, current_V, current_W

    def local_refine(U, V, W, max_iter=50, step=0.5):
        """Refine a near-miss decomposition by adjusting entries."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = T - 1.0
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if np.max(np.abs(err)) < 1e-9:
                break
            
            idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            i, j, k = idx
            
            best_improvement = np.inf
            best_change = None
            
            for c in range(rank):
                for delta in [-step, step]:
                    U_test = U_np.copy()
                    U_test[i, j, c] += delta
                    U_r_test = U_test.reshape(n, m, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r_test, V_r, W_r)
                    max_err = np.max(np.abs(T_test - 1.0))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('U', i, j, c, delta)
            
            for c in range(rank):
                for delta in [-step, step]:
                    V_test = V_np.copy()
                    V_test[j, k, c] += delta
                    V_r_test = V_test.reshape(m, p, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r_test, W_r)
                    max_err = np.max(np.abs(T_test - 1.0))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('V', j, k, c, delta)
            
            for c in range(rank):
                for delta in [-step, step]:
                    W_test = W_np.copy()
                    W_test[k, i, c] += delta
                    W_r_test = W_test.reshape(p, n, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r_test)
                    max_err = np.max(np.abs(T_test - 1.0))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('W', k, i, c, delta)
            
            if best_change is None:
                break
            
            mat_name, a, b, c, delta = best_change
            if mat_name == 'U':
                U_np[a, b, c] += delta
            elif mat_name == 'V':
                V_np[a, b, c] += delta
            else:
                W_np[a, b, c] += delta
            
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = T - 1.0
        
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
        
        U_rounded = [[round(v * 2) / 2 for v in row] for row in U_list]
        V_rounded = [[round(v * 2) / 2 for v in row] for row in V_list]
        W_rounded = [[round(v * 2) / 2 for v in row] for row in W_list]
        
        if is_valid_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        return None

    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p

    # Phase 1: Greedy reduction from schoolbook
    if time.time() < deadline - 5:
        U_g, V_g, W_g = greedy_reduce(best_result['U'], best_result['V'], best_result['W'], max_time=30)
        if is_valid_decomposition(U_g, V_g, W_g):
            best_result = {"U": U_g, "V": V_g, "W": W_g}
            best_rank = U_g[0].__len__() if U_g else 0

    # Phase 2: Structured sparse random search
    value_pool = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for target_rank in range(max(28, best_rank - 20), best_rank):
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 2:
            break
        
        if target_rank < 40:
            max_attempts = min(30, int(remaining_time * 0.2))
        elif target_rank < 55:
            max_attempts = min(15, int(remaining_time * 0.1))
        else:
            max_attempts = min(5, int(remaining_time * 0.05))
        
        if max_attempts < 1:
            max_attempts = 1
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Sparse generation: most entries are 0
            U = [[random.choice(value_pool) if random.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) if random.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) if random.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_numpy(U, V, W):
                if is_valid_decomposition(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
            
            max_err = compute_error(U, V, W)
            if max_err < 0.5:
                refined = local_refine(U, V, W, max_iter=15, step=0.5)
                if refined is not None:
                    U_ref, V_ref, W_ref = refined
                    if is_valid_decomposition(U_ref, V_ref, W_ref):
                        best_result = {"U": U_ref, "V": V_ref, "W": W_ref}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            continue

    return best_result
# EVOLVE-BLOCK-END

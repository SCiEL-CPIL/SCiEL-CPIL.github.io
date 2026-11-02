"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

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

    def verify_numpy(U, V, W, atol=1e-9):
        """Vectorized verification using numpy."""
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
        """Check if all values are integers or half-integers."""
        for v in values:
            if not np.isfinite(v):
                return False
            if abs(v * 2 - round(v * 2)) > 1e-9:
                return False
        return True

    def is_valid_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        # Check entries are integers or half-integers
        for mat in [U, V, W]:
            for row in mat:
                if not check_half_integer(row):
                    return False
        # Verify the tensor decomposition
        return verify_numpy(U, V, W)

    def local_refine(U, V, W, max_iter=100, step=0.5):
        """Try to refine a near-miss decomposition by adjusting entries."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Compute error
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
            
            # Find the largest error
            idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            i, j, k = idx
            err_val = err[i, j, k]
            
            # Try adjusting one entry in U, V, or W
            best_improvement = np.inf
            best_change = None
            
            # Try adjusting U[i,j,c]
            for c in range(rank):
                for delta in [-step, step]:
                    U_test = U_np.copy()
                    U_test[i, j, c] += delta
                    U_r_test = U_test.reshape(n, m, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r_test, V_r, W_r)
                    err_test = T_test - 1.0
                    max_err = np.max(np.abs(err_test))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('U', i, j, c, delta)
            
            # Try adjusting V[j,k,c]
            for c in range(rank):
                for delta in [-step, step]:
                    V_test = V_np.copy()
                    V_test[j, k, c] += delta
                    V_r_test = V_test.reshape(m, p, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r_test, W_r)
                    err_test = T_test - 1.0
                    max_err = np.max(np.abs(err_test))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('V', j, k, c, delta)
            
            # Try adjusting W[k,i,c]
            for c in range(rank):
                for delta in [-step, step]:
                    W_test = W_np.copy()
                    W_test[k, i, c] += delta
                    W_r_test = W_test.reshape(p, n, rank)
                    T_test = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r_test)
                    err_test = T_test - 1.0
                    max_err = np.max(np.abs(err_test))
                    if max_err < best_improvement:
                        best_improvement = max_err
                        best_change = ('W', k, i, c, delta)
            
            if best_change is None:
                break
            
            # Apply the best change
            mat_name, a, b, c, delta = best_change
            if mat_name == 'U':
                U_np[a, b, c] += delta
            elif mat_name == 'V':
                V_np[a, b, c] += delta
            else:
                W_np[a, b, c] += delta
            
            # Recompute error
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = T - 1.0
        
        # Convert back to lists and check if valid
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
        
        # Round to nearest half-integer
        U_rounded = [[round(v * 2) / 2 for v in row] for row in U_list]
        V_rounded = [[round(v * 2) / 2 for v in row] for row in V_list]
        W_rounded = [[round(v * 2) / 2 for v in row] for row in W_list]
        
        if is_valid_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        return None

    best_result = schoolbook()
    best_rank = n * m * p

    # More comprehensive value pool
    value_pool = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    
    # Try ranks from lower bound up to best_rank-1
    # Focus more on lower ranks
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
        
        # Allocate attempts based on rank
        if target_rank < 35:
            max_attempts = min(50, int(remaining_time * 0.3))
        elif target_rank < 45:
            max_attempts = min(30, int(remaining_time * 0.2))
        elif target_rank < 60:
            max_attempts = min(15, int(remaining_time * 0.1))
        else:
            max_attempts = min(5, int(remaining_time * 0.05))
        
        if max_attempts < 1:
            max_attempts = 1
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding
            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Generate random matrices with values from the pool
            U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check: verify the decomposition
            if verify_numpy(U, V, W):
                # Check half-integer constraint (should be satisfied by construction)
                if is_valid_decomposition(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
            
            # If close but not exact, try local refinement
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            U_r = U_np.reshape(n, m, target_rank)
            V_r = V_np.reshape(m, p, target_rank)
            W_r = W_np.reshape(p, n, target_rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            max_err = np.max(np.abs(T - 1.0))
            
            if max_err < 1.0:  # Close enough to try refinement
                refined = local_refine(U, V, W, max_iter=20, step=0.5)
                if refined is not None:
                    U_ref, V_ref, W_ref = refined
                    if is_valid_decomposition(U_ref, V_ref, W_ref):
                        best_result = {"U": U_ref, "V": V_ref, "W": W_ref}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            continue  # Move to next rank
    
    return best_result
# EVOLVE-BLOCK-END

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

    import numpy as np

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

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        """Check if entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    # Check if x is integer or half-integer
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    # Strategy: Greedy rank reduction from schoolbook
    # Start with schoolbook (rank 84) and try to eliminate terms by finding
    # linear dependencies in the factor spaces
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def greedy_reduce(U, V, W, max_iterations=1000):
        """Try to reduce rank by removing dependent terms."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        current_rank = U_np.shape[1]
        
        # We'll work with numpy arrays for efficiency
        # Try to remove one term at a time
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            if current_rank <= 28:  # Lower bound
                break
            
            # Try each possible term to remove
            found_removal = False
            for c_remove in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Create matrices without column c_remove
                mask = [i for i in range(current_rank) if i != c_remove]
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]
                
                # Check if the tensor is still valid
                U_r = U_test.reshape(n, m, -1)
                V_r = V_test.reshape(m, p, -1)
                W_r = W_test.reshape(p, n, -1)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                if np.allclose(T, 1.0, atol=1e-9):
                    # Check integer/half-integer constraint
                    if verify_integer(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                        U_np = U_test
                        V_np = V_test
                        W_np = W_test
                        current_rank -= 1
                        found_removal = True
                        break
            
            if not found_removal:
                # Try a different strategy: look for linear combinations
                # For now, try random perturbations to escape local optima
                if random.random() < 0.3 and current_rank > 30:
                    # Try adding a random term and then reducing
                    new_rank = current_rank + 1
                    U_new = np.zeros((n*m, new_rank))
                    V_new = np.zeros((m*p, new_rank))
                    W_new = np.zeros((p*n, new_rank))
                    
                    # Copy existing
                    U_new[:, :current_rank] = U_np
                    V_new[:, :current_rank] = V_np
                    W_new[:, :current_rank] = W_np
                    
                    # Add random term
                    pool = [0, 1, -1, 0.5, -0.5]
                    U_new[:, current_rank] = [random.choice(pool) for _ in range(n*m)]
                    V_new[:, current_rank] = [random.choice(pool) for _ in range(m*p)]
                    W_new[:, current_rank] = [random.choice(pool) for _ in range(p*n)]
                    
                    # Now try to reduce from this larger rank
                    success = False
                    for c_remove in range(new_rank):
                        mask = [i for i in range(new_rank) if i != c_remove]
                        U_test = U_new[:, mask]
                        V_test = V_new[:, mask]
                        W_test = W_new[:, mask]
                        
                        U_r = U_test.reshape(n, m, -1)
                        V_r = V_test.reshape(m, p, -1)
                        W_r = W_test.reshape(p, n, -1)
                        
                        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                        
                        if np.allclose(T, 1.0, atol=1e-9):
                            if verify_integer(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                                U_np = U_test
                                V_np = V_test
                                W_np = W_test
                                current_rank = new_rank - 1
                                success = True
                                break
                    
                    if not success:
                        break
        
        return U_np.tolist(), V_np.tolist(), W_np.tolist(), current_rank
    
    # Apply greedy reduction
    if time.time() < deadline - 5:
        U_reduced, V_reduced, W_reduced, rank_reduced = greedy_reduce(
            best_result["U"], best_result["V"], best_result["W"]
        )
        if rank_reduced < best_rank:
            best_result = {"U": U_reduced, "V": V_reduced, "W": W_reduced}
            best_rank = rank_reduced
    
    # Also try random search for potentially better results
    value_pool = [0, 1, -1, 0.5, -0.5]
    
    # Try ranks from lower bound up to current best
    lower_bound = max(n*m, m*p, p*n)
    for target_rank in range(lower_bound, min(best_rank, 45)):
        if time.time() > deadline:
            break
        
        # Limit attempts per rank to stay within time budget
        max_attempts = 20 if target_rank < 35 else 10
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured initialization
            if attempt < 8:
                pool = [0, 1, -1]
            else:
                pool = value_pool
            
            # Generate random matrices with some structure
            U = [[random.choice(pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick sparsity check
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_fast(U, V, W):
                if verify_integer(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

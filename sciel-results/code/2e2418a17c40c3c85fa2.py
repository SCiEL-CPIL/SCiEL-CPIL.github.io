import time

DIMENSIONS = [3, 4, 5]
_DEADLINE = 150.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE - 5.0

# EVOLVE-BLOCK-START
def construct():
    # Attempt to find a lower-rank decomposition for 3x4x5 matrix multiplication.
    # Fall back to schoolbook rank-60 if no improvement found within time budget.
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    def _schoolbook():
        rank = schoolbook_rank
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
    
    # Try to find a rank-r decomposition for r < 60 using numerical optimization
    # with half-integer rounding and verification
    
    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta[i,k] for all i,j,k
        # Actually, the tensor for matrix multiplication T[i,j,k] = 1 if i,k are free and j is contracted
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this indexing)
        # Wait, let me reconsider. The matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k.
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def _try_rank(r, time_budget):
        """Try to find a rank-r decomposition using numerical optimization."""
        if r >= schoolbook_rank:
            return None
        
        start_time = time.time()
        
        # Initialize with random half-integer values
        import random
        random.seed(42)
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Simple iterative approach: start from a perturbed schoolbook-like initialization
        # and try to reduce rank by zeroing out columns and adjusting
        
        # For now, try a few random initializations and gradient descent
        for attempt in range(3):
            if time.time() - start_time > time_budget:
                break
            
            # Initialize U, V, W with small random half-integers
            U = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(p * n)]
            
            # Iteratively improve using coordinate descent
            for iteration in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                # Compute error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for rr in range(r):
                                total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                            error += (total - 1) ** 2
                
                if error < 1e-12:
                    # Found exact solution
                    return (U, V, W)
                
                if error < best_error:
                    best_error = error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                
                # Coordinate descent: update one entry at a time
                # This is very slow for large r, so skip if r is large
                if r > 30:
                    break
                
                # Try updating a few random entries
                for _ in range(10):
                    mat_choice = random.randint(0, 2)  # 0=U, 1=V, 2=W
                    if mat_choice == 0:
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, r - 1)
                        old_val = U[row][col]
                        # Try a few candidate values
                        for new_val in [0, 0.5, -0.5, 1, -1]:
                            if new_val == old_val:
                                continue
                            U[row][col] = new_val
                            err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        total = 0
                                        for rr in range(r):
                                            total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                                        err += (total - 1) ** 2
                            if err < error:
                                error = err
                                if error < best_error:
                                    best_error = error
                                    best_U = [row2[:] for row2 in U]
                                    best_V = [row2[:] for row2 in V]
                                    best_W = [row2[:] for row2 in W]
                            else:
                                U[row][col] = old_val
                    elif mat_choice == 1:
                        row = random.randint(0, m * p - 1)
                        col = random.randint(0, r - 1)
                        old_val = V[row][col]
                        for new_val in [0, 0.5, -0.5, 1, -1]:
                            if new_val == old_val:
                                continue
                            V[row][col] = new_val
                            err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        total = 0
                                        for rr in range(r):
                                            total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                                        err += (total - 1) ** 2
                            if err < error:
                                error = err
                                if error < best_error:
                                    best_error = error
                                    best_U = [row2[:] for row2 in U]
                                    best_V = [row2[:] for row2 in V]
                                    best_W = [row2[:] for row2 in W]
                            else:
                                V[row][col] = old_val
                    else:
                        row = random.randint(0, p * n - 1)
                        col = random.randint(0, r - 1)
                        old_val = W[row][col]
                        for new_val in [0, 0.5, -0.5, 1, -1]:
                            if new_val == old_val:
                                continue
                            W[row][col] = new_val
                            err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        total = 0
                                        for rr in range(r):
                                            total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                                        err += (total - 1) ** 2
                            if err < error:
                                error = err
                                if error < best_error:
                                    best_error = error
                                    best_U = [row2[:] for row2 in U]
                                    best_V = [row2[:] for row2 in V]
                                    best_W = [row2[:] for row2 in W]
                            else:
                                W[row][col] = old_val
        
        # Check if best solution is exact
        if best_error < 1e-9 and best_U is not None:
            if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from 59 down to 30, with decreasing time budgets
    # Use most of the time for higher ranks (more likely to succeed)
    total_time = 120.0  # safety margin
    elapsed = time.time() - _START_TIME
    remaining = max(0, total_time - elapsed - 5.0)  # 5s safety margin
    
    # Allocate time: try a few promising ranks
    ranks_to_try = [59, 58, 57, 56, 55, 50, 45, 40]
    time_per_rank = remaining / len(ranks_to_try)
    
    for r in ranks_to_try:
        if _check_deadline():
            break
        if time.time() - _START_TIME > _DEADLINE - 10.0:
            break
        
        result = _try_rank(r, time_per_rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

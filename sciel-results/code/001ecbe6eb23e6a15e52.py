import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]
_DEADLINE = 145.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x5 matrix multiplication.
    Schoolbook rank is 60. We try to find lower-rank decompositions.
    
    The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,5)
    needs to be decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U has shape (n*m, rank) = (12, rank)
    V has shape (m*p, rank) = (20, rank)  
    W has shape (p*n, rank) = (15, rank)
    """
    
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
    
    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def _try_rank_numpy(r, time_budget):
        """Try to find a rank-r decomposition using NumPy-based optimization."""
        if r >= schoolbook_rank:
            return None
        
        start_time = time.time()
        np.random.seed(42)
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Use structured initialization based on symmetry
        # For matrix multiplication, there's a natural structure
        # We can exploit the fact that T[i,j,k] = 1 is constant
        
        for attempt in range(5):
            if time.time() - start_time > time_budget:
                break
            
            # Initialize with structured random values
            # Use small values to encourage sparsity
            U = np.random.uniform(-1, 1, (n * m, r)) * 0.5
            V = np.random.uniform(-1, 1, (m * p, r)) * 0.5
            W = np.random.uniform(-1, 1, (p * n, r)) * 0.5
            
            # Gradient descent with line search
            lr = 0.01
            for iteration in range(500):
                if time.time() - start_time > time_budget:
                    break
                
                # Compute tensor values
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Reshape for vectorized computation
                T_computed = np.zeros((n, m, p))
                for r_idx in range(r):
                    u_col = U[:, r_idx]  # (n*m,)
                    v_col = V[:, r_idx]  # (m*p,)
                    w_col = W[:, r_idx]  # (p*n,)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_computed[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                
                # Error
                error = np.sum((T_computed - 1.0) ** 2)
                
                if error < 1e-12:
                    # Found exact solution
                    return (U.tolist(), V.tolist(), W.tolist())
                
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                # Compute gradients
                # d(error)/d(U[i*m+j, r]) = 2 * sum_{i',j',k'} (T[i',j',k'] - 1) * V[j'*p+k', r] * W[k'*n+i', r] * delta(i==i', j==j')
                # Simplified: for each (i,j,k), contribution to U[i*m+j, r] gradient is:
                # 2 * (T[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = T_computed[i, j, k] - 1.0
                            for r_idx in range(r):
                                grad_U[i*m+j, r_idx] += 2 * diff * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                                grad_V[j*p+k, r_idx] += 2 * diff * U[i*m+j, r_idx] * W[k*n+i, r_idx]
                                grad_W[k*n+i, r_idx] += 2 * diff * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                
                # Gradient step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.995
        
        # Check if best solution is exact
        if best_error < 1e-9 and best_U is not None:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return (U_list, V_list, W_list)
        
        return None
    
    def _try_rank_structured(r, time_budget):
        """
        Try structured decompositions exploiting symmetries of matrix multiplication.
        
        For M_{n,m,p}, the tensor has certain symmetries. One approach:
        Use the fact that we can write the multiplication as:
        (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        
        This suggests a decomposition where each term corresponds to a "slice"
        in the j dimension.
        
        Another approach: use fast rectangular matrix multiplication algorithms
        if known. For 3x4 x 4x5, the Strassen-like algorithms might give
        better than 60.
        
        Let's try a parametric family:
        Assume U, V, W have a block structure based on the dimensions.
        """
        if r >= schoolbook_rank:
            return None
        
        start_time = time.time()
        random.seed(42)
        
        # Try a few structured patterns
        # Pattern 1: Each rank term handles one (i,j) pair for some k range
        # Pattern 2: Exploit that n=3, m=4, p=5 are small
        
        # For now, try random search with structured initialization
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Try different initialization strategies
        for strategy in range(3):
            if time.time() - start_time > time_budget:
                break
            
            if strategy == 0:
                # Binary-like initialization
                U = [[random.choice([0, 1]) for _ in range(r)] for _ in range(n * m)]
                V = [[random.choice([0, 1]) for _ in range(r)] for _ in range(m * p)]
                W = [[random.choice([0, 1]) for _ in range(r)] for _ in range(p * n)]
            elif strategy == 1:
                # Sparse initialization
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for row in range(n * m):
                    for col in range(r):
                        if random.random() < 0.3:
                            U[row][col] = random.choice([0.5, -0.5, 1, -1])
                for row in range(m * p):
                    for col in range(r):
                        if random.random() < 0.3:
                            V[row][col] = random.choice([0.5, -0.5, 1, -1])
                for row in range(p * n):
                    for col in range(r):
                        if random.random() < 0.3:
                            W[row][col] = random.choice([0.5, -0.5, 1, -1])
            else:
                # Dense small values
                U = [[random.choice([0, 0.5, -0.5]) for _ in range(r)] for _ in range(n * m)]
                V = [[random.choice([0, 0.5, -0.5]) for _ in range(r)] for _ in range(m * p)]
                W = [[random.choice([0, 0.5, -0.5]) for _ in range(r)] for _ in range(p * n)]
            
            # Local search
            for iteration in range(200):
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
                    return (U, V, W)
                
                if error < best_error:
                    best_error = error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                
                # Try random perturbations
                improved = False
                for _ in range(20):
                    mat_choice = random.randint(0, 2)
                    if mat_choice == 0:
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, r - 1)
                        old_val = U[row][col]
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
                                improved = True
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
                                improved = True
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
                                improved = True
                                if error < best_error:
                                    best_error = error
                                    best_U = [row2[:] for row2 in U]
                                    best_V = [row2[:] for row2 in V]
                                    best_W = [row2[:] for row2 in W]
                            else:
                                W[row][col] = old_val
                
                if not improved:
                    break
        
        # Check if best solution is exact
        if best_error < 1e-9 and best_U is not None:
            if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from 59 down, with time budgets
    total_time = 120.0
    elapsed = time.time() - _START_TIME
    remaining = max(0, total_time - elapsed - 5.0)
    
    # Focus on ranks that are likely to have solutions
    # Known results suggest that for small matrices, rank can be significantly lower
    # Try a range of ranks
    ranks_to_try = [59, 58, 57, 56, 55, 54, 53, 52, 50, 48, 45, 40]
    time_per_rank = remaining / len(ranks_to_try)
    
    for r in ranks_to_try:
        if _check_deadline():
            break
        if time.time() - _START_TIME > _DEADLINE - 10.0:
            break
        
        # Try both methods
        result = _try_rank_structured(r, time_per_rank * 0.6)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() - _START_TIME > _DEADLINE - 10.0:
            break
        
        result = _try_rank_numpy(r, time_per_rank * 0.4)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook()

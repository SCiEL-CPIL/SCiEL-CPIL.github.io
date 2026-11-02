"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 140s deadline (10s safety margin)
    
    def schoolbook():
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
    
    def validate(U, V, W):
        """Check if decomposition is exact using integer/rational arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def validate_fast(U, V, W):
        """Fast validation using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        rank = U_np.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if s != 1:
                        return False
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Start with schoolbook, then try to reduce rank by finding
    # linear dependencies or merging terms
    
    # Approach 1: Greedy term elimination
    # Start with schoolbook, try to remove each term and see if we can fix
    # the resulting errors by adjusting other terms
    
    def greedy_elimination(start_rank, time_limit=10.0):
        """Try to eliminate terms from a decomposition."""
        start_time = time.time()
        U = np.zeros((n*m, start_rank), dtype=np.int64)
        V = np.zeros((m*p, start_rank), dtype=np.int64)
        W = np.zeros((p*n, start_rank), dtype=np.int64)
        
        # Initialize with schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        current_rank = start_rank
        U_active = U[:, :current_rank].copy()
        V_active = V[:, :current_rank].copy()
        W_active = W[:, :current_rank].copy()
        
        # Try to eliminate each column
        for col_to_remove in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            if current_rank <= best_rank:
                break
                
            # Check if this column can be eliminated
            # A column c can be eliminated if its contribution is a linear combination
            # of other columns, or if we can adjust other columns to compensate
            
            # Simple test: zero out this column and see if we can fix it
            test_U = U_active.copy()
            test_V = V_active.copy()
            test_W = W_active.copy()
            
            # Zero out column col_to_remove
            test_U[:, col_to_remove] = 0
            test_V[:, col_to_remove] = 0
            test_W[:, col_to_remove] = 0
            
            # Check if still valid
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = np.sum(test_U[i*m+j] * test_V[j*p+k] * test_W[k*n+i])
                        if s != 1:
                            valid = False
                            break
            
            if valid:
                # This column was redundant! Remove it
                mask = np.ones(current_rank, dtype=bool)
                mask[col_to_remove] = False
                U_active = test_U[:, mask]
                V_active = test_V[:, mask]
                W_active = test_W[:, mask]
                current_rank -= 1
        
        if current_rank < best_rank:
            U_list = U_active.tolist()
            V_list = V_active.tolist()
            W_list = W_active.tolist()
            if validate(U_list, V_list, W_list):
                return U_list, V_list, W_list, current_rank
        
        return None
    
    # Try greedy elimination from schoolbook
    result = greedy_elimination(best_rank, time_limit=20.0)
    if result is not None:
        U_list, V_list, W_list, rank = result
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Approach 2: Structured random search with better initialization
    np.random.seed(42)
    
    def compute_error(U, V, W):
        """Compute total error of decomposition."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    err += (s - 1) ** 2
        return err
    
    def local_search(U, V, W, time_limit=5.0):
        """Local search to fix errors."""
        start_time = time.time()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        best_err = compute_error(U, V, W)
        if best_err == 0:
            return U.tolist(), V.tolist(), W.tolist(), True
        
        for iteration in range(10000):
            if time.time() - start_time > time_limit:
                break
            
            # Try flipping a random entry
            mat_choice = np.random.random()
            if mat_choice < 0.33:
                row = np.random.randint(n*m)
                col = np.random.randint(rank)
                if U[row, col] != 0:
                    U[row, col] *= -1
            elif mat_choice < 0.66:
                row = np.random.randint(m*p)
                col = np.random.randint(rank)
                if V[row, col] != 0:
                    V[row, col] *= -1
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(rank)
                if W[row, col] != 0:
                    W[row, col] *= -1
            
            new_err = compute_error(U, V, W)
            if new_err < best_err:
                best_err = new_err
                if best_err == 0:
                    return U.tolist(), V.tolist(), W.tolist(), True
            else:
                # Revert
                if mat_choice < 0.33:
                    U[row, col] *= -1
                elif mat_choice < 0.66:
                    V[row, col] *= -1
                else:
                    W[row, col] *= -1
        
        return None, None, None, False
    
    # Try structured decompositions with lower rank
    # Use block structure: split dimensions and use Strassen-like methods
    
    # For 3x4*4x7, try to use the fact that we can decompose into smaller problems
    # C = A*B where A is 3x4, B is 4x7
    
    # Try rank values from best_rank-1 down to 50
    for target_rank in range(best_rank - 1, max(50, best_rank - 20), -1):
        if time.time() > deadline - 30:
            break
        
        # Initialize with sparse random
        U = np.zeros((n*m, target_rank), dtype=np.int64)
        V = np.zeros((m*p, target_rank), dtype=np.int64)
        W = np.zeros((p*n, target_rank), dtype=np.int64)
        
        # Initialize with schoolbook-like pattern, truncated
        for c in range(min(target_rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            if i < n and j < m and k < p:
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        
        # Add some random entries
        for c in range(target_rank):
            if np.random.random() < 0.5:
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i*m+j, c] = np.random.choice([-1, 1])
                V[j*p+k, c] = np.random.choice([-1, 1])
                W[k*n+i, c] = np.random.choice([-1, 1])
        
        # Local search
        U_list, V_list, W_list, found = local_search(U, V, W, time_limit=3.0)
        if found:
            if validate(U_list, V_list, W_list):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
# EVOLVE-BLOCK-END

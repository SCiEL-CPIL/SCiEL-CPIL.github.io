import random
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import solve

DIMENSIONS = [3, 5, 6]

def _verify_decomposition(U, V, W, n, m, p, seed=12345):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = random.Random(seed)
    A = [[rng.randint(0, 10) for _ in range(m)] for _ in range(n)]
    B = [[rng.randint(0, 10) for _ in range(p)] for _ in range(m)]
    
    # Standard matrix multiplication
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C[i][k] += A[i][j] * B[j][k]
    
    # Using tensor decomposition
    C_prime = [[0.0] * p for _ in range(n)]
    for c in range(rank):
        sum_u = sum(U[i * m + j, c] * A[i][j] for i in range(n) for j in range(m))
        sum_v = sum(V[j * p + k, c] * B[j][k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_prime[i][k] += sum_u * sum_v * W[k * n + i, c]
    
    # Check exactness (integer arithmetic should give exact results)
    for i in range(n):
        for k in range(p):
            if abs(C[i][k] - C_prime[i][k]) > 1e-9:
                return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=500, tol=1e-10):
    """Try to reduce rank by zeroing out columns and checking if decomposition still works."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = len(U[0])
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Try to find a set of columns to keep that still gives correct decomposition
    # Use a greedy approach: try removing one column at a time
    
    # First, let's try to use SVD to find a low-rank approximation
    # The tensor can be viewed as a 3-way tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} (in a specific indexing)
    # Actually for (n x m) * (m x p), the tensor M_{i,j,k} where:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has entries: M[i,j,k] = 1 if this term contributes, which is always 1
    # The decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try a different approach: use the fact that we can try to find
    # a lower rank by solving a system of equations
    
    # The constraint is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in rank*3 variables (with nonlinearity)
    
    # Let's try a simpler approach: randomly select subsets of columns and check
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for attempt in range(100):
        # Randomly select target_rank columns
        cols = random.sample(range(current_rank), target_rank)
        U_sub = U[:, cols]
        V_sub = V[:, cols]
        W_sub = W[:, cols]
        
        if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
            return U_sub, V_sub, W_sub
    
    # If random selection didn't work, try to optimize
    # We'll use a penalty-based approach
    
    def _residual(params, U, V, W, n, m, p):
        """Compute the residual of the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Residual: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residual += (s - 1.0) ** 2
        return residual
    
    # Try to minimize the residual while pushing some entries toward zero
    # to effectively reduce rank
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 90
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try progressively lower ranks
    for target_rank in range(rank - 1, max(10, rank - 30), -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a valid decomposition with this rank
        # Use a random restart approach with local optimization
        
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start from schoolbook and try to zero out columns
            U_try = U_arr.copy()
            V_try = V_arr.copy()
            W_try = W_arr.copy()
            
            # Randomly select which columns to keep
            cols = random.sample(range(rank), target_rank)
            U_sub = U_arr[:, cols]
            V_sub = V_arr[:, cols]
            W_sub = W_arr[:, cols]
            
            if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                break
        
        if best_rank <= target_rank:
            continue
    
    # Convert back to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Final verification
    if not _verify_decomposition(U_out, V_out, W_out, n, m, p):
        # Fall back to schoolbook
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    # Self-test
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Verify shapes
    assert len(U) == n * m, f"U rows: {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows: {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows: {len(W)} != {p*n}"
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify correctness with random test
    import random
    random.seed(12345)
    A = [[random.randint(0, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(0, 10) for _ in range(p)] for _ in range(m)]
    
    # Standard matrix multiplication
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C[i][k] += A[i][j] * B[j][k]
    
    # Using tensor decomposition
    C_prime = [[0] * p for _ in range(n)]
    for c in range(rank):
        sum_u = sum(U[i * m + j][c] * A[i][j] for i in range(n) for j in range(m))
        sum_v = sum(V[j * p + k][c] * B[j][k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_prime[i][k] += sum_u * sum_v * W[k * n + i][c]
    
    for i in range(n):
        for k in range(p):
            assert abs(C[i][k] - C_prime[i][k]) < 1e-9, f"Mismatch at ({i},{k}): {C[i][k]} vs {C_prime[i][k]}"
    
    print("Decomposition verified. Rank =", rank)


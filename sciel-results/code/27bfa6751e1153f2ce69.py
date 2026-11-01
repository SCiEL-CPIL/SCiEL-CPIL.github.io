"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def _verify_decomposition(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    rng = np.random.RandomState(42)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # factor_A[i] = sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            # factor_B[k] = sum_l V[l*p+k, c] * B[l,k]
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[l * p + k][c] * B[l, k] for l in range(m))
            # Add contribution
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_schoolbook(n=2, m=4, p=8):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _try_remove_column(U, V, W, col_idx, n=2, m=4, p=8, trials=30):
    """Try removing column col_idx and verify."""
    rank = len(U[0])
    U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
    V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
    W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
    return _verify_decomposition(U_test, V_test, W_test, n, m, p, trials=trials)

def _greedy_rank_reduction(U, V, W, n=2, m=4, p=8, max_rounds=20, trials=30):
    """Greedily remove columns that can be removed without breaking correctness."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    for _ in range(max_rounds):
        removed = False
        for c in range(rank):
            if _try_remove_column(U, V, W, c, n, m, p, trials=trials):
                U = [row[:c] + row[c+1:] for row in U]
                V = [row[:c] + row[c+1:] for row in V]
                W = [row[:c] + row[c+1:] for row in W]
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, rank

def _randomized_search(start_rank, n=2, m=4, p=8, max_attempts=100, deadline=None):
    """Try to find a lower-rank decomposition via random perturbations and reduction."""
    best_U, best_V, best_W, best_rank = None, None, None, start_rank
    
    # Start with schoolbook
    U0, V0, W0, _ = _construct_schoolbook(n, m, p)
    
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline:
            break
        
        # Create a perturbed version
        U = [row[:] for row in U0]
        V = [row[:] for row in V0]
        W = [row[:] for row in W0]
        
        # Add small random perturbations to create linear dependencies
        for _ in range(3):
            ri = random.randint(0, len(U) - 1)
            ci = random.randint(0, start_rank - 1)
            perturb = random.choice([-0.5, 0.5, -1, 1])
            U[ri][ci] += perturb
            ri_v = random.randint(0, len(V) - 1)
            V[ri_v][ci] += perturb
            ri_w = random.randint(0, len(W) - 1)
            W[ri_w][ci] += perturb
        
        # Verify and try to reduce
        if _verify_decomposition(U, V, W, n, m, p, trials=20):
            U_r, V_r, W_r, rank_r = _greedy_rank_reduction(U, V, W, n, m, p, max_rounds=10, trials=20)
            if rank_r < best_rank:
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_r
                print(f"Found rank {best_rank}")
    
    return best_U, best_V, best_W, best_rank

def _structured_search(n=2, m=4, p=8, deadline=None):
    """Try structured lower-rank decompositions using algebraic identities."""
    best_U, best_V, best_W, best_rank = None, None, None, n * m * p
    
    # Try to find a decomposition where we can merge some terms
    # Key insight: for fixed i, the products A[i,j]*B[j,k] for different k
    # can sometimes be expressed with shared rank-1 terms
    
    # Attempt 1: Try rank = n*m*p - 1 by finding one linear dependency
    # The constraint is: sum_c W[k*n+i,c] * U[i*m+j,c] * V[l*p+k,c] = delta_{j,l}
    
    # Let's try to find a solution with rank 63
    # We'll use a heuristic: start with schoolbook, zero out one column,
    # and adjust others to compensate
    
    for target_rank in range(n * m * p - 1, n * m * p - 10, -1):
        if deadline and time.time() > deadline:
            break
        
        # Try random initialization with target rank
        for trial in range(20):
            if deadline and time.time() > deadline:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with random integer/half-integer values
            for i in range(n * m):
                for c in range(target_rank):
                    U[i][c] = random.choice([0, 0.5, -0.5, 1, -1])
            for i in range(m * p):
                for c in range(target_rank):
                    V[i][c] = random.choice([0, 0.5, -0.5, 1, -1])
            for i in range(p * n):
                for c in range(target_rank):
                    W[i][c] = random.choice([0, 0.5, -0.5, 1, -1])
            
            if _verify_decomposition(U, V, W, n, m, p, trials=10):
                U_r, V_r, W_r, rank_r = _greedy_rank_reduction(U, V, W, n, m, p, max_rounds=5, trials=10)
                if rank_r < best_rank:
                    best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_r
                    print(f"Structured search found rank {best_rank}")
    
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W, rank = _construct_schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U[:], V[:], W[:], rank
    
    # Try greedy rank reduction on schoolbook
    U_r, V_r, W_r, rank_r = _greedy_rank_reduction(U, V, W, n, m, p, max_rounds=5, trials=20)
    if rank_r < best_rank:
        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_r
    
    # Try randomized search
    U_rand, V_rand, W_rand, rank_rand = _randomized_search(schoolbook_rank, n, m, p, max_attempts=50, deadline=deadline)
    if rank_rand < best_rank:
        best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rand, rank_rand
    
    # Try structured search
    U_str, V_str, W_str, rank_str = _structured_search(n, m, p, deadline=deadline)
    if rank_str < best_rank:
        best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, trials=50):
        # Fall back to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

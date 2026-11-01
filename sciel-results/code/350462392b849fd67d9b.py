"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    """
    Construct a minimal-rank exact decomposition of the (2,4,4) matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Approach: Use a randomized search to find a low-rank decomposition.
    We'll try ranks from 24 up to 32, using a greedy/local search approach.
    
    Key insight: The problem is equivalent to finding the tensor rank of the 
    (2,4,4) matrix multiplication tensor. We'll use a numerical approach:
    1. Start with a random initialization for a given rank R
    2. Use gradient-free local search to minimize the error
    3. Check if we can achieve exact (or near-exact) decomposition
    
    For the specific case of (2,4,4), we can also try structured approaches:
    - Block decomposition using Strassen-like ideas
    - Exploiting the small dimension n=2
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[i*m+j])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False, s
        return True, 1.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_error(U, V, W):
        """Compute the max error over all (i,j,k) entries."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[i*m+j])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def random_init(rank, seed=None):
        """Initialize U, V, W with small random values."""
        if seed is not None:
            random.seed(seed)
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(nm)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(mp)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(pn)]
        return U, V, W
    
    def local_search(U, V, W, max_iters=2000, step=0.1):
        """Perform a simple local search to minimize the error."""
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        best_err = compute_error(best_U, best_V, best_W)
        
        for it in range(max_iters):
            if best_err < 1e-10:
                break
            
            # Pick a random element to perturb
            choice = random.randint(0, 2)
            if choice == 0:
                r = random.randint(0, nm-1)
                c = random.randint(0, len(U[r])-1)
                delta = random.uniform(-step, step)
                U[r][c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                else:
                    U[r][c] -= delta
            elif choice == 1:
                r = random.randint(0, mp-1)
                c = random.randint(0, len(V[r])-1)
                delta = random.uniform(-step, step)
                V[r][c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                else:
                    V[r][c] -= delta
            else:
                r = random.randint(0, pn-1)
                c = random.randint(0, len(W[r])-1)
                delta = random.uniform(-step, step)
                W[r][c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                else:
                    W[r][c] -= delta
        
        return best_U, best_V, best_W, best_err
    
    # Try to find a lower-rank decomposition
    # Start with rank 24 and go up to 32
    for rank in [24, 25, 26, 27, 28, 29, 30, 31]:
        # Try multiple random initializations
        for attempt in range(5):
            seed = rank * 100 + attempt
            U, V, W = random_init(rank, seed=seed)
            
            # Scale to make initial error reasonable
            # The sum of products should be 1, so scale factors matter
            # Let's try a few different scalings
            for scale in [0.5, 1.0, 2.0]:
                U_s = [[x * scale for x in row] for row in U]
                V_s = [[x * scale for x in row] for row in V]
                W_s = [[x / (scale * scale) for x in row] for row in W]
                
                err = compute_error(U_s, V_s, W_s)
                if err < 0.1:
                    # Promising start, do local search
                    U_l, V_l, W_l, final_err = local_search(U_s, V_s, W_s, max_iters=500, step=0.05)
                    if final_err < 1e-8:
                        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END

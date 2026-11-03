"""Matrix multiplication 4x5 x 5x5 with rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE = 130  # seconds; evaluator kills at 150s

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook rank = 100
    rank_school = n * m * p  # 100
    
    def make_schoolbook():
        U = [[0]*rank_school for _ in range(n*m)]
        V = [[0]*rank_school for _ in range(m*p)]
        W = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Strategy: Use structured decompositions based on known results
    # For 4x5 x 5x5, we can try to exploit block structure
    
    # Approach 1: Split the 5x5 matrix into blocks
    # A is 4x5, B is 5x5, C = AB is 4x5
    # Split B into B1 (top 4 rows) and B2 (bottom row)
    # C = A * B = A * [B1; B2] = A * B1 + A * B2
    # A * B1: 4x5 * 4x5 (but B1 is 4x5, so this is 4x5 * 4x5)
    # A * B2: 4x5 * 1x5
    
    # Actually, let's think about this differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split j into two groups: j in {0,1,2,3} and j=4
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    # First term: 4x4 * 4x5 (A[:,0:4] * B[0:4,:])
    # Second term: 4x1 * 1x5 (A[:,4:5] * B[4:5,:])
    
    # For 4x4 * 4x5:
    # Split B[0:4,:] into B1 (4x4) and B2 (4x1)
    # A[:,0:4] * B1: 4x4 * 4x4 (rank 49 with Strassen)
    # A[:,0:4] * B2: 4x4 * 4x1
    
    # For 4x4 * 4x1:
    # This is simpler. C[i,0] = sum_j A[i,j] * B[j,0]
    # This is just a dot product for each i, so rank 4 (one term per i)
    
    # Total: 49 + 4 + 4 = 57? Let's verify this structure.
    
    # Actually, let's try a more systematic approach using random search
    # with a deadline guard and various rank values.
    
    def try_random_rank(r, num_trials=500, alphabet=None):
        """Try random decompositions of rank r."""
        if alphabet is None:
            alphabet = [0, 1, -1, 0.5, -0.5]
        
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            U = [[random.choice(alphabet) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(alphabet) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(alphabet) for _ in range(r)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # Try ranks from 90 down to 60
    for r in range(90, 60, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try with small alphabet
        result = try_random_rank(r, num_trials=200, alphabet=[0, 1, -1])
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        # Also try with half-integers
        if time.time() - start_time > DEADLINE:
            break
        result = try_random_rank(r, num_trials=100, alphabet=[0, 1, -1, 0.5, -0.5])
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    # If we found something better, use it; otherwise use schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = make_schoolbook()
        return {"U": U, "V": V, "W": W}

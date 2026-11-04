import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based rank-41 decomposition for 4x4x5
    # Use Strassen for 4x4x4 (rank 49 is too high, let's use a better approach)
    # Actually, let's try to construct a rank-35 solution:
    # 4x4x4 part with rank 19 (BCL) + 4x4x1 part with rank 16 = 35
    # But I need BCL coefficients. Let me try a different known algorithm.
    
    # Alternative: Use the fact that we can build 4x4x5 from smaller blocks.
    # Let's try a direct construction using known low-rank methods.
    
    # For now, return schoolbook as baseline
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Try random search for better ranks
    for target_rank in range(35, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with small integers/halves
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(target_rank, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return best

# EVOLVE-BLOCK-END

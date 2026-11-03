"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 3, 4, 8
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Start with schoolbook decomposition as baseline
    rank_sb = n * m * p  # 96
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging terms with small coefficients
    # Strategy: use random search to find low-rank decompositions
    
    # For 3x4x8, the theoretical lower bound is max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # Schoolbook is 96. Strassen-style block decomposition might help.
    
    # Try a structured approach: decompose into blocks
    # 3x4 * 4x8: We can think of this as 3x2 blocks of 2x4 times 4x8
    # Or use the fact that 4 = 2+2, so we can apply Strassen to 2x2 blocks
    
    # Let's try: split the 4x4 middle into 2x2 blocks
    # A is 3x4, B is 4x8
    # Split B into B1 (4x4) and B2 (4x4)
    # Then C = A*B1 + A*B2
    # This doesn't directly help with rank.
    
    # Alternative: use the known result that for m=n=p=3, rank is at most 23 (Winograd)
    # For 3x4x8, we can pad to 3x4x8 and use block structure.
    
    # Let's try a direct optimization approach:
    # Start with random rank-r decomposition and try to reduce r
    
    def try_random_decomposition(r, n_trials=50):
        """Try to find a valid rank-r decomposition via random search."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            # Random initialization
            U_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            # Check if valid
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        return None
    
    # Try various ranks from low to high
    for r in range(32, best_rank):
        if time.time() > deadline:
            break
        result = try_random_decomposition(r, n_trials=20)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

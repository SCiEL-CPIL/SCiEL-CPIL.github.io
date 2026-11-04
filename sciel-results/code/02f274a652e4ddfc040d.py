"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _check_exact(U, V, W, n, m, p, timeout=30):
    """Check if the decomposition is exact using random integer tests."""
    start = time.time()
    for trial in range(200):
        if time.time() - start > timeout:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # True product
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Decomposed product
        C_dec = [[0] * p for _ in range(n)]
        for c in range(len(U[0])):
            a_col = [U[i * m + j][c] for i in range(n) for j in range(m)]
            b_col = [V[j * p + k][c] for j in range(m) for k in range(p)]
            w_col = [W[k * n + i][c] for i in range(n) for k in range(p)]
            # Sum over c
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_dec[i][k] += a_col[i * m + j] * b_col[j * p + k] * w_col[k * n + i]
        for i in range(n):
            for k in range(p):
                if C_dec[i][k] != C_true[i][k]:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s out of 120s budget
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # Strategy: Randomized search over ranks below schoolbook
    # We'll try rank values from schoolbook_rank - 1 down to some lower bound
    
    # Known lower bound for M_{2,5,6}: at least 2*5 + 2*6 - 4 = 18? 
    # Actually r(M_{n,m,p}) >= n*m + n*p - 1 for certain cases, but generally r >= max(n*m, m*p, n*p)
    # A simple lower bound is max(n*m, n*p, m*p) = max(10, 12, 30) = 30
    # But we know r(M_{2,2,2})=7 > 8, so it's not simply max
    # For 2x5x6, a reasonable lower bound might be around 30-40
    
    # Let's try a few promising ranks
    # We'll use a random restart strategy: try to build a rank-r decomposition
    
    def try_rank(r):
        """Try to construct a rank-r decomposition via random initialization + optimization."""
        if r >= best_rank:
            return None
        if r < max(n*m, n*p, m*p):  # Basic lower bound
            return None
        
        # We'll use a simple gradient-free approach: random integer/rational entries
        # This is heuristic and may not find exact solutions, but worth trying
        
        max_iter = 5000
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Initialize with random small integers/half-integers
            scale = 1.0
            U = [[random.choice([0, 1, -1, Fraction(1,2), Fraction(-1,2)]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([0, 1, -1, Fraction(1,2), Fraction(-1,2)]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([0, 1, -1, Fraction(1,2), Fraction(-1,2)]) for _ in range(r)] for _ in range(p*n)]
            
            # Check if it's exact
            if _check_exact(U, V, W, n, m, p, timeout=5):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks in decreasing order, but limit total time
    ranks_to_try = []
    for r in range(schoolbook_rank - 1, max(30, schoolbook_rank // 2), -1):
        ranks_to_try.append(r)
        if time.time() > deadline - 20:
            break
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_result = result
            # Found a lower rank, can stop or continue for even lower
            break
    
    return best_result
# EVOLVE-BLOCK-END

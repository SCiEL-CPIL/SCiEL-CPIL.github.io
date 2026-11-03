"""Matrix multiplication 4x5 x 5x5 with rank-minimal exact tensor decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
DEADLINE = 135  # seconds; evaluator kills at 150s

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
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

def _verify_float(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the multiplication tensor using floats.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
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

    # Strategy: Use structured decomposition based on block structure
    # 4x5 x 5x5: Think of A as [A1 A2] where A1 is 4x3, A2 is 4x2
    # B as [[B11][B12]; [B21][B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = [A1 A2] [[B11][B12]; [B21][B22]] = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # For now, let's try random search with small alphabets for various ranks.
    
    def try_random_rank(r, num_trials=1000, alphabet=None):
        """Try random decompositions of rank r."""
        if alphabet is None:
            alphabet = [0, 1, -1, 0.5, -0.5, 2, -2]
        
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            U = [[random.choice(alphabet) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(alphabet) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(alphabet) for _ in range(r)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # Try structured decompositions based on block structure
    # 4x5 x 5x5: Think of A as [A1 A2] where A1 is 4x3, A2 is 4x2
    # B as [[B11][B12]; [B21][B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = [A1 A2] [[B11][B12]; [B21][B22]] = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # Rank contributions:
    # A1*B11: 4*3*3 = 36 (schoolbook) or lower
    # A2*B21: 4*2*3 = 24
    # A1*B12: 4*3*2 = 24
    # A2*B22: 4*2*2 = 16
    # Total schoolbook: 100
    
    # With Strassen-like for 3x3: rank 19 instead of 27 for B11 part
    # But the structure is more complex because it's rectangular.
    
    # Let's try a different approach: use the fact that we need
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 100 equations in r variables per "term"
    # Actually, it's r^3 variables total (r entries in each of U, V, W)
    # with 100 constraints.
    
    # For r < 100, this is an underdetermined system in a sense, but nonlinear.
    
    # Let's try a more targeted search:
    # Fix most entries to 0 or 1, and search for the remaining degrees of freedom.
    
    # Approach: Start from schoolbook and try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if their contributions can be combined.
    
    # Actually, let's try a completely different approach:
    # Use the known rank-19 decomposition for 3x3x3 and build up.
    
    # Lader's rank-19 decomposition for 3x3x3:
    # This is complex to encode from scratch. Let's try a simpler structured approach.
    
    # For 4x5 x 5x5, one approach:
    # Pad to 5x5 x 5x5 (rank 97 known) and then project.
    # But projection may not preserve rank.
    
    # Let's try random search with a deadline guard.
    
    # Try ranks from 95 down to 60
    for r in range(95, 60, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try with small alphabet
        result = try_random_rank(r, num_trials=500, alphabet=[0, 1, -1])
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        # Also try with half-integers
        if time.time() - start_time > DEADLINE:
            break
        result = try_random_rank(r, num_trials=200, alphabet=[0, 1, -1, 0.5, -0.5])
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

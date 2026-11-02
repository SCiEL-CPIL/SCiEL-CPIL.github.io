"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # For each (i,j,k), the coefficient should be 1 if we're computing (A*B)[i,k]
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all valid i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def _try_rank(r, n, m, p, deadline, max_attempts=5000):
    """Try to find a rank-r exact decomposition using random search with integer entries."""
    start = time.time()
    rng = random.Random(42)
    
    # Search space: entries in {-1, 0, 1} or small integers
    # We'll use a greedy/iterative approach
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random matrices with small integer entries
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_search(r, n, m, p, deadline):
    """
    Structured search for rank-r decomposition.
    
    For matrix multiplication, we can exploit structure. 
    The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    We look for decompositions where each rank-1 term has a structured form.
    """
    start = time.time()
    rng = random.Random(123)
    
    # Try various structured patterns
    # Pattern 1: Each term covers a subset of (i,j,k) triples
    # We need to partition the n*m*p triples into r groups, 
    # but each rank-1 term must have the product structure.
    
    # A rank-1 term U_c ⊗ V_c ⊗ W_c contributes to T[i,j,k] 
    # the value U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # For the sum to equal 1 for all (i,j,k), we need careful construction.
    
    # Try using the fact that for n=2, we might be able to use fewer terms
    # by exploiting linearity in i.
    
    # Alternative: use known results. For 2x3 * 3x5, the schoolbook rank is 30.
    # The tensor rank of matrix multiplication is at most n*m*p, but can be lower.
    
    # Let's try a more systematic approach: 
    # Fix U and V, then solve for W.
    
    # Actually, let's try a different approach: use the fact that
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as sum over j, which gives rank m*p*n = 30.
    # But we might do better by grouping.
    
    # For n=2, m=3, p=5:
    # Result has 2*5=10 entries, each is a sum of 3 products.
    # Total "work" is 30 multiplications.
    
    # Can we share computations? In exact tensor decomposition, 
    # each rank-1 term corresponds to one "multiplication" in the algorithm.
    
    # Let's try random search with a wider range of values
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Use values in {-2, -1, 0, 1, 2}
        U = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: leave 10 seconds margin before 120s limit
    deadline = time.time() + 100
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(10, 15, 6) = 15
    # But tensor rank can be higher than these bounds.
    
    for r in range(15, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try random search
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            best = result
            break
        
        # Try structured search
        result = _structured_search(r, n, m, p, deadline)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END

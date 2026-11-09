"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
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
    return U, V, W

def _decomposition_is_valid(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # For each output element (i, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] == delta_{i,k}
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication (A[i,j]*B[j,k])
    # W[k*n+i, c] is the coefficient such that T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually, the standard convention is:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta_{i,k} (since we're decomposing the multiplication map)
    # Rank-r decomposition: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if actual != expected:
                    return False
    return True

def _try_rank_search(n, m, p, max_rank, time_limit=120):
    """
    Search for a rank-r decomposition with r < max_rank.
    Uses the fact that if a rank-r decomposition exists, we can find it by
    solving a system of polynomial equations.
    
    Strategy: Start from schoolbook and try to eliminate terms by finding
    linear dependencies.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # First, verify schoolbook works
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    if not _decomposition_is_valid(U_sb, V_sb, W_sb, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower rank by systematic search
    # For small dimensions, we can try all subsets of the schoolbook terms
    # and check if a linear combination gives a valid decomposition
    
    # The key insight: if we can find coefficients alpha_c such that
    # sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0, then we can eliminate one term
    # by expressing it in terms of others.
    
    # For a rank-r decomposition to exist with r < n*m*p, there must be
    # linear dependencies among the tensor terms.
    
    # Let's try a more direct approach: use the Strassen-like idea
    # of finding a decomposition with rank < n*m*p.
    
    # For 2x3x7, known lower bounds suggest rank might be around 30-35.
    # Let's try to find a valid decomposition by solving the tensor equation.
    
    # Approach: Parameterize U, V, W with r columns and solve for the
    # polynomial constraints T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # This is a system of n*m*p polynomial equations in 3*r variables (per column).
    # For small r, we can try random initializations and Newton's method.
    
    # Alternative: Use the fact that the tensor has a specific structure.
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p
    # and at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    
    # Let's try ranks from 21 upward and see if we can find a valid decomposition.
    
    for r in range(21, best_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try to find a rank-r decomposition
        # We'll use a randomized approach with exact rational arithmetic
        
        success = False
        for trial in range(50):  # Limit trials per rank
            if time.time() - start_time > time_limit - 10:
                break
            
            # Initialize with random rational values in [-1, 1]
            U = [[Fraction(0)] * r for _ in range(n * m)]
            V = [[Fraction(0)] * r for _ in range(m * p)]
            W = [[Fraction(0)] * r for _ in range(p * n)]
            
            # Set some entries to 1 to help with the structure
            # Use the pattern from schoolbook but with fewer columns
            for c in range(r):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
            
            # Now try to adjust to make it work
            # This is a nonlinear system, so let's try a different approach
            
            # Actually, for this specific problem, let's use a known result:
            # The rank of 2x3x7 matrix multiplication is at most 24 (by known bounds).
            # Let's try to find a rank-24 decomposition.
            
            if r == 24:
                # Try a structured decomposition
                # Based on the pattern, we can try to combine terms
                U24 = [[Fraction(0)] * 24 for _ in range(6)]
                V24 = [[Fraction(0)] * 24 for _ in range(21)]
                W24 = [[Fraction(0)] * 24 for _ in range(14)]
                
                # Initialize with schoolbook pattern for first 24 terms
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < 24:
                                U24[i * m + j][col] = 1
                                V24[j * p + k][col] = 1
                                W24[k * n + i][col] = 1
                                col += 1
                
                # Now we need to "fix" the remaining terms by adjusting
                # The missing terms are those with col >= 24
                # We need to express them as linear combinations of existing terms
                
                # This requires solving a system of equations
                # For each missing (i,j,k) triple, we need:
                # sum_{c=0}^{23} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}
                
                # This is complex, so let's try a different strategy
                # Use the fact that for matrix multiplication, we can use
                # block decomposition ideas
                
                pass
        
        # If we found a valid decomposition, update best
        if success:
            best_rank = r
            break
    
    # For now, return the schoolbook decomposition since finding a lower rank
    # requires more sophisticated methods
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a lower-rank decomposition
    try:
        U, V, W, rank = _try_rank_search(n, m, p, n * m * p, time_limit=110)
    except Exception as e:
        print(f"Search failed: {e}")
        U, V, W = _schoolbook_decomposition(n, m, p)
        rank = n * m * p
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

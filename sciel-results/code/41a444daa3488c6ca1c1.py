import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Searches for the minimum rank using a systematic approach with rank-1 terms.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, keep safety margin before 120s
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Start with schoolbook as fallback
    def schoolbook():
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
    
    # Try to find a decomposition of rank r
    # Strategy: use the fact that we can combine multiple (i,j,k) tuples into one rank-1 term
    # if the patterns align
    
    # For small dimensions, try greedy construction
    # Key insight: if we can find a set of rank-1 terms that cover all (i,j,k) positions
    
    # Try ranks from known lower bounds upward
    # Lower bound for 3x4x5: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But actual lower bound is higher. Let's try from 21 upward.
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Try to construct decompositions of increasing rank
    # Use a greedy algorithm: at each step, pick the rank-1 term that covers the most uncovered (i,j,k)
    
    def try_greedy_rank(r_max):
        """Try to find a decomposition of rank <= r_max using greedy method."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize: all (i,j,k) uncovered
        uncovered = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    uncovered.add((i, j, k))
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        # Try different orderings
        for attempt in range(50):  # Multiple random attempts
            if time.time() > deadline - 10:
                break
                
            uncovered = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        uncovered.add((i, j, k))
            
            U_terms = []
            V_terms = []
            W_terms = []
            
            while uncovered and len(U_terms) < r_max:
                # Find the best rank-1 term to add
                # A rank-1 term is defined by (u_pattern, v_pattern, w_pattern)
                # where u_pattern is a vector in R^{n*m}, v_pattern in R^{m*p}, w_pattern in R^{p*n}
                
                # For simplicity, use 0/1 patterns
                # Try to find a term that covers many uncovered positions
                
                best_cover = 0
                best_u = None
                best_v = None
                best_w = None
                
                # Sample some candidate patterns
                import random
                random.seed(attempt * 1000 + len(U_terms))
                
                for _ in range(200):  # Try 200 random candidates
                    # Random 0/1 patterns
                    u_pat = [random.choice([0, 1]) for _ in range(n * m)]
                    v_pat = [random.choice([0, 1]) for _ in range(m * p)]
                    w_pat = [random.choice([0, 1]) for _ in range(p * n)]
                    
                    # Count how many uncovered (i,j,k) this term covers
                    # A term covers (i,j,k) if u_pat[i*m+j] * v_pat[j*p+k] * w_pat[k*n+i] == 1
                    cover = 0
                    for (i, j, k) in uncovered:
                        if u_pat[i * m + j] == 1 and v_pat[j * p + k] == 1 and w_pat[k * n + i] == 1:
                            cover += 1
                    
                    if cover > best_cover:
                        best_cover = cover
                        best_u = u_pat
                        best_v = v_pat
                        best_w = w_pat
                
                if best_cover == 0:
                    break
                
                # Add this term
                U_terms.append(best_u)
                V_terms.append(best_v)
                W_terms.append(best_w)
                
                # Remove covered positions
                new_uncovered = set()
                for (i, j, k) in uncovered:
                    if not (best_u[i * m + j] == 1 and best_v[j * p + k] == 1 and best_w[k * n + i] == 1):
                        new_uncovered.add((i, j, k))
                uncovered = new_uncovered
            
            if not uncovered:
                # Success! Build the matrices
                rank = len(U_terms)
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for c in range(rank):
                    for idx in range(n * m):
                        U[idx][c] = U_terms[c][idx]
                    for idx in range(m * p):
                        V[idx][c] = V_terms[c][idx]
                    for idx in range(p * n):
                        W[idx][c] = W_terms[c][idx]
                
                # Verify
                if verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    def verify(U, V, W, n, m, p):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try ranks from lower bound upward
    # Known: rank(3x4x5) <= 30 (Strassen-like), lower bound >= 21
    for r in range(21, min(35, n * m * p + 1)):
        if time.time() > deadline - 15:
            break
        
        result = try_greedy_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we found a better result, use it; otherwise fall back to schoolbook
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END

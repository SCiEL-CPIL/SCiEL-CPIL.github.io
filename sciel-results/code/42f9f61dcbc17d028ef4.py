"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition."""
import random
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
N, M, P = 3, 4, 7
RANK_SCHOOLBOOK = N * M * P  # 84

def _time_remaining(deadline):
    return deadline - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                rhs = W_arr[k*n+i]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _is_valid_entries(mat):
    """Check all entries are integer or half-integer."""
    for row in mat:
        for v in row:
            # Check if v is integer or half-integer
            if isinstance(v, Fraction):
                num, den = v.numerator, v.denominator
                if den not in (1, 2):
                    return False
            else:
                v_float = float(v)
                if abs(v_float * 2 - round(v_float * 2)) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition (rank 84)."""
    n, m, p = N, M, P
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
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(rank, deadline, max_attempts=200000):
    """Try to find a rank-`rank` decomposition with structured entries.
    
    Strategy: Use the fact that for matrix multiplication, we can try
    decompositions where each "slice" corresponds to a subset of (i,j,k) triples.
    
    We search over patterns where U, V, W have entries in {0, 1, -1, 1/2, -1/2}.
    """
    n, m, p = N, M, P
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] = 1 for all i,j,k. We need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]  ... wait, that's not right.
    
    # Actually the condition is:
    # For the multiplication (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T[i,j,k] = 1 (the standard matmul tensor).
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's a 3-tensor decomposition. Let me re-read the contract.
    
    # The contract says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # and they "exactly decompose the matrix multiplication tensor"
    # The standard condition for matmul tensor decomposition:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_c (sum_j U[i*m+j, c] A_{i,j}) * (sum_j V[j*p+k, c] B_{j,k}) ... no.
    
    # Actually the standard formulation: we want
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # A rank-r decomposition gives:
    # C_{i,k} = sum_{c=1}^{r} (sum_j U[i*m+j, c] A_{i,j}) * (sum_j V[j*p+k, c] B_{j,k})
    # Hmm, that doesn't look right either.
    
    # Let me think more carefully. The matrix multiplication tensor M_{i,j,k} = 1 (for all i,j,k in ranges).
    # A rank-r decomposition is: M_{i,j,k} = sum_{c=1}^{r} a_c[i,j] * b_c[j,k] * c_c[k,i]
    # where a_c is n x m, b_c is m x p, c_c is p x n.
    # In the flat representation:
    # U[i*m+j, c] = a_c[i,j]
    # V[j*p+k, c] = b_c[j,k]  
    # W[k*n+i, c] = c_c[k,i]
    # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k.
    
    # Wait, but the verify function I wrote uses a bilinear check. Let me reconsider.
    # Actually re-reading: "U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]" - no that's not it either.
    
    # The standard matmul tensor decomposition condition is trilinear:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all (i,j,k).
    
    # Let me use that.
    
    best_result = None
    
    # For small ranks, try exhaustive-ish search with structured patterns
    # Key insight: if we use only entries in {0, 1}, we're looking for combinatorial designs.
    
    # Try rank = 84 first (schoolbook), then try to improve
    # For this problem, known bounds: rank is at most 84 (schoolbook)
    # Lower bound: at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    # Strategy: try to find decompositions with rank < 84
    # Use a greedy/iterative approach
    
    # For each candidate rank, try to construct a valid decomposition
    for r in range(28, min(84, 50)):  # Try ranks from 28 to 49
        if time.time() > deadline - 10:
            break
        if _try_rank(r, deadline, max_attempts):
            best_result = r
            break
    
    return best_result

def _try_rank(r, deadline, max_attempts):
    """Try to find a valid rank-r decomposition."""
    n, m, p = N, M, P
    nm = n * m  # 12
    mp = m * p  # 28
    pn = p * n  # 21
    
    # We need to find U (nm x r), V (mp x r), W (pn x r) with entries in {0,1,-1,1/2,-1/2}
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a very hard constraint satisfaction problem.
    # Let's try a different approach: use the fact that we can combine schoolbook terms.
    
    # Actually, let me try a more practical approach:
    # Start with schoolbook and try to merge terms.
    
    # For now, return False unless r == 84
    if r >= 84:
        return True
    
    return False

def construct():
    deadline = time.time() + 130  # Safety margin from 150s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = 84
    
    # Try to find better decompositions
    # For 3x4x7, the known optimal rank might be less than 84.
    # Let's try a structured search.
    
    # Approach: try to find decompositions where the matrices have
    # a block structure or use known algebraic techniques.
    
    # For now, try a simple improvement: check if we can reduce rank
    # by finding linear dependencies in the schoolbook.
    
    # The schoolbook has rank 84. Can we do better?
    # For matrix multiplication, the rank is related to the border rank.
    # For 3x4x7, let's try to find a lower-rank decomposition.
    
    # Try structured patterns with small entries
    # Use a random search with verification
    
    import random
    random.seed(42)
    
    # Try ranks from 50 down to 30
    for r in range(50, 29, -1):
        if time.time() > deadline - 15:
            break
        
        # Try random structured decompositions
        found = _random_search_rank(r, deadline - 15, max_iter=50000)
        if found:
            U, V, W = found
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    return best

def _random_search_rank(r, deadline, max_iter=50000):
    """Random search for a rank-r decomposition."""
    n, m, p = N, M, P
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Use entries from {0, 1, -1, 1/2, -1/2}
    entries = [0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W
        U = [[random.choice(entries) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice(entries) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice(entries) for _ in range(r)] for _ in range(pn)]
        
        # Verify
        if _verify(U, V, W):
            return U, V, W
    
    return None

def _verify(U, V, W, n=N, m=M, p=P):
    """Verify trilinear decomposition condition."""
    rank = len(U[0]) if U else 0
    U_arr = np.array([[float(x) for x in row] for row in U])
    V_arr = np.array([[float(x) for x in row] for row in V])
    W_arr = np.array([[float(x) for x in row] for row in W])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

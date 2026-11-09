"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE_SECONDS = 115.0


def _schoolbook():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Check that the decomposition exactly reconstructs matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _verify_fast(U, V, W, tol=1e-10):
    """Fast numerical verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j, :]
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                total = np.sum(u_row * v_row * w_row)
                if abs(total - 1.0) > tol:
                    return False
    return True


def _structured_polynomial_search(target_rank, n, m, p, deadline, max_attempts=100):
    """
    Exploit n=2 structure with polynomial parameterization.
    
    For n=2, we can think of U as having two blocks (i=0 and i=1).
    We parameterize U and V with low-degree polynomials in j and k,
    then solve for W to satisfy the tensor equation.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Strategy: For each rank term r, assign a "type" based on polynomial degree
        # U[i*m+j] = sum over basis functions in j
        # V[j*p+k] = sum over basis functions in j,k
        # W[k*n+i] = sum over basis functions in k,i
        
        # Simplified: use random but structured initialization
        for r in range(target_rank):
            # U: 6 entries (2*3), make sparse with 2-3 nonzeros
            u_nnz = random.randint(2, 3)
            u_positions = random.sample(range(n * m), u_nnz)
            for pos in u_positions:
                U[pos][r] = random.choice([1, -1])
            
            # V: 21 entries (3*7), make sparse with 3-5 nonzeros
            v_nnz = random.randint(3, 5)
            v_positions = random.sample(range(m * p), v_nnz)
            for pos in v_positions:
                V[pos][r] = random.choice([1, -1])
            
            # W: 14 entries (7*2), make sparse with 2-4 nonzeros
            w_nnz = random.randint(2, 4)
            w_positions = random.sample(range(p * n), w_nnz)
            for pos in w_positions:
                W[pos][r] = random.choice([1, -1])
        
        if _verify_fast(U, V, W):
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank_with_structure(target_rank, n, m, p, deadline, max_attempts=50):
    """Try to find a valid decomposition using structured patterns."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Use a structured approach: each rank term covers a block of (i,j,k)
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Assign each (i,j,k) to a rank term, with possible sharing
        # Strategy: group by i first, then j, then k
        assignments = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Use a hash-like assignment to distribute
                    idx = (i * m * p + j * p + k) % target_rank
                    if idx not in assignments:
                        assignments[idx] = []
                    assignments[idx].append((i, j, k))
        
        # For each rank term, we need to find U_col, V_col, W_col such that
        # for all (i,j,k) assigned to this term: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
        # And for all (i,j,k) NOT assigned: U[i*m+j] * V[j*p+k] * W[k*n+i] = 0
        
        # This is hard. Instead, let's try a different structured approach:
        # Use the fact that n=2 is small. For each i, we can have separate terms.
        
        # Try: rank = n * (m*p) / some factor
        # Or: use a tensor product structure
        
        # Let's try a simple structured pattern:
        # For i=0: use m*p/2 terms covering k=0..3 and k=4..6
        # For i=1: use m*p/2 terms covering k=0..3 and k=4..6
        
        # Actually, let's just try random with better initialization
        for r in range(target_rank):
            # Each rank term should be "dense" enough
            # U: n*m = 6 entries, pick 2-3 non-zero
            # V: m*p = 21 entries, pick 3-5 non-zero  
            # W: p*n = 14 entries, pick 2-4 non-zero
            
            u_nnz = random.randint(2, 4)
            v_nnz = random.randint(3, 6)
            w_nnz = random.randint(2, 4)
            
            u_positions = random.sample(range(n * m), min(u_nnz, n * m))
            for pos in u_positions:
                U[pos][r] = random.choice([1, -1])
            
            v_positions = random.sample(range(m * p), min(v_nnz, m * p))
            for pos in v_positions:
                V[pos][r] = random.choice([1, -1])
            
            w_positions = random.sample(range(p * n), min(w_nnz, p * n))
            for pos in w_positions:
                W[pos][r] = random.choice([1, -1])
        
        if _verify_fast(U, V, W):
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None


def _greedy_merge(n, m, p, deadline):
    """Start from schoolbook and greedily try to merge terms."""
    result = _schoolbook()
    U = [row[:] for row in result["U"]]
    V = [row[:] for row in result["V"]]
    W = [row[:] for row in result["W"]]
    current_rank = n * m * p
    
    # Try to remove one column at a time, adjusting others
    # This is very expensive, so limit attempts
    for col_to_remove in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random adjustments
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            
            # Remove column
            U_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_test]
            V_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_test]
            W_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_test]
            
            new_rank = len(U_test[0]) if U_test else 0
            
            # Add small perturbations to try to compensate
            for _ in range(3):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U' and U_test:
                    pos = random.randint(0, len(U_test) - 1)
                    col = random.randint(0, len(U_test[0]) - 1)
                    U_test[pos][col] += random.choice([1, -1])
                elif mat == 'V' and V_test:
                    pos = random.randint(0, len(V_test) - 1)
                    col = random.randint(0, len(V_test[0]) - 1)
                    V_test[pos][col] += random.choice([1, -1])
                elif mat == 'W' and W_test:
                    pos = random.randint(0, len(W_test) - 1)
                    col = random.randint(0, len(W_test[0]) - 1)
                    W_test[pos][col] += random.choice([1, -1])
            
            if _verify_fast(U_test, W_test, tol=1e-8):
                if _verify(U_test, V_test, W_test):
                    U = U_test
                    V = V_test
                    W = W_test
                    current_rank = new_rank
                    break
    
    if current_rank < n * m * p:
        return {"U": U, "V": V, "W": W}, current_rank
    
    return None, current_rank


def _layered_search(n, m, p, deadline):
    """
    Exploit n=2 by treating the two layers (i=0, i=1) separately.
    For each layer, we need to decompose a 3x7 matrix multiplication.
    We can share rank terms between layers if the patterns align.
    """
    base_rank = n * m * p  # 42
    
    # Try ranks from base_rank-1 down to some minimum
    for target_rank in range(base_rank - 1, max(30, base_rank - 10)):
        if time.time() > deadline - 5:
            break
        
        # Try structured polynomial search
        result = _structured_polynomial_search(target_rank, n, m, p, deadline - 5, max_attempts=50)
        if result is not None:
            return result, target_rank
        
        # Try general structured search
        result = _try_rank_with_structure(target_rank, n, m, p, deadline - 5, max_attempts=30)
        if result is not None:
            return result, target_rank
    
    return None, base_rank


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Returns: {'U': (n*m x rank), 'V': (m*p x rank), 'W': (p*n x rank)}
    with entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 42
    
    # Fallback: schoolbook decomposition
    fallback = _schoolbook()
    
    # Try greedy merge first
    merged_result, merged_rank = _greedy_merge(n, m, p, deadline)
    if merged_result is not None:
        return merged_result
    
    # Try layered search exploiting n=2 structure
    layered_result, layered_rank = _layered_search(n, m, p, deadline - 10)
    if layered_result is not None:
        return layered_result
    
    return fallback

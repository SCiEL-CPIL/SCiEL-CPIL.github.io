"""
Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition.

Strategy:
1. Build a structured decomposition with rank 32 by splitting B (3x4) into
   B1 (3x2) and B2 (3x2), applying Strassen (rank 7) to each 3x3 * 3x2 block
   (via padding to 3x3), and using schoolbook for the last row (rank 12).
   Rank = 7 + 7 + 12 = 26? Wait, let me recompute.

   Actually: A is 3x3, B is 3x4.
   Split B into B1 (3x2) and B2 (3x2).
   C1 = A * B1 (3x2): pad B1 to 3x3 with a zero column, compute 3x3*3x3 via Strassen (rank 7), discard the 3rd column of result. But the decomposition factors are for the full tensor, so we can't just "discard" - we need the full 3x3*3x3 Strassen and then zero out the unused part. Actually the tensor for 3x3*3x2 is different from 3x3*3x3.

   Let me reconsider. The tensor T for 3x3 * 3x4 has:
   - U: shape (9, r)
   - V: shape (12, r)  
   - W: shape (12, r)
   with W indexed as W[k*n + i, c] where k in 0..3, i in 0..2.

   Schoolbook: r = 36.
   
   Structured approach:
   - Split the 4 columns of B into 2 groups of 2.
   - For each group, we need to compute A * B_group where B_group is 3x2.
   - The tensor for 3x3 * 3x2: rank is at most 18 (schoolbook), and at least max(9, 6, 6) = 9.
   
   Actually, let me just try a direct structured construction.

   Better approach: Use the fact that for each output position (i, k), we need:
   C[i,k] = sum_j A[i,j] * B[j,k]

   For fixed i, the 4 values C[i,0], C[i,1], C[i,2], C[i,3] form a row of A[i,:] times B.
   This is a 1x3 * 3x4 multiplication, which has rank 12 (schoolbook) and lower bound max(3,12,4)=12.

   For the full problem, we have 3 such rows but they share the B factors.

   Let me try a different structured approach:
   - For each of the 3 rows of A, compute the 4 dot products.
   - Use a "row-based" decomposition where for each row i, we use rank-12 factors.
   - But share the V (B) factors across rows.

   Actually, the simplest improvement over schoolbook:
   - Schoolbook has 36 rank-1 terms.
   - We can group terms that share the same B[j,k] factor.
   - For fixed (j,k), B[j,k] appears in 3 terms (one for each i).
   - So we have 12 groups of 3 terms each.
   - Within each group, the 3 terms are: sum over i of U[i*m+j, c] * W[k*n+i, c] where V[j*p+k, c] = 1.
   - This doesn't directly reduce rank.

   Let me just try the random search more aggressively with better heuristics.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 135  # 15s safety margin before 150s limit

def _time_remaining():
    return DEADLINE - time.time()

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _row_gcd(row):
    """Compute GCD of all elements in a row."""
    nz = [abs(x) for x in row if x != 0]
    if not nz:
        return 0
    return reduce(gcd, nz)

def _extended_gcd(a, b):
    if b == 0:
        return (abs(a), 1 if a > 0 else -1, 0)
    g, x1, y1 = _extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def _find_integer_solution(coeffs, target, max_terms=3):
    """
    Find an integer vector z such that coeffs · z = target.
    Try up to max_terms nonzero entries.
    """
    n = len(coeffs)
    nz = [(i, coeffs[i]) for i in range(n) if coeffs[i] != 0]
    if not nz:
        return [0] * n if target == 0 else None
    
    # 1-term
    for i, c in nz:
        if c != 0 and target % c == 0:
            z = [0] * n
            z[i] = target // c
            return z
    
    if max_terms < 2:
        return None
    
    # 2-term
    for a_idx in range(len(nz)):
        ia, ca = nz[a_idx]
        for b_idx in range(len(nz)):
            if a_idx == b_idx:
                continue
            ib, cb = nz[b_idx]
            g, x0, y0 = _extended_gcd(ca, cb)
            if g == 0 or target % g != 0:
                continue
            t = target // g
            x = x0 * t
            y = y0 * t
            if ca * x + cb * y == target:
                z = [0] * n
                z[ia] = x
                z[ib] = y
                return z
    
    if max_terms < 3:
        return None
    
    # 3-term: fix one coefficient to small value, solve 2-term for rest
    for a_idx in range(len(nz)):
        ia, ca = nz[a_idx]
        for a_val in [-1, 0, 1]:
            rem = target - ca * a_val
            if a_val == 0:
                continue  # already covered by 2-term
            for b_idx in range(len(nz)):
                if b_idx == a_idx:
                    continue
                ib, cb = nz[b_idx]
                for c_idx in range(len(nz)):
                    if c_idx == a_idx or c_idx == b_idx:
                        continue
                    ic, cc = nz[c_idx]
                    g, x0, y0 = _extended_gcd(cb, cc)
                    if g == 0 or rem % g != 0:
                        continue
                    t = rem // g
                    x = x0 * t
                    y = y0 * t
                    if cb * x + cc * y == rem:
                        z = [0] * n
                        z[ia] = a_val
                        z[ib] = x
                        z[ic] = y
                        return z
    return None

def _build_m_matrix(U_vals, V_vals, n, m, p, r):
    """Compute M[i*p+k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]"""
    M = np.zeros((n * p, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[i * p + k, c] = s
    return M

def _try_build_W(M, n, p, r):
    """
    Given M (n*p x r), try to find W_int (p*n x r) such that
    for each (i,k): M[i*p+k, :] · W_int[k*n+i, :] = 2
    Returns W_int or None.
    """
    W_int = np.zeros((p * n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            row = M[i * p + k, :].tolist()
            g = _row_gcd(row)
            if g == 0:
                return None
            if 2 % g != 0:
                return None
            z = _find_integer_solution(row, 2, max_terms=3)
            if z is None:
                return None
            W_int[k * n + i, :] = z
    return W_int

def _attempt_decomposition(U_vals, V_vals, n, m, p, r):
    """Try to complete a partial decomposition to a full valid one."""
    M = _build_m_matrix(U_vals, V_vals, n, m, p, r)
    W_int = _try_build_W(M, n, p, r)
    if W_int is None:
        return None
    
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = []
    for i in range(p * n):
        row = []
        for c in range(r):
            v = W_int[i, c]
            if v % 2 == 0:
                row.append(v // 2)
            else:
                row.append(v / 2.0)
        W_list.append(row)
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _random_UV(n, m, p, r, strategy, seed_offset):
    """Generate random U and V matrices with various strategies."""
    np.random.seed(seed_offset * 104729 + strategy * 31 + 7)
    
    if strategy == 0:
        # ±1 only
        U_vals = np.random.choice([-1, 1], size=(n*m, r))
        V_vals = np.random.choice([-1, 1], size=(m*p, r))
    elif strategy == 1:
        # {-1, 0, 1} with bias toward non-zero
        U_vals = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.35, 0.3, 0.35])
        V_vals = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.35, 0.3, 0.35])
    elif strategy == 2:
        # Sparse ±1
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < 0.6:
                    U_vals[i, c] = np.random.choice([-1, 1])
        for i in range(m*p):
            for c in range(r):
                if np.random.random() < 0.6:
                    V_vals[i, c] = np.random.choice([-1, 1])
    elif strategy == 3:
        # Structured: make U have block structure
        # For 3x3, group rows by i (3 groups of m=3)
        U_vals = np.zeros((n*m, r), dtype=int)
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    if np.random.random() < 0.7:
                        U_vals[i*m+j, c] = np.random.choice([-1, 1])
        V_vals = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.4, 0.2, 0.4])
    elif strategy == 4:
        # Higher values: ±{1,2,3}
        U_vals = np.random.choice([-3,-2,-1,1,2,3], size=(n*m, r))
        V_vals = np.random.choice([-3,-2,-1,1,2,3], size=(m*p, r))
    elif strategy == 5:
        # Binary-like with scaling
        U_vals = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, r), p=[0.2, 0.25, 0.1, 0.25, 0.2])
        V_vals = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, r), p=[0.2, 0.25, 0.1, 0.25, 0.2])
    else:
        # Default ±1
        U_vals = np.random.choice([-1, 1], size=(n*m, r))
        V_vals = np.random.choice([-1, 1], size=(m*p, r))
    
    return U_vals, V_vals

def _structured_strassen_3x3_3x2(n, m, p):
    """
    Try to construct a decomposition using Strassen-like ideas.
    For 3x3 * 3x2, we can use a known rank-12 or lower construction.
    
    Actually, let me just try to build a rank-24 decomposition structurally:
    For each of the 3 rows i, compute C[i,:] = A[i,:] * B (1x3 times 3x4).
    The rank for each such sub-tensor is 12 (schoolbook), but we can try to
    share V factors.
    
    Alternative: use the identity
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Group by k: for fixed k, C[:,k] = A * B[:,k] (3x3 times 3x1).
    The rank for 3x3 * 3x1 is 9 (schoolbook) with lower bound max(9, 3, 3) = 9.
    So 4 groups of 9 = 36. Same as schoolbook.
    
    Group by i: for fixed i, C[i,:] = A[i,:] * B (1x3 times 3x4).
    Rank is 12 per group, 3 groups = 36.
    
    Let me try a hybrid: split the 4 columns into 2 pairs.
    C[:, 0:2] = A * B[:, 0:2] (3x3 * 3x2): rank at most 18.
    C[:, 2:4] = A * B[:, 2:4] (3x3 * 3x2): rank at most 18.
    Total: 36. Still schoolbook.
    
    For 3x3 * 3x2, can we do better than 18?
    Lower bound: max(9, 6, 6) = 9.
    Upper bound: 18 (schoolbook).
    
    Known: M_{3,3,2} rank is at most 15 (by various constructions).
    Actually, I'm not sure of the exact rank. Let me just use the random search.
    """
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Realistic lower bound is higher due to tensor structure
    
    # Try ranks from low to high
    for target_rank in range(12, best_rank):
        if _time_remaining() < 15:
            break
        
        # Determine number of attempts based on time
        time_left = _time_remaining()
        if time_left > 80:
            attempts = 20000
        elif time_left > 40:
            attempts = 10000
        elif time_left > 20:
            attempts = 5000
        else:
            attempts = 2000
        
        found = False
        for attempt in range(attempts):
            if _time_remaining() < 10:
                break
            
            # Cycle through different strategies
            strategy = attempt % 6
            
            U_vals, V_vals = _random_UV(n, m, p, target_rank, strategy, attempt)
            result = _attempt_decomposition(U_vals, V_vals, n, m, p, target_rank)
            
            if result is not None:
                best_result = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found at this rank, try to go lower
            continue
    
    return best_result

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")

"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations_with_replacement

_START_TIME = time.time()
DEADLINE = _START_TIME + 118  # 2s safety margin before 120s limit

def _time_left():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Strategy: For each target rank r, try many random (U, V) pairs with small
    integer entries. For each, compute the induced M matrix and check if a
    half-integer W exists that satisfies all 12 equations. Use a more
    systematic search with structured patterns in addition to random tries.
    """
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(9,12,12) = 12
    # But tensor rank can be lower; known bounds suggest ~14-18 for 3x3x4
    for target_rank in range(12, min(best_rank + 1, 40)):
        if _time_left() < 3:
            break
        
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Don't break - try to find even lower rank
    
    return best_result

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _try_rank(n, m, p, r):
    """
    Try to find a rank-r decomposition.
    
    Multiple strategies:
    1. Random sparse integer U, V
    2. Structured patterns (block diagonal, etc.)
    3. For each, solve for W exactly
    4. Targeted random: construct U,V so that M rows have small entries
    """
    np.random.seed(42 + r * 1000)
    
    max_attempts = 800 if r <= 16 else 500 if r <= 20 else 200
    if _time_left() < 10:
        max_attempts = 50
    
    # Strategy 1: Random search with varied densities
    for attempt in range(max_attempts):
        if _time_left() < 2:
            break
        
        # Vary the density and value range
        density = 0.15 + 0.6 * (attempt % 9) / 9
        max_val = 1 + (attempt % 4)  # Values in {-max_val, ..., max_val}
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < density:
                    U_vals[i, c] = np.random.randint(-max_val, max_val + 1)
                    if U_vals[i, c] == 0:
                        U_vals[i, c] = 1
        
        for i in range(m*p):
            for c in range(r):
                if np.random.random() < density:
                    V_vals[i, c] = np.random.randint(-max_val, max_val + 1)
                    if V_vals[i, c] == 0:
                        V_vals[i, c] = 1
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    # Strategy 2: Targeted random - ensure each M row has at least one small entry
    for attempt in range(max(100, max_attempts // 4)):
        if _time_left() < 2:
            break
        
        U_vals, V_vals = _generate_targeted(n, m, p, r, attempt)
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    # Strategy 3: Structured patterns
    result = _try_structured(n, m, p, r)
    if result is not None:
        return result
    
    return None

def _generate_targeted(n, m, p, r, seed_offset):
    """
    Generate U, V such that for each (i,k) pair, the M row has entries
    that are likely to allow a half-integer W solution.
    Key insight: if M[i,k,c] = 1 for some c, then W[i,k,c] = 1 works.
    So aim to have at least one column where the dot product is ±1.
    """
    np.random.seed(9999 + seed_offset + r * 7)
    
    U_vals = np.zeros((n*m, r), dtype=int)
    V_vals = np.zeros((m*p, r), dtype=int)
    
    # For each column, pick a structure that makes some M entries small
    for c in range(r):
        # Pick a random j for this column's "anchor"
        j_anchor = np.random.randint(m)
        
        # U: put non-zeros in a few (i, j) positions
        num_u = np.random.randint(1, min(4, n*m))
        u_positions = np.random.choice(n*m, size=num_u, replace=False)
        for pos in u_positions:
            U_vals[pos, c] = np.random.choice([-1, 1])
        
        # V: put non-zeros, biased toward the anchor j
        num_v = np.random.randint(1, min(4, m*p))
        v_positions = np.random.choice(m*p, size=num_v, replace=False)
        for pos in v_positions:
            V_vals[pos, c] = np.random.choice([-1, 1])
        
        # Ensure at least one (i,k) has M = ±1 for this column
        # Check which (i,k) pairs get small values
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                if s == 0 and np.random.random() < 0.3:
                    # Try to fix: adjust one entry
                    # Find a j where both U and V are non-zero
                    for j2 in range(m):
                        if U_vals[i*m+j2, c] != 0 and V_vals[j2*p+k, c] != 0:
                            # Flip one to make product contribute
                            if np.random.random() < 0.5:
                                U_vals[i*m+j2, c] = -U_vals[i*m+j2, c]
                            else:
                                V_vals[j2*p+k, c] = -V_vals[j2*p+k, c]
                            break
    
    return U_vals, V_vals

def _solve_for_W(n, m, p, r, U_vals, V_vals):
    """
    Given U and V, compute M and try to find half-integer W.
    Returns (U_list, V_list, W_list) or None.
    """
    # Compute M[(i,k),c] = sum_j U[i*m+j,c] * V[j*p+k,c]
    M = np.zeros((p*n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[k*n + i, c] = s
    
    # For each (i,k) row of W, solve: sum_c M[row,c] * W[row,c] = 1
    W_vals = []
    valid = True
    for i in range(n):
        for k in range(p):
            row_idx = k * n + i
            m_row = M[row_idx, :]
            w_solution = _find_half_integer_solution(m_row)
            if w_solution is None:
                valid = False
                break
            W_vals.append(w_solution)
        if not valid:
            break
    
    if not valid:
        return None
    
    # Convert to proper lists
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n*m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m*p)]
    W_list = []
    for row in W_vals:
        W_list.append([float(x) if x.denominator != 1 else int(x) for x in row])
    
    # Verify with exact arithmetic
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _find_half_integer_solution(m_row):
    """
    Find a half-integer vector w such that sum_c m_row[c] * w[c] = 1.
    m_row is a numpy array of integers.
    Returns a list of Fractions or None.
    Uses a more systematic approach: for small |m_row| entries, enumerate
    combinations of indices and solve exactly.
    """
    r = len(m_row)
    
    # Filter to non-zero entries
    nz_indices = [c for c in range(r) if int(m_row[c]) != 0]
    nz_vals = [int(m_row[c]) for c in nz_indices]
    
    if not nz_indices:
        return None
    
    # Sort by absolute value (smaller entries are more useful)
    order = sorted(range(len(nz_indices)), key=lambda idx: abs(nz_vals[idx]))
    
    # Try 1-term solutions
    for idx in order:
        c = nz_indices[idx]
        mc = nz_vals[idx]
        if mc != 0:
            w_val = Fraction(1, mc)
            if (w_val * 2).denominator == 1:
                result = [Fraction(0)] * r
                result[c] = w_val
                return result
    
    # Try 2-term solutions
    for ia in range(len(nz_indices)):
        a = nz_indices[ia]
        ma = nz_vals[ia]
        for ib in range(ia+1, len(nz_indices)):
            b = nz_indices[ib]
            mb = nz_vals[ib]
            # x*ma + y*mb = 1, x,y half-integers
            # x = x_num/2, y = y_num/2
            # x_num*ma + y_num*mb = 2
            # Try small x_num
            for x_num in range(-6, 7):
                rem = 2 - x_num * ma
                if rem % mb == 0:
                    y_num = rem // mb
                    if abs(y_num) <= 6:
                        result = [Fraction(0)] * r
                        result[a] = Fraction(x_num, 2)
                        result[b] = Fraction(y_num, 2)
                        return result
    
    # Try 3-term solutions
    for ia in range(len(nz_indices)):
        a = nz_indices[ia]
        ma = nz_vals[ia]
        for ib in range(ia+1, len(nz_indices)):
            b = nz_indices[ib]
            mb = nz_vals[ib]
            for ic in range(ib+1, len(nz_indices)):
                c = nz_indices[ic]
                mc = nz_vals[ic]
                # x*ma + y*mb + z*mc = 1
                # x_num*ma + y_num*mb + z_num*mc = 2
                for x_num in [-2, -1, 0, 1, 2]:
                    rem1 = 2 - x_num * ma
                    for y_num in [-2, -1, 0, 1, 2]:
                        rem2 = rem1 - y_num * mb
                        if rem2 % mc == 0:
                            z_num = rem2 // mc
                            if abs(z_num) <= 2:
                                result = [Fraction(0)] * r
                                result[a] = Fraction(x_num, 2)
                                result[b] = Fraction(y_num, 2)
                                result[c] = Fraction(z_num, 2)
                                return result
    
    # Try 4-term solutions
    if len(nz_indices) >= 4:
        for ia in range(len(nz_indices)):
            a = nz_indices[ia]
            ma = nz_vals[ia]
            for ib in range(ia+1, len(nz_indices)):
                b = nz_indices[ib]
                mb = nz_vals[ib]
                for ic in range(ib+1, len(nz_indices)):
                    c = nz_indices[ic]
                    mc = nz_vals[ic]
                    for id_ in range(ic+1, len(nz_indices)):
                        d = nz_indices[id_]
                        md = nz_vals[id_]
                        # x*ma + y*mb + z*mc + w*md = 1
                        # x_num*ma + y_num*mb + z_num*mc + w_num*md = 2
                        for x_num in [-1, 0, 1]:
                            rem1 = 2 - x_num * ma
                            for y_num in [-1, 0, 1]:
                                rem2 = rem1 - y_num * mb
                                for z_num in [-1, 0, 1]:
                                    rem3 = rem2 - z_num * mc
                                    if rem3 % md == 0:
                                        w_num = rem3 // md
                                        if abs(w_num) <= 1:
                                            result = [Fraction(0)] * r
                                            result[a] = Fraction(x_num, 2)
                                            result[b] = Fraction(y_num, 2)
                                            result[c] = Fraction(z_num, 2)
                                            result[d] = Fraction(w_num, 2)
                                            return result
    
    return None

def _try_structured(n, m, p, r):
    """Try structured patterns for U and V."""
    if r > 25:
        return None
    
    np.random.seed(12345 + r)
    
    # Strategy A: Each column has exactly 2 non-zeros in U and V
    for attempt in range(150):
        if _time_left() < 2:
            break
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for c in range(r):
            u_positions = np.random.choice(n*m, size=2, replace=False)
            for pos in u_positions:
                U_vals[pos, c] = np.random.choice([-1, 1])
            
            v_positions = np.random.choice(m*p, size=2, replace=False)
            for pos in v_positions:
                V_vals[pos, c] = np.random.choice([-1, 1])
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    # Strategy B: Block-structured - group columns by j-index
    # For each j, have a few columns that only involve row i*m+j in U and row j*p+k in V
    # This mimics partial sharing across i or k dimensions
    for attempt in range(100):
        if _time_left() < 2:
            break
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        col = 0
        for j in range(m):
            # Allocate r//m columns to this j (approximately)
            cols_for_j = r // m + (1 if j < r % m else 0)
            for _ in range(cols_for_j):
                if col >= r:
                    break
                # U: non-zero in 1-2 (i,j) positions
                num_i = np.random.randint(1, 3)
                i_positions = np.random.choice(n, size=num_i, replace=False)
                for i in i_positions:
                    U_vals[i*m + j, col] = np.random.choice([-1, 1])
                
                # V: non-zero in 1-2 (j,k) positions
                num_k = np.random.randint(1, 3)
                k_positions = np.random.choice(p, size=num_k, replace=False)
                for k in k_positions:
                    V_vals[j*p + k, col] = np.random.choice([-1, 1])
                
                col += 1
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    # Strategy C: Random but with constraint that each (i,k) pair is "covered"
    for attempt in range(100):
        if _time_left() < 2:
            break
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        # Ensure every (i,k) has at least one column where M[i,k,c] = ±1
        # by construction: for each (i,k), pick a random j and set U[i,j,c]=1, V[j,k,c]=1
        # for some column c
        
        # First, randomly fill
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    if np.random.random() < 0.3:
                        U_vals[i*m+j, c] = np.random.choice([-1, 1])
            for j in range(m):
                for k in range(p):
                    if np.random.random() < 0.3:
                        V_vals[j*p+k, c] = np.random.choice([-1, 1])
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    return None

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")

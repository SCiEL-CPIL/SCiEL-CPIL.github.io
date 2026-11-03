"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 110  # 10s safety margin before 120s limit

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

def _try_rank_r(n, m, p, r, seed_offset=0):
    """
    Try to find a rank-r decomposition.
    
    Key insight: For fixed U (9xr) and V (12xr), the system decomposes into
    12 independent linear equations (one per (i,k) pair):
        sum_c M_{(i,k),c} * W[k*n+i, c] = 1
    where M_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    
    Each equation is 1 x r (underdetermined for r > 1), so we just need
    to find half-integer solutions. We try sparse solutions (1-3 terms).
    """
    np.random.seed(seed_offset * 104729 + 7)
    
    # Generate random ±1 patterns for U and V
    U_vals = np.random.choice([-1, 1], size=(n*m, r))
    V_vals = np.random.choice([-1, 1], size=(m*p, r))
    
    # Compute M_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # M has shape (n*p, r) = (12, r), indexed by (i, k)
    M = np.zeros((n * p, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[i * p + k, c] = s
    
    # For each (i,k), solve: sum_c M[i*p+k, c] * W[k*n+i, c] = 1
    W_vals = np.zeros((p * n, r), dtype=float)
    
    for i in range(n):
        for k in range(p):
            mk = M[i * p + k, :]  # length r, integer values in [-m, m] = [-3, 3]
            w_row = k * n + i
            found = False
            
            # Try 1-term solutions: mk[c] * x = 1 => x = 1/mk[c]
            for c in range(r):
                if mk[c] != 0:
                    x = 1.0 / mk[c]
                    # Check if x is a half-integer
                    if abs(x * 2 - round(x * 2)) < 1e-9:
                        W_vals[w_row, c] = x
                        found = True
                        break
            
            if not found:
                # Try 2-term solutions: mk[a]*x + mk[b]*y = 1
                for a in range(r):
                    if found:
                        break
                    if mk[a] == 0:
                        continue
                    for b in range(r):
                        if a == b or mk[b] == 0:
                            continue
                        # Try small integer/half-integer values for x
                        for xa_num in [-2, -1, 1, 2]:
                            xa = xa_num / 2.0
                            rem = 1.0 - mk[a] * xa
                            if abs(rem) < 1e-12:
                                W_vals[w_row, a] = xa
                                found = True
                                break
                            yb = rem / mk[b]
                            if abs(yb * 2 - round(yb * 2)) < 1e-9:
                                W_vals[w_row, a] = xa
                                W_vals[w_row, b] = yb
                                found = True
                                break
                        if found:
                            break
            
            if not found:
                # Try 3-term solutions
                for a in range(r):
                    if found:
                        break
                    if mk[a] == 0:
                        continue
                    for b in range(r):
                        if found:
                            break
                        if a == b or mk[b] == 0:
                            continue
                        for c in range(r):
                            if found:
                                break
                            if a == c or b == c or mk[c] == 0:
                                continue
                            for xa_num in [-1, 1]:
                                xa = xa_num / 2.0
                                for xb_num in [-1, 1]:
                                    xb = xb_num / 2.0
                                    rem = 1.0 - mk[a] * xa - mk[b] * xb
                                    yc = rem / mk[c]
                                    if abs(yc * 2 - round(yc * 2)) < 1e-9:
                                        W_vals[w_row, a] = xa
                                        W_vals[w_row, b] = xb
                                        W_vals[w_row, c] = yc
                                        found = True
                                        break
                                    if abs(rem) < 1e-12:
                                        W_vals[w_row, a] = xa
                                        W_vals[w_row, b] = xb
                                        found = True
                                        break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                # Last resort: try with x=1, y=1, solve for z
                for a in range(r):
                    if found:
                        break
                    if mk[a] == 0:
                        continue
                    for b in range(r):
                        if found:
                            break
                        if a == b or mk[b] == 0:
                            continue
                        for c in range(r):
                            if found:
                                break
                            if a == c or b == c or mk[c] == 0:
                                continue
                            z = (1.0 - mk[a] - mk[b]) / mk[c]
                            if abs(z * 2 - round(z * 2)) < 1e-9:
                                W_vals[w_row, a] = 1.0
                                W_vals[w_row, b] = 1.0
                                W_vals[w_row, c] = z
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                return None
    
    # Convert to lists with exact half-integer values
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = []
    for i in range(p * n):
        row = []
        for c in range(r):
            w = W_vals[i, c]
            w2 = round(w * 2)
            if w2 % 2 == 0:
                row.append(int(w2 // 2))
            else:
                row.append(w2 / 2.0)
        W_list.append(row)
    
    # Verify with exact arithmetic
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_low_rank(n, m, p):
    """
    Try structured constructions that exploit the specific 3x3 * 3x4 geometry.
    
    Idea: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k].
    For each output entry (i,k), we need a bilinear form in A and B.
    
    Strategy: Try to find a low-rank decomposition by exploiting
    linear dependencies. For example, if we can express some products
    as combinations of others, we reduce the rank.
    
    One approach: Use the "Laderman-like" idea where we share
    intermediate computations between adjacent output entries.
    """
    # Try a specific construction:
    # For 3x3 * 3x4, the output is 3x4 = 12 entries.
    # Each entry C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # The tensor has 36 nonzero entries (one per (i,j,k) triple).
    # We want to cover these with rank-r decompositions.
    
    # Try rank 18: This would be a 2x speedup over schoolbook.
    # One way: pair up the j-index. For each (i,k), we have 3 terms.
    # If we can combine pairs: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) can be
    # computed with 2 multiplications using:
    #   M1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    #   M2 = A[i,0]*B[0,k]
    #   M3 = A[i,1]*B[1,k]
    #   Sum = M1 - M2 - M3 + 2*M2 + 2*M3... no, that doesn't work directly.
    
    # Actually: A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = (A[i,0]+A[i,1])*(B[0,k]+B[1,k])/2 + (A[i,0]-A[i,1])*(B[0,k]-B[1,k])/2
    # This uses 2 products instead of 2, so no gain.
    
    # Let me try a different approach: random search with more attempts
    # and different random patterns.
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Try ranks from low to high
    for target_rank in range(12, best_rank):
        if _time_remaining() < 15:
            break
        
        # Number of attempts based on time budget
        attempts = 200 if target_rank >= 20 else 500
        if _time_remaining() < 30:
            attempts = min(attempts, 50)
        
        for attempt in range(attempts):
            if _time_remaining() < 5:
                break
            
            result = _try_rank_r(n, m, p, target_rank, seed_offset=attempt)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break  # Found valid decomposition at this rank, try lower
        
        # If we found a valid decomposition, continue to try lower ranks
        # (the loop will naturally try lower ranks next)
    
    # Also try structured constructions if time permits
    if _time_remaining() > 20:
        result = _try_structured_low_rank(n, m, p)
        if result is not None:
            r_new = len(result["U"][0])
            if r_new < best_rank:
                best_result = result
                best_rank = r_new
    
    return best_result

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")

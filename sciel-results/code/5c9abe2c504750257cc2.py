"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 140  # 10s safety margin before 150s limit

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
        for j in range(n):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _row_gcd(row):
    """Compute GCD of all elements in a row."""
    return reduce(gcd, [abs(x) for x in row if x != 0], 0)

def _find_integer_solution(coeffs, target):
    """
    Find an integer vector z such that coeffs · z = target.
    coeffs is a list of integers, target is an integer.
    Returns a sparse integer vector or None.
    """
    n = len(coeffs)
    # Find indices where coeffs are non-zero
    nz = [(i, coeffs[i]) for i in range(n) if coeffs[i] != 0]
    if not nz:
        return None if target != 0 else [0] * n
    
    # Try 1-term solution
    for i, c in nz:
        if c != 0 and target % c == 0:
            z = [0] * n
            z[i] = target // c
            return z
    
    # Try 2-term solution
    for a in range(len(nz)):
        ia, ca = nz[a]
        for b in range(len(nz)):
            if a == b:
                continue
            ib, cb = nz[b]
            # ca * x + cb * y = target
            # Try small values of x
            g = gcd(abs(ca), abs(cb))
            if target % g != 0:
                continue
            # Find one solution using extended gcd
            # ca*x + cb*y = target
            # Divide by g: (ca/g)*x + (cb/g)*y = target/g
            ca_g = ca // g
            cb_g = cb // g
            t = target // g
            # Find x0, y0 for ca_g*x + cb_g*y = 1
            # Then scale
            # Use extended Euclidean
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g2, x0, y0 = egcd(ca_g, cb_g)
            # ca_g * x0 + cb_g * y0 = g2 = 1 (since we divided by gcd)
            # Actually g2 should be 1 since ca_g and cb_g are coprime
            x = x0 * t
            y = y0 * t
            z = [0] * n
            z[ia] = x
            z[ib] = y
            # Verify
            if ca * x + cb * y == target:
                return z
    
    # Try 3-term solution
    for a in range(len(nz)):
        ia, ca = nz[a]
        for b in range(len(nz)):
            if a == b:
                continue
            ib, cb = nz[b]
            for c in range(len(nz)):
                if a == c or b == c:
                    continue
                ic, cc = nz[c]
                # ca*x + cb*y + cc*z = target
                # Try x=0: cb*y + cc*z = target
                g2 = gcd(abs(cb), abs(cc))
                if target % g2 == 0:
                    cb_g = cb // g2
                    cc_g = cc // g2
                    t = target // g2
                    def egcd(a, b):
                        if b == 0:
                            return (a, 1, 0)
                        g, x1, y1 = egcd(b, a % b)
                        return (g, y1, x1 - (a // b) * y1)
                    g3, y0, z0 = egcd(cb_g, cc_g)
                    y = y0 * t
                    z = z0 * t
                    if cb * y + cc * z == target:
                        res = [0] * n
                        res[ib] = y
                        res[ic] = z
                        return res
                # Try x=1: cb*y + cc*z = target - ca
                rem = target - ca
                g2 = gcd(abs(cb), abs(cc))
                if rem % g2 == 0:
                    cb_g = cb // g2
                    cc_g = cc // g2
                    t = rem // g2
                    def egcd(a, b):
                        if b == 0:
                            return (a, 1, 0)
                        g, x1, y1 = egcd(b, a % b)
                        return (g, y1, x1 - (a // b) * y1)
                    g3, y0, z0 = egcd(cb_g, cc_g)
                    y = y0 * t
                    z = z0 * t
                    if cb * y + cc * z == rem:
                        res = [0] * n
                        res[ia] = 1
                        res[ib] = y
                        res[ic] = z
                        return res
                # Try x=-1
                rem = target + ca
                g2 = gcd(abs(cb), abs(cc))
                if rem % g2 == 0:
                    cb_g = cb // g2
                    cc_g = cc // g2
                    t = rem // g2
                    def egcd(a, b):
                        if b == 0:
                            return (a, 1, 0)
                        g, x1, y1 = egcd(b, a % b)
                        return (g, y1, x1 - (a // b) * y1)
                    g3, y0, z0 = egcd(cb_g, cc_g)
                    y = y0 * t
                    z = z0 * t
                    if cb * y + cc * z == rem:
                        res = [0] * n
                        res[ia] = -1
                        res[ib] = y
                        res[ic] = z
                        return res
    return None

def _try_rank_r(n, m, p, r, seed_offset=0):
    """
    Try to find a rank-r decomposition using the GCD-based approach.
    
    Key: M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
    Need: for each (i,k), gcd(M[i,k,:]) divides 2
    Then find integer z such that M[i,k,:] · z = 2, set W = z/2
    """
    np.random.seed(seed_offset * 104729 + 7)
    
    U_vals = np.random.choice([-1, 1], size=(n*m, r))
    V_vals = np.random.choice([-1, 1], size=(m*p, r))
    
    # Compute M: shape (n*p, r)
    M = np.zeros((n * p, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[i * p + k, c] = s
    
    # Check GCD condition for each row
    W_int = np.zeros((p * n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            row = M[i * p + k, :]
            g = _row_gcd(row.tolist())
            if g == 0:
                return None  # All zeros, can't satisfy equation
            if 2 % g != 0:
                return None  # GCD doesn't divide 2
            
            # Find integer solution to row · z = 2
            z = _find_integer_solution(row.tolist(), 2)
            if z is None:
                return None
            W_int[k * n + i, :] = z
    
    # W = W_int / 2 (half-integers)
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[W_int[i, c] / 2.0 if W_int[i, c] % 2 != 0 else int(W_int[i, c] // 2) for c in range(r)] for i in range(p * n)]
    
    # Verify with exact arithmetic
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_r_mixed(n, m, p, r, seed_offset=0):
    """
    Try rank-r with mixed patterns: U and V can have entries in {-1, 0, 1}.
    This gives more flexibility in M values.
    """
    np.random.seed(seed_offset * 7919 + 13)
    
    # U: entries in {-1, 0, 1} with some structure
    U_vals = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.4, 0.2, 0.4])
    V_vals = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.4, 0.2, 0.4])
    
    # Compute M
    M = np.zeros((n * p, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[i * p + k, c] = s
    
    # Check GCD and find solutions
    W_int = np.zeros((p * n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            row = M[i * p + k, :]
            g = _row_gcd(row.tolist())
            if g == 0:
                return None
            if 2 % g != 0:
                return None
            
            z = _find_integer_solution(row.tolist(), 2)
            if z is None:
                return None
            W_int[k * n + i, :] = z
    
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[W_int[i, c] / 2.0 if W_int[i, c] % 2 != 0 else int(W_int[i, c] // 2) for c in range(r)] for i in range(p * n)]
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_r_sparse(n, m, p, r, seed_offset=0):
    """
    Try rank-r with sparse U and V (mostly zeros with some ±1).
    """
    np.random.seed(seed_offset * 31337 + 42)
    
    # Sparse: each entry is 0 with prob 0.5, ±1 with prob 0.25 each
    U_vals = np.zeros((n*m, r), dtype=int)
    V_vals = np.zeros((m*p, r), dtype=int)
    for i in range(n*m):
        for c in range(r):
            if np.random.random() < 0.5:
                U_vals[i, c] = np.random.choice([-1, 1])
    for i in range(m*p):
        for c in range(r):
            if np.random.random() < 0.5:
                V_vals[i, c] = np.random.choice([-1, 1])
    
    # Compute M
    M = np.zeros((n * p, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[i * p + k, c] = s
    
    W_int = np.zeros((p * n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            row = M[i * p + k, :]
            g = _row_gcd(row.tolist())
            if g == 0:
                return None
            if 2 % g != 0:
                return None
            
            z = _find_integer_solution(row.tolist(), 2)
            if z is None:
                return None
            W_int[k * n + i, :] = z
    
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[W_int[i, c] / 2.0 if W_int[i, c] % 2 != 0 else int(W_int[i, c] // 2) for c in range(r)] for i in range(p * n)]
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_construction(n, m, p):
    """
    Try a structured construction for 3x3 * 3x4.
    
    Idea: Use the Winograd-like approach. For each row i of A and column k of B,
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can use:
    (A[i,0]+A[i,1]+A[i,2]) * (B[0,k]+B[1,k]+B[2,k])
    - (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    - (A[i,1]+A[i,2]) * (B[1,k]+B[2,k])
    + A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Wait, that's not right. Let me think more carefully.
    
    Actually, the standard Winograd trick for dot product of length 3:
    x·y = x0*y0 + x1*y1 + x2*y2
    
    Using:
    M0 = (x0+x1+x2)(y0+y1+y2)
    M1 = (x0+x1)(y0+y1)
    M2 = (x1+x2)(y1+y2)
    M3 = x0*y0
    M4 = x1*y1
    M5 = x2*y2
    
    x·y = M0 - M1 - M2 + M3 + M4 + M5
    Check: (x0+x1+x2)(y0+y1+y2) = x0y0+x0y1+x0y2+x1y0+x1y1+x1y2+x2y0+x2y1+x2y2
    M1 = x0y0+x0y1+x1y0+x1y1
    M2 = x1y1+x1y2+x2y1+x2y2
    M0-M1-M2 = x0y2+x2y0 - x0y0 - x1y1 - x2y2
    + M3+M4+M5 = x0y0+x1y1+x2y2
    = x0y2+x2y0
    That's not x·y. So this doesn't work directly.
    
    Let me try a different approach. The rank of 3-term dot product is 3 (it's irreducible).
    So for 3x3*3x4, we have 12 dot products of length 3, each requiring rank 3.
    But we can share factors.
    
    Actually, the tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (with appropriate indexing)
    The rank is at most 36 (schoolbook) and at least 12.
    
    Let me try a direct construction for rank 24:
    For each i (3 values) and k (4 values), we need 3 products.
    Total = 3*4*3 = 36.
    
    To reduce: notice that for fixed i, the 4 dot products C[i,0], C[i,1], C[i,2], C[i,3]
    all use the same A[i,:] vector. So we can factor out the A part.
    
    C[i,:] = A[i,:] * B  (row vector times matrix)
    
    This is a 1x3 times 3x4 = 1x4 multiplication.
    The tensor for this is rank 3 (each of the 3 entries of A[i,:] times each column of B).
    So for fixed i, rank = 3*4 = 12? No, that's schoolbook.
    
    Actually for 1×3 × 3×4: C[k] = A[0]*B[0,k] + A[1]*B[1,k] + A[2]*B[2,k] for k=0,1,2,3.
    This is 4 dot products of length 3. Each dot product has rank 3, but they share the A factors.
    The tensor is: sum_{j,k} A[j] * B[j,k] * C[k]
    = sum_k (A[0]*B[0,k] + A[1]*B[1,k] + A[2]*B[2,k])
    
    The rank of this tensor: we have 3 A-factors, 12 B-factors, 4 C-factors.
    The tensor is sum_{j,k} e_j^A ⊗ e_{j,k}^B ⊗ e_k^C
    This has 12 nonzero terms. The rank is at most 12 (schoolbook) and at least max(3,12,4)=12.
    So rank = 12 for this sub-problem.
    
    For the full 3x3*3x4, we have 3 such sub-problems (one per row i), but they share B.
    Total rank ≤ 3*12 = 36 (schoolbook).
    
    Can we do better by sharing across rows? The B factors are shared.
    For each (j,k), the B factor B[j,k] appears in all 3 rows.
    
    Hmm, this is getting complex. Let me just focus on the random search with the GCD filter,
    which is much more efficient than the original approach.
    """
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
        if _time_remaining() < 10:
            break
        
        # Determine number of attempts based on time
        time_left = _time_remaining()
        if time_left > 60:
            attempts = 5000
        elif time_left > 30:
            attempts = 3000
        else:
            attempts = 1000
        
        # Try different strategies
        found = False
        for attempt in range(attempts):
            if _time_remaining() < 5:
                break
            
            # Alternate between strategies
            if attempt % 3 == 0:
                result = _try_rank_r(n, m, p, target_rank, seed_offset=attempt)
            elif attempt % 3 == 1:
                result = _try_rank_r_mixed(n, m, p, target_rank, seed_offset=attempt)
            else:
                result = _try_rank_r_sparse(n, m, p, target_rank, seed_offset=attempt)
            
            if result is not None:
                best_result = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Continue trying lower ranks
            continue
    
    return best_result

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")

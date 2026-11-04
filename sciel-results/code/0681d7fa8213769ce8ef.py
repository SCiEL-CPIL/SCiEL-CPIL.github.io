"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """Strassen rank-7 decomposition for 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _construct_3x4x4_winograd():
    """Construct a rank-44 decomposition for 3x4x4 using Strassen on 2x2 sub-blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    sU, sV, sW = _winograd_2x2x2()
    col = 0
    for g in range(2):
        j0, j1 = 2*g, 2*g+1
        for h in range(2):
            k0, k1 = 2*h, 2*h+1
            for s in range(7):
                c = col + s
                U[0*m + j0][c] = sU[0][s]
                U[0*m + j1][c] = sU[1][s]
                U[1*m + j0][c] = sU[2][s]
                U[1*m + j1][c] = sU[3][s]
                V[j0*p + k0][c] = sV[0][s]
                V[j0*p + k1][c] = sV[1][s]
                V[j1*p + k0][c] = sV[2][s]
                V[j1*p + k1][c] = sV[3][s]
                W[k0*n + 0][c] = sW[0][s]
                W[k0*n + 1][c] = sW[1][s]
                W[k1*n + 0][c] = sW[2][s]
                W[k1*n + 1][c] = sW[3][s]
                col += 1
        for k in range(4):
            c = col
            U[2*m + j0][c] = 1
            V[j0*p + k][c] = 1
            W[k*n + 2][c] = 1
            col += 1
            U[2*m + j1][c] = 1
            V[j1*p + k][c] = 1
            W[k*n + 2][c] = 1
            col += 1
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}

def _rational_gauss_eliminate(A, b, tol=1e-12):
    """Solve Ax = b over rationals using exact Gaussian elimination.
    A: list of lists (m x n), b: list of length m.
    Returns list of Fractions of length n, or None if inconsistent.
    """
    m = len(A)
    n = len(A[0])
    # Build augmented matrix
    M = []
    for i in range(m):
        row = [Fraction(x) for x in A[i]] + [Fraction(b[i])]
        M.append(row)
    
    # Forward elimination
    pivot_rows = []
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(len(pivot_rows), m):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        if pivot != len(pivot_rows):
            M[len(pivot_rows)], M[pivot] = M[pivot], M[len(pivot_rows)]
        pivot_rows.append(len(pivot_rows))
        pv = M[pivot][col]
        # Normalize pivot row
        for j in range(col, n+1):
            M[pivot][j] /= pv
        # Eliminate other rows
        for row in range(m):
            if row == pivot:
                continue
            factor = M[row][col]
            if factor != 0:
                for j in range(col, n+1):
                    M[row][j] -= factor * M[pivot][j]
    
    # Check for inconsistency
    for row in range(m):
        if all(M[row][j] == 0 for j in range(n)) and M[row][n] != 0:
            return None
    
    # Extract solution (set free variables to 0)
    x = [Fraction(0)] * n
    for i, row in enumerate(pivot_rows):
        # Find the pivot column for this row
        for col in range(n):
            if M[row][col] != 0:
                x[col] = M[row][n]
                break
    return x

def _search_2x4x4_low_rank(r, seed_offset=0, max_iters=200):
    """
    Search for a rank-r decomposition of the 2x4x4 tensor (A is 2x4, B is 4x4, C is 2x4).
    Uses random linear combinations and exact rational solve.
    """
    n2, m, p = 2, 4, 4
    import random
    rng = random.Random(seed_offset)
    
    # Number of constraints: n2 * m * p = 32
    # Number of unknowns: r * (n2 + m + p) = r * 10
    # Need r * 10 >= 32, so r >= 4. But realistically need more.
    
    # Strategy: 
    # 1. Randomly initialize U, V, W with small integers
    # 2. Fix U and V, solve for W (linear in W)
    # 3. Check if W has integer/half-integer entries
    
    # Actually, let's use a different approach:
    # Fix the structure: U and V have random integer entries in {-1, 0, 1}
    # Then for each (i,k) pair, we have:
    # sum_c U[i,j] V[j,k] W[k,i] = delta_{i,k} (for the 2x4x4 case, C[i,k] = sum_j A[i,j] B[j,k])
    # Wait, the tensor is T[i,j,k] = A[i,j] B[j,k] C[k,i] where C is 2x4.
    # So T[i,j,k] = 1 if i is in {0,1}, j in {0..3}, k in {0..3} and we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor entries are T[i,j,k] = 1 for all i,j,k (since each product term contributes 1).
    # Actually no. The tensor M_{i,j,k} for matrix multiplication is 1 if the indices are valid.
    # For 2x4x4: M[i,j,k] = 1 for all i in {0,1}, j in {0..3}, k in {0..3}.
    # So we need sum_c U[i,j] V[j,k] W[k,i] = 1 for all i,j,k.
    
    # Let's fix U and V to have entries in {-1, 0, 1} and solve for W.
    # For fixed U, V, the system is linear in W:
    # For each (i,j,k): sum_c (U[i,j] * V[j,k]) * W[k,i] = 1
    # Group by (k,i): sum_j U[i,j] V[j,k] * W[k,i] = sum_j U[i,j] V[j,k] = 1
    
    # Wait, that's not right. Let me re-derive.
    # We need: for all i, j, k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # This is nonlinear in U, V, W.
    
    # Alternative: Fix U and V, then for each c, we have a rank-1 contribution.
    # The sum over c must equal the tensor.
    
    # Let's try: fix U and V with small integers, then solve for W.
    # The system is: for each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # This is still nonlinear because W[k,i,c] is multiplied by U[i,j,c]*V[j,k,c].
    
    # Better approach: Fix U and V. Then define for each c:
    # M_c[i,j,k] = U[i,j,c] * V[j,k,c]
    # We need sum_c M_c[i,j,k] * W[k,i,c] = 1
    # This is linear in W! For each (k,i), we have:
    # sum_c (sum_j M_c[i,j,k]) * W[k,i,c] = sum_j 1 = m = 4
    # Wait, that's not right either. Let me be more careful.
    
    # We need: for all i,j,k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # Let's fix U and V. Then for each (i,k), the variable is W[k,i,c] for c=0..r-1.
    # But W[k,i,c] appears in multiple equations (for different j).
    
    # For fixed (i,k), the equations over j are:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1, for j=0..3
    # This is a system of 4 equations in r unknowns W[k,i,0],...,W[k,i,r-1].
    
    # So for each (i,k) pair (there are 2*4=8 pairs), we have a 4x r linear system.
    # We can solve each independently!
    
    # Let's try this:
    # 1. Randomly initialize U and V with entries in {-1, 0, 1}
    # 2. For each (i,k), solve the 4x r system for W[k,i,:]
    # 3. Check if all solutions are integers or half-integers
    
    for it in range(max_iters):
        # Random U: 2x4xr
        U = [[rng.choice([-1,0,1]) for _ in range(r)] for _ in range(2*4)]
        # Random V: 4x4xr
        V = [[rng.choice([-1,0,1]) for _ in range(r)] for _ in range(4*4)]
        
        # Solve for W
        W = [[0]*r for _ in range(4*2)]  # p*n = 8 rows
        
        valid = True
        for i in range(2):
            for k in range(4):
                # Build 4 x r system: A[j,c] = U[i*4+j, c] * V[j*4+k, c]
                # b[j] = 1
                A = [[U[i*4+j, c] * V[j*4+k, c] for c in range(r)] for j in range(4)]
                b = [1, 1, 1, 1]
                
                # Solve A * W[k*2+i, :]^T = b
                # Use numpy for speed, then check rationality
                A_np = np.array(A, dtype=float)
                b_np = np.array(b, dtype=float)
                
                try:
                    # Use least squares as a check
                    sol, residuals, rank_sys, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                    # Check if solution is exact
                    if np.max(np.abs(A_np @ sol - b_np)) > 1e-9:
                        valid = False
                        break
                    
                    # Check if solution entries are close to integers or half-integers
                    w_row = []
                    for val in sol:
                        # Check if close to integer or half-integer
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < 1e-9:
                            w_row.append(int(nearest_int))
                        elif abs(val - nearest_half) < 1e-9:
                            w_row.append(nearest_half)
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                    W[k*2+i] = w_row
                except Exception:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Verify the full decomposition for 2x4x4
        U_full = [[0]*r for _ in range(2*4)]
        V_full = [[0]*r for _ in range(4*4)]
        W_full = [[0]*r for _ in range(4*2)]
        for i in range(2):
            for j in range(4):
                U_full[i*4+j] = U[i*4+j]
        for j in range(4):
            for k in range(4):
                V_full[j*4+k] = V[j*4+k]
        for k in range(4):
            for i in range(2):
                W_full[k*2+i] = W[k*2+i]
        
        # Verify
        ok = True
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for c in range(r):
                        s += U_full[i*4+j][c] * V_full[j*4+k][c] * W_full[k*2+i][c]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return U_full, V_full, W_full
    
    return None

def _construct_3x4x4_rational_search(max_time=60):
    """
    Search for a low-rank decomposition of 3x4x4 by:
    1. Fixing the 1x4x4 part (rank 16, schoolbook)
    2. Searching for a low-rank 2x4x4 part
    """
    n, m, p = 3, 4, 4
    start_time = time.time()
    
    # The 1x4x4 part uses rank 16 (schoolbook)
    # The 2x4x4 part: we search for rank r
    # Total rank = 16 + r
    
    # Try different ranks for the 2x4x4 part
    # Known: 2x4x4 can be done with rank 28 (Strassen on 2x2 blocks)
    # Let's try r = 20, 22, 24, 26, 28
    
    best_2x4x4 = None
    best_r = None
    
    for r in [20, 22, 24, 26, 28]:
        if time.time() - start_time > max_time - 10:
            break
        
        result = _search_2x4x4_low_rank(r, seed_offset=r, max_iters=50)
        if result is not None:
            best_2x4x4 = result
            best_r = r
            break  # Found the lowest rank, stop
    
    if best_2x4x4 is None:
        return None
    
    U2, V2, W2 = best_2x4x4
    r2 = best_r
    
    # Build the full 3x4x4 decomposition
    total_rank = 16 + r2
    U = [[0] * total_rank for _ in range(3*4)]
    V = [[0] * total_rank for _ in range(4*4)]
    W = [[0] * total_rank for _ in range(4*3)]
    
    # 1x4x4 part (row 2 of A)
    col = 0
    for j in range(4):
        for k in range(4):
            U[2*4+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    # 2x4x4 part
    for i in range(2):
        for j in range(4):
            for c in range(r2):
                U[i*4+j][col+c] = U2[i*4+j][c]
    for j in range(4):
        for k in range(4):
            for c in range(r2):
                V[j*4+k][col+c] = V2[j*4+k][c]
    for k in range(4):
        for i in range(2):
            for c in range(r2):
                W[k*3+i][col+c] = W2[k*2+i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try the Winograd construction (rank 44)
    try:
        candidate = _construct_3x4x4_winograd()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            rank = len(candidate["U"][0])
            if rank < best_rank:
                best = candidate
                best_rank = rank
    except Exception:
        pass
    
    # Try rational search for lower rank
    if time.time() < deadline - 30:
        try:
            candidate = _construct_3x4x4_rational_search(max_time=60)
            if candidate:
                rank = len(candidate["U"][0])
                if rank < best_rank and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                    best = candidate
                    best_rank = rank
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END

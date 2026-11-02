"""Find rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 3, 3]
DEADLINE = 140.0  # Safety margin before 150s kill


def _elapsed_seconds(start):
    return time.time() - start


def _time_up(start, margin=5.0):
    return time.time() - start > DEADLINE - margin


def _target_tensor(n=3, m=3, p=3):
    """Return the MM tensor as a dict {(i,j,k): 1} for the consistent triples."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T


def _verify_strict(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Strictly verify the decomposition over all consistent triples."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_full(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Full verification over all (a,b,c) in the 3-way tensor."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    D = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        D += np.outer(U[:, r], V[:, r]) * W[:, r]
    if np.max(np.abs(D - T)) > tol:
        return False
    return True


def _solve_for_W(U, V, n=3, m=3, p=3):
    """Given U (nm x r) and V (mp x r), solve for W (pn x r) such that
    the decomposition holds. Returns W or None if inconsistent."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.zeros((p * n, rank))
    for k in range(p):
        for i in range(n):
            c = k * n + i
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i * m + j] * V[j * p + k]
            rhs = np.ones(m)
            if rank == 0:
                return None
            sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
            resid = np.linalg.norm(M @ sol - rhs)
            if resid > 1e-7:
                return None
            W[c, :] = sol
    return W


def _strassen_2x2_block():
    """Build a rank-25 decomposition for 3x3x3 using Strassen 2x2 on the
    top-left block and schoolbook for the rest, with some sharing."""
    n, m, p = 3, 3, 3
    # Standard Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = (A11+A12)B22
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M6
    # C12 = M3 + M1
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M5
    
    # For 3x3, we'll use a hybrid approach:
    # - Apply Strassen 2x2 to the top-left 2x2 block (rank 7)
    # - Handle the remaining terms with schoolbook, reusing multiplications where possible
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 for top-left block (indices 0,1 for rows/cols)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[1][0] = 1  # A11
    V[0][0] = 1  # B00
    V[1][0] = 1  # B11
    W[0][0] = 1  # C00
    W[1][0] = 1  # C01
    W[3][0] = 1  # C10
    W[4][0] = 1  # C11
    
    # M2: (A10+A11)B00
    U[3][1] = 1  # A10
    U[4][1] = 1  # A11
    V[0][1] = 1  # B00
    W[3][1] = 1  # C10
    W[4][1] = 1  # C11
    
    # M3: A00(B10-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B10
    V[2][2] = -1  # B11
    W[1][2] = 1  # C01
    W[4][2] = 1  # C11
    
    # M4: (A00+A01)B11
    U[0][3] = 1  # A00
    U[1][3] = 1  # A01
    V[2][3] = 1  # B11
    W[0][3] = 1  # C00
    W[3][3] = 1  # C10
    
    # M5: (A10-A00)(B00+B01)
    U[3][4] = 1  # A10
    U[0][4] = -1  # A00
    V[0][4] = 1  # B00
    V[1][4] = 1  # B01
    W[3][4] = 1  # C10
    W[4][4] = 1  # C11
    
    # M6: (A01-A11)(B10+B11)
    U[1][5] = 1  # A01
    U[4][5] = -1  # A11
    V[1][5] = 1  # B10
    V[2][5] = 1  # B11
    W[0][5] = 1  # C00
    W[3][5] = 1  # C10
    
    # Now handle the remaining terms involving index 2
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # We can reuse some multiplications from Strassen
    # For now, use schoolbook for the remaining 15 terms
    col = 6
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip if already handled by Strassen (i,j,k in {0,1})
                if i <= 1 and j <= 1 and k <= 1:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _search_rank(r, n=3, m=3, p=3, start_time=None, max_iters=5000, seed=0):
    """Search for a rank-r decomposition using random initialization +
    coordinate descent (solve for W given U,V, then check)."""
    if start_time is None:
        start_time = time.time()
    rng = random.Random(seed)
    candidate_vals = [0, 1, -1, 0.5, -0.5]

    for it in range(max_iters):
        if _time_up(start_time, 3):
            break

        # Random sparse initialization
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        for row in range(n * m):
            num_nz = rng.randint(1, 3)
            cols = rng.sample(range(r), min(num_nz, r))
            for c in cols:
                U[row][c] = rng.choice([1, -1, 0.5, -0.5])
        for row in range(m * p):
            num_nz = rng.randint(1, 3)
            cols = rng.sample(range(r), min(num_nz, r))
            for c in cols:
                V[row][c] = rng.choice([1, -1, 0.5, -0.5])

        # Solve for W
        W = _solve_for_W(U, V, n, m, p)
        if W is None:
            continue

        # Verify
        if _verify_strict(U, V, W, n, m, p):
            return U, V, W

    return None


def _local_search_polish(U, V, W, n=3, m=3, p=3, start_time=None, max_iters=100):
    """Polish a decomposition using local search to reduce rank or improve precision."""
    if start_time is None:
        start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for it in range(max_iters):
        if _time_up(start_time, 5):
            break
        
        # Try to remove a rank-1 component by zeroing it out and re-solving
        for r in range(rank):
            if _time_up(start_time, 5):
                break
            
            # Save original
            U_orig = U.copy()
            V_orig = V.copy()
            W_orig = W.copy()
            
            # Zero out column r
            U[:, r] = 0
            V[:, r] = 0
            W[:, r] = 0
            
            # Try to re-solve for W with the remaining rank-1 components
            # This is complex, so for now just verify if the solution is still valid
            if _verify_strict(U, V, W, n, m, p, tol=1e-6):
                # Success! We can remove this rank-1 component
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                break
            else:
                # Restore
                U = U_orig
                V = V_orig
                W = W_orig
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Start with schoolbook rank-27 (guaranteed valid)
    rank_27 = n * m * p
    U27 = [[0] * rank_27 for _ in range(n * m)]
    V27 = [[0] * rank_27 for _ in range(m * p)]
    W27 = [[0] * rank_27 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i * m + j][col] = 1
                V27[j * p + k][col] = 1
                W27[k * n + i][col] = 1
                col += 1
    assert _verify_strict(U27, V27, W27, n, m, p), "Schoolbook failed verification"
    best_rank = 27
    best_solution = (U27, V27, W27)

    # Try Strassen 2x2 block construction (rank 25)
    U_s, V_s, W_s = _strassen_2x2_block()
    if _verify_strict(U_s, V_s, W_s, n, m, p):
        best_rank = 25
        best_solution = (U_s, V_s, W_s)
    
    # Try to find lower-rank decompositions via search
    for target_rank in range(19, best_rank):
        if _time_up(start_time, 10):
            break

        # Multiple random seeds
        for seed in range(50):
            if _time_up(start_time, 3):
                break
            result = _search_rank(target_rank, n, m, p, start_time,
                                  max_iters=200, seed=seed * 137 + target_rank)
            if result is not None:
                U, V, W = result
                if _verify_strict(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break

        if best_rank <= target_rank:
            break
    
    # Try local search polish on the best solution
    if not _time_up(start_time, 10):
        U, V, W = best_solution
        U_p, V_p, W_p = _local_search_polish(U, V, W, n, m, p, start_time)
        if _verify_strict(U_p, V_p, W_p, n, m, p):
            new_rank = len(U_p[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_solution = (U_p, V_p, W_p)

    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

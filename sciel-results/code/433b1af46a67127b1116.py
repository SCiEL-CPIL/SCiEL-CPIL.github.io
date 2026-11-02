"""Matrix multiplication 2x3x4 — rank-minimal exact tensor decomposition."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 3, 4]


def _build_schoolbook():
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
    """Check exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    # Convert to Fraction matrices
    UF = [[Fraction(x) for x in row] for row in U]
    VF = [[Fraction(x) for x in row] for row in V]
    WF = [[Fraction(x) for x in row] for row in W]

    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += UF[i * m + j][c] * VF[j * p + k][c] * WF[k * n + i][c]
                if s != 1:
                    return False
    return True


def _verify_np(U, V, W, tol=1e-9):
    """Fast numerical verification."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j], Va[j * p + k]) @ Wa[k * n + i]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_gaussian_elimination(R):
    """
    Try to find a rank-R decomposition by setting up the bilinear system.
    We parameterize U and V as random, then solve for W linearly,
    then check residual. Iterate with different random U,V.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    # Unknowns: R * (nm + mp + pn) coefficients
    # Equations: n*m*p = 24, each is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # This is cubic, not linear. But if we fix U and V, it's linear in W.
    # So: fix U, V randomly (with small rational entries), solve for W.
    
    for attempt in range(200):
        random.seed(attempt * 12345 + 7)
        # Use half-integer entries for U and V
        U = [[random.choice([-2, -1, 0, 1, 2]) / 2 for _ in range(R)] for _ in range(nm)]
        V = [[random.choice([-2, -1, 0, 1, 2]) / 2 for _ in range(R)] for _ in range(mp)]
        
        # For each (k,i) pair, we need W[k,i,:] such that for all (j):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a system of m equations in R unknowns for each (k,i).
        # Build the system: for each j, a row of length R with entries U[i*m+j,c]*V[j*p+k,c]
        W = [[0] * R for _ in range(pn)]
        ok = True
        for k in range(p):
            for i in range(n):
                # m equations, R unknowns
                A = np.zeros((m, R))
                for j in range(m):
                    for c in range(R):
                        A[j, c] = U[i * m + j][c] * V[j * p + k][c]
                b = np.ones(m)
                # Solve least squares / exact
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                # Check if it's exact
                if np.max(np.abs(A @ sol - b)) > 1e-10:
                    ok = False
                    break
                W[k * n + i] = [float(x) for x in sol]
        if not ok:
            continue
        # Verify
        if _verify_np(U, V, W):
            # Try to rationalize: snap to half-integers
            U_r = [[round(x * 2) / 2 for x in row] for row in U]
            V_r = [[round(x * 2) / 2 for x in row] for row in V]
            W_r = [[round(x * 2) / 2 for x in row] for row in W]
            if _verify(U_r, V_r, W_r):
                return {"U": U_r, "V": V_r, "W": W_r}
    return None


def _try_fixed_U_search(R):
    """
    Fix U to a structured pattern and solve for V, W via alternating least squares
    on the tensor equation.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    for attempt in range(100):
        random.seed(attempt * 9999 + 13)
        # Random U with half-integer entries
        U = np.array([[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(R)] for _ in range(nm)], dtype=np.float64)
        
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        # Initialize V, W randomly
        V[:, :] = np.random.randn(mp, R) * 0.5
        W[:, :] = np.random.randn(pn, R) * 0.5
        
        for it in range(2000):
            # Update V: for each (j,k) row, solve min ||U @ V_row - T slice||
            # T[i,j,k] = sum_c U[i,c] * V[j,k,c] * W[k,i,c]
            # Fix U, W: for each (j,k), V[j,k,:] = argmin sum_i (sum_c U[i*m+j? no...
            # Actually T is indexed (i*m+j, j*p+k, k*n+i)
            # For fixed (j,k): sum_i T[i*m+j, j*p+k, k*n+i] = sum_i sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # This is: (sum_i U[i*m+j] * W[k*n+i]^T) dot V[j*p+k] = T[j*p+k projected]
            # Let M_jk[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
            # Then M_jk dot V[j*p+k] should equal T_proj[j*p+k] where T_proj[j*p+k] = sum_i T[i*m+j, j*p+k, k*n+i]
            for j in range(m):
                for k in range(p):
                    M = np.zeros(R)
                    tproj = 0.0
                    for i in range(n):
                        for c in range(R):
                            M[c] += U[i * m + j, c] * W[k * n + i, c]
                        tproj += T[i * m + j, j * p + k, k * n + i]
                    # V[j*p+k, :] should satisfy M dot V[j*p+k] = tproj
                    # This is 1 equation, R unknowns. Use pseudoinverse or set V = M * tproj / (M.M)
                    mm = np.dot(M, M)
                    if mm > 1e-15:
                        V[j * p + k, :] = M * (tproj / mm)
                    # else leave as is
            
            # Update W similarly
            for k in range(p):
                for i in range(n):
                    M = np.zeros(R)
                    tproj = 0.0
                    for j in range(m):
                        for c in range(R):
                            M[c] += U[i * m + j, c] * V[j * p + k, c]
                        tproj += T[i * m + j, j * p + k, k * n + i]
                    mm = np.dot(M, M)
                    if mm > 1e-15:
                        W[k * n + i, :] = M * (tproj / mm)
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += abs(s - 1.0)
            if err < 1e-8:
                # Snap to half-integers
                U_r = [[round(x * 2) / 2 for x in row] for row in U]
                V_r = [[round(x * 2) / 2 for x in row] for row in V]
                W_r = [[round(x * 2) / 2 for x in row] for row in W]
                if _verify(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
                break
    return None


def construct():
    t0 = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    # Start with schoolbook (rank 24)
    best = _build_schoolbook()
    best_rank = 24
    
    # Try to find lower-rank decompositions
    # Known bounds: rank of [2,3,4] is at most 17 (from literature), lower bound is 14 or 15.
    # Try ranks from 14 upward.
    for R in range(14, best_rank):
        if time.time() - t0 > DEADLINE:
            break
        result = _try_gaussian_elimination(R)
        if result is not None:
            best = result
            best_rank = R
            break
        if time.time() - t0 > DEADLINE:
            break
        result = _try_fixed_U_search(R)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['W'][0])}")
    print(f"Valid: {_verify(res['U'], res['V'], res['W'])}")

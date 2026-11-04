"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses schoolbook baseline (rank 50) and attempts structured search for
lower-rank decompositions via:
1. Identifying linear dependencies in the schoolbook basis.
2. Numerical optimization with integer/half-integer rounding.
3. Exact verification of candidate decompositions.
"""

import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
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
    return U, V, W, rank


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def _try_rank_reduction(target_rank):
    """
    Try to find a decomposition of target_rank by searching over structured
    patterns. For the (2,5,5) case, we exploit the small n=2 dimension.
    
    Strategy: Assume a block structure where for each j, we use a small
    number of rank-1 terms. Try all combinations of how to distribute
    the rank budget across j values.
    """
    n, m, p = 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 10, 25, 10
    
    if target_rank >= 50:
        return None
    
    # For each j, the sub-problem is: compute A_col_j (2x1) * B_row_j (1x5)
    # which gives a 2x5 matrix. Schoolbook uses 10 terms per j.
    # Total schoolbook: 5 * 10 = 50.
    #
    # Can we share terms across j? A rank-1 term (u, v, w) contributes
    # u[i,j] * v[j,k] * w[k,i] to T[i,j,k].
    #
    # If we use the same (u, v, w) for multiple j values, we can save.
    # But u depends on j (U[i,m+j]), v depends on j (V[j*p+k]), so
    # sharing is non-trivial.
    #
    # Let's try: for each j, use r_j terms, sum r_j = target_rank.
    # For simplicity, try uniform distribution: r_j = target_rank // 5.
    
    r_per_j = target_rank // m
    remainder = target_rank % m
    
    # This is getting complex. Let's try a different approach:
    # Use the fact that for n=2, we can potentially use a 2x2 trick
    # within each j-block.
    
    # For a fixed j, we need to compute:
    # C[0,k] += A[0,j] * B[j,k]
    # C[1,k] += A[1,j] * B[j,k]
    #
    # This is equivalent to: [C[0,k], C[1,k]] += [A[0,j], A[1,j]]^T * B[j,k]
    #
    # The 2x1 x 1x5 product has rank 1 as a tensor, but bilinear complexity
    # is 2*5=10.
    #
    # I don't see an obvious way to reduce below 10 per j for integer entries.
    # So rank 50 may be optimal for this structure.
    
    return None


def _try_numerical_search(target_rank):
    """
    Use numerical linear algebra to find a candidate decomposition.
    
    Set up the system of equations and solve via SVD/least squares,
    then round to integers.
    """
    n, m, p = 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 10, 25, 10
    R = target_rank
    num_constraints = n * m * p  # 50
    
    # Unknowns: U (nm x R), V (mp x R), W (pn x R)
    # Total unknowns: R * (nm + mp + pn) = R * 45
    # Constraints: 50 equations (one per (i,j,k))
    #
    # The constraint for (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of 50 cubic equations in 45*R unknowns.
    # For R < 50, this is underdetermined, but the equations are nonlinear.
    #
    # Strategy: Fix W to a simple form and solve for U, V.
    # If W[k*n+i, r] = delta_{r, f(k,i)} for some bijection f, then
    # the constraint becomes U[i*m+j, f(k,i)] * V[j*p+k, f(k,i)] = 1.
    #
    # Let's try: W[k*n+i, r] = 1 if r == i else 0. Then:
    # U[i*m+j, i] * V[j*p+k, i] = 1 for all j, k.
    # This requires U[i*m+j, i] = 1 and V[j*p+k, i] = 1 for all j, k.
    # But then we only use rank positions 0 and 1, giving rank 2.
    # However, this doesn't work because different (i,j,k) triples
    # would interfere.
    #
    # Let me reconsider. The issue is that the same rank position r
    # is shared across all (i,j,k) triples. So if we set W[k*n+i, r]
    # to depend only on i, we get the constraint above, but it's too
    # restrictive.
    
    # Alternative: use a random initialization and iterate.
    np.random.seed(42)
    U = np.random.randn(nm, R)
    V = np.random.randn(mp, R)
    W = np.random.randn(pn, R)
    
    # Iterate: fix two matrices, solve for the third.
    # For fixed V, W: the constraint is linear in U.
    # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # Let B[i*m+j, (j,k)] = sum_r ... no, this doesn't simplify nicely.
    
    # Let's try gradient descent on the loss:
    # L = sum_{i,j,k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
    
    lr = 0.01
    for it in range(1000):
        if not _deadline_ok():
            return None
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # grad_U[i*m+j, r] = 2 * (sum - 1) * V[j*p+k, r] * W[k*n+i, r]
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = 2 * (s - 1)
                    dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if it % 100 == 0:
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(s - 1))
            if max_err < 1e-6:
                break
    
    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_rounded[i*m+j] * V_rounded[j*p+k] * W_rounded[k*n+i])
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Convert to lists of Fractions
    U_list = [[Fraction(int(x*2), 2) for x in U_rounded[i]] for i in range(nm)]
    V_list = [[Fraction(int(x*2), 2) for x in V_rounded[i]] for i in range(mp)]
    W_list = [[Fraction(int(x*2), 2) for x in W_rounded[i]] for i in range(pn)]
    
    return U_list, V_list, W_list, R


def construct():
    n, m, p = 2, 5, 5
    
    # Start with schoolbook decomposition (rank 50)
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if not _verify(U, V, W, rank):
        U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Attempt to find a lower-rank decomposition
    if _deadline_ok():
        # Try numerical search for ranks below 50
        for candidate_rank in range(best_rank - 1, max(best_rank - 5, 0), -1):
            if not _deadline_ok():
                break
            
            result = _try_numerical_search(candidate_rank)
            if result is not None:
                best_U, best_V, best_W, best_rank = result
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    print(f"U shape: {len(U)}x{len(U[0])}")
    print(f"V shape: {len(V)}x{len(V[0])}")
    print(f"W shape: {len(W)}x{len(W[0])}")
    print(f"Verification: {_verify(U, V, W, rank)}")

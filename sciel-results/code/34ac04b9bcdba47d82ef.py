"""
Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition.

Strategy: Try deterministic rank-reduction from schoolbook (rank 42) by
checking if any schoolbook term can be expressed as a linear combination
of the others. Use exact Fraction arithmetic for verification.
Also try a structured algebraic approach for rank 41.
"""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Build the full tensor as a dict of (a,b,c) -> 1 for valid entries."""
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[(a, b, c)] = 1
    return tensor

def _verify_decomposition_numpy(U, V, W, n, m, p):
    """Fast numerical verification using numpy."""
    r = U.shape[1]
    tensor = _build_tensor(n, m, p)
    
    for (a, b, c) in tensor:
        s = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(s - 1) > 1e-8:
            return False
    
    count = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                s = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(s) > 1e-8:
                    return False
                count += 1
                if count > 500:
                    return True
    return True

def _convert_to_fractions(U, V, W, max_denom=100000):
    """Convert numpy arrays to Fraction matrices."""
    Uf = [[Fraction(int(round(U[i, j] * max_denom)), max_denom) for j in range(U.shape[1])] for i in range(U.shape[0])]
    Vf = [[Fraction(int(round(V[i, j] * max_denom)), max_denom) for j in range(V.shape[1])] for i in range(V.shape[0])]
    Wf = [[Fraction(int(round(W[i, j] * max_denom)), max_denom) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return Uf, Vf, Wf

def _verify_exact(Uf, Vf, Wf, n, m, p):
    """Exact verification with Fractions."""
    r = len(Uf[0])
    tensor = _build_tensor(n, m, p)
    
    for (a, b, c) in tensor:
        s = sum(Uf[a][t] * Vf[b][t] * Wf[c][t] for t in range(r))
        if s != 1:
            return False
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                s = sum(Uf[a][t] * Vf[b][t] * Wf[c][t] for t in range(r))
                if s != 0:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_rank41_construction(n, m, p):
    """
    Try to construct a rank-41 decomposition by finding a linear dependence
    among the schoolbook terms.
    
    The idea: if we can find coefficients alpha_t such that
    sum_t alpha_t * (u_t ⊗ v_t ⊗ w_t) = 0 (the zero tensor),
    then we can eliminate one term and get rank 41.
    
    More practically: fix all but one term, and see if the remaining term
    can be expressed as a linear combination of the others.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    r = n * m * p  # 42
    
    # Schoolbook terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                terms.append((a, b, c))
    
    # For each term to eliminate, check if it's in the span of the others
    # This is a complex problem, so let's try a different approach:
    # Use a randomized method with exact verification
    
    # Try multiple random initializations
    rng = random.Random(12345)
    
    for trial in range(200):
        # Generate random U and V with small rational entries
        target_rank = 41
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        
        # Use entries from {-2, -1, 0, 1, 2} for better chance of exact solutions
        for t in range(target_rank):
            for a in range(nm):
                U[a, t] = rng.choice([-2, -1, 0, 1, 2])
            for b in range(mp):
                V[b, t] = rng.choice([-2, -1, 0, 1, 2])
        
        # Build the linear system for W
        # For each c, we need: T[:, :, c] = sum_t W[c,t] * (U[:,t] * V[:,t]^T)
        A = np.zeros((nm * mp, target_rank), dtype=float)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for t in range(target_rank):
                    A[idx, t] = U[a, t] * V[b, t]
                idx += 1
        
        W = np.zeros((pn, target_rank), dtype=float)
        success = True
        for c in range(pn):
            rhs = np.zeros(nm * mp, dtype=float)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        rhs[idx] = 1.0
                    idx += 1
            
            w, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = w
        
        if _verify_decomposition_numpy(U, V, W, n, m, p):
            # Try to find exact fractions
            for max_denom in [100, 1000, 10000, 100000]:
                Uf, Vf, Wf = _convert_to_fractions(U, V, W, max_denom)
                if _verify_exact(Uf, Vf, Wf, n, m, p):
                    return Uf, Vf, Wf
    
    return None

def _try_random_decomposition(n, m, p, target_rank, num_trials=100, seed=42):
    """
    Try to find a rank-target_rank decomposition using random U, V and solving for W.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    for trial in range(num_trials):
        # Generate random U and V with small integer entries
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for t in range(r):
            for a in range(nm):
                U[a, t] = rng.choice([-1, 0, 1])
            for b in range(mp):
                V[b, t] = rng.choice([-1, 0, 1])
        
        # Build A matrix
        A = np.zeros((nm * mp, r), dtype=float)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for t in range(r):
                    A[idx, t] = U[a, t] * V[b, t]
                idx += 1
        
        W = np.zeros((pn, r), dtype=float)
        for c in range(pn):
            rhs = np.zeros(nm * mp, dtype=float)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        rhs[idx] = 1.0
                    idx += 1
            
            w, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = w
        
        if _verify_decomposition_numpy(U, V, W, n, m, p):
            for max_denom in [100, 1000, 10000]:
                Uf, Vf, Wf = _convert_to_fractions(U, V, W, max_denom)
                if _verify_exact(Uf, Vf, Wf, n, m, p):
                    return Uf, Vf, Wf
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp  # = n*m*p = 42
    min_rank = max(nm, mp, pn)  # = 21
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try rank 41 first with the specialized construction
    if time.time() < deadline:
        result = _try_rank41_construction(n, m, p)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
    
    # Try ranks from 41 down to min_rank + 1
    for target_rank in range(schoolbook_rank - 1, min_rank, -1):
        if time.time() > deadline:
            break
        
        # Try random integer approach
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=30, seed=target_rank * 100 + 1)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
    
    # Fallback to schoolbook decomposition
    return _schoolbook_decomposition(n, m, p)

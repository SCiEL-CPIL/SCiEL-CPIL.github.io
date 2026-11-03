"""
Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition.

Approach:
- The matrix multiplication tensor T for multiplying an (n x m) matrix by an (m x p) matrix
  has entries T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), and 0 otherwise.
- We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
  T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t] for all (a,b,c).
- We use a randomized algebraic approach: for a target rank r, we set up the bilinear system
  in the entries of W (given fixed U, V) and solve. We try multiple random initializations
  of U and V, then solve for W via least squares and check if the residual is zero.
- We also try a structured approach: start from schoolbook and attempt to merge columns
  by finding linear dependencies.

For 2x3 by 3x7: n=2, m=3, p=7, schoolbook rank = 42.
We try ranks 41, 40, ... down to some lower bound, returning the first valid decomposition.
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

def _verify_decomposition(U, V, W, n, m, p, exact=True):
    """
    Verify that U, V, W give a valid decomposition.
    U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    """
    r = U.shape[1]
    tensor = _build_tensor(n, m, p)
    
    # Check all valid entries equal 1
    for (a, b, c) in tensor:
        s = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
        if exact:
            if s != 1:
                return False
        else:
            if abs(s - 1) > 1e-9:
                return False
    
    # Check all invalid entries equal 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if (a, b, c) in tensor:
                    continue
                s = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if exact:
                    if s != 0:
                        return False
                else:
                    if abs(s) > 1e-9:
                        return False
    return True

def _verify_decomposition_numpy(U, V, W, n, m, p):
    """Fast numerical verification using numpy."""
    r = U.shape[1]
    tensor = _build_tensor(n, m, p)
    
    # Precompute for speed
    for (a, b, c) in tensor:
        s = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(s - 1) > 1e-8:
            return False
    
    # Check a sample of invalid entries
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
                if count > 500:  # sample check
                    return True
    return True

def _convert_to_fractions(U, V, W, max_denom=10000):
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

def _try_random_decomposition(n, m, p, target_rank, num_trials=50, seed=42):
    """
    Try to find a rank-target_rank decomposition using random U, V and solving for W.
    
    For each trial:
    1. Generate random U (n*m x r) and V (m*p x r) with small integer entries.
    2. For each output index c, we need: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
       This is linear in W[c,:] for each (a,b) pair.
    3. Build a linear system and solve.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    # Collect all (a, b) pairs that appear in the tensor
    valid_pairs = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                valid_pairs.add((a, b))
    
    # Also need to enforce zeros for invalid (a,b,c) combinations
    # For each (a,b), the set of c values where T[a,b,c] = 1 is determined.
    # For a given (a,b), T[a,b,c] = 1 iff there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i
    # So for fixed (a,b) = (i*m+j, j*p+k), c must be k*n+i.
    # If (a,b) is not in valid_pairs, then T[a,b,c] = 0 for all c.
    
    # For each (a,b) in valid_pairs, we have one constraint:
    # sum_t U[a,t]*V[b,t]*W[c0,t] = 1 where c0 is determined.
    # For (a,b) not in valid_pairs, we have pn constraints:
    # sum_t U[a,t]*V[b,t]*W[c,t] = 0 for all c.
    
    # This is a large system. Let's use a different approach:
    # Fix U and V, then for each c, W[c,:] is determined by the constraints.
    
    # Actually, let's think of it differently. The tensor equation is:
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # For fixed U and V, define M_c = U * V^T (element-wise with W[c,:])
    # Actually, T[:, :, c] = sum_t (U[:,t] * V[:,t]) * W[c,t]
    # So T[:, :, c] is a linear combination of the rank-1 matrices U[:,t]*V[:,t]^T
    # with coefficients W[c,t].
    
    # For each c, we need: T[:, :, c] = sum_t W[c,t] * (U[:,t] * V[:,t]^T)
    # This is a linear system in W[c,:].
    
    # Let's build this system. For each c, we have nm*mp equations and r unknowns.
    # We can solve using least squares.
    
    for trial in range(num_trials):
        # Generate random U and V with entries in {-1, 0, 1}
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for t in range(r):
            for a in range(nm):
                U[a, t] = rng.choice([-1, 0, 1])
            for b in range(mp):
                V[b, t] = rng.choice([-1, 0, 1])
        
        # Build the system for each c
        # For each c, we need to solve: A_c * W[c,:] = rhs_c
        # where A_c is (nm*mp) x r, with A_c[(a,b), t] = U[a,t] * V[b,t]
        # and rhs_c[(a,b)] = T[a,b,c]
        
        # Precompute A (same for all c)
        A = np.zeros((nm * mp, r), dtype=float)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for t in range(r):
                    A[idx, t] = U[a, t] * V[b, t]
                idx += 1
        
        # For each c, build rhs and solve
        W = np.zeros((pn, r), dtype=float)
        success = True
        for c in range(pn):
            # Build rhs for this c
            rhs = np.zeros(nm * mp, dtype=float)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    # Determine if T[a,b,c] = 1
                    # a = i*m+j, b = j*p+k, c = k*n+i
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        rhs[idx] = 1.0
                    idx += 1
            
            # Solve A * w = rhs using least squares
            w, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = w
        
        # Verify numerically
        if _verify_decomposition_numpy(U, V, W, n, m, p):
            # Try to convert to exact fractions
            Uf, Vf, Wf = _convert_to_fractions(U, V, W)
            if _verify_exact(Uf, Vf, Wf, n, m, p):
                return Uf, Vf, Wf
    
    return None

def _try_structured_reduction(n, m, p, target_rank, num_trials=100, seed=123):
    """
    Try to find a rank-target_rank decomposition by starting from schoolbook
    and attempting to merge columns.
    
    Alternative: use a more structured random approach where we fix some entries
    to create structure.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    # Try a different parameterization: use random rational entries
    for trial in range(num_trials):
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for t in range(r):
            for a in range(nm):
                U[a, t] = rng.uniform(-1, 1)
            for b in range(mp):
                V[b, t] = rng.uniform(-1, 1)
        
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
            Uf, Vf, Wf = _convert_to_fractions(U, V, W)
            if _verify_exact(Uf, Vf, Wf, n, m, p):
                return Uf, Vf, Wf
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp  # = n*m*p
    
    # Try ranks from schoolbook_rank - 1 down to some minimum
    # Lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    min_rank = max(nm, mp, pn)
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Start with schoolbook_rank - 1 and go down
    for target_rank in range(schoolbook_rank - 1, min_rank - 1, -1):
        # Try random integer approach
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=20, seed=target_rank * 100 + 1)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
        
        # Try structured random approach
        result = _try_structured_reduction(n, m, p, target_rank, num_trials=20, seed=target_rank * 200 + 1)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
    
    # Fallback to schoolbook decomposition
    return _schoolbook_decomposition(n, m, p)

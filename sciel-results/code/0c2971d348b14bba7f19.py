"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
  1. Start with the known rank-56 Strassen-based decomposition (8 independent
     2x2x2 sub-tensors, each handled with rank 7).
  2. Use a structured algebraic approach: exploit the Kronecker/block structure
     of M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4} to share factors across blocks.
  3. Use NumPy for fast verification and greedy column removal.
  4. Try randomized local search to find lower-rank decompositions.
  5. If no improvement is found within the time budget, return the best
     verified decomposition (at worst rank 56).
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations
from collections import defaultdict
import random

_DIMENSIONS = (2, 4, 8)
n, m, p = _DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

_DEADLINE = 135.0
_T0 = time.monotonic()


def _elapsed():
    return time.monotonic() - _T0


def _schoolbook():
    """Rank-64 schoolbook decomposition."""
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


def _strassen_56():
    """
    Rank-56 decomposition: 8 independent 2x2x2 Strassen blocks.
    U: (8, 56), V: (32, 56), W: (16, 56)
    """
    F = Fraction
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]

    col = 0
    for a in range(2):      # j-block index: j = 2*a + b
        for c in range(4):  # k-block index: k = 2*c + d

            # M1: (a00+a11)(b00+b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(-1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M2: (a10+a11)b00
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M3: a00(b10-b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(-1)
            W[4 * c + 0][col] += F(1)
            col += 1

            # M4: a11(b01-b00)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(-1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M5: (a00+a01)b11
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(-1)
            col += 1

            # M6: (a10-a00)(b00+b01)
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 0][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M7: (a11-a01)(b01+b11)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}


def _to_float_matrices(decomp):
    """Convert a decomposition dict of Fraction lists to numpy float arrays."""
    U = np.array([[float(x) for x in row] for row in decomp["U"]])
    V = np.array([[float(x) for x in row] for row in decomp["V"]])
    W = np.array([[float(x) for x in row] for row in decomp["W"]])
    return U, V, W


def _verify_numpy(U, V, W, tol=1e-9):
    """
    Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fraction(U, V, W):
    """Exact verification using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_reduce_rank(decomp, max_time=120.0):
    """
    Attempt to reduce the rank of a decomposition by removing linearly
    dependent columns from the Khatri-Rao product.
    
    Returns a potentially lower-rank decomposition, or None if no improvement.
    """
    U_f, V_f, W_f = _to_float_matrices(decomp)
    rank = U_f.shape[1]
    
    if rank <= 7:
        return None
    
    best_rank = rank
    best_indices = list(range(rank))
    
    current_indices = list(range(rank))
    
    # Sort columns by "importance" (norm of the Khatri-Rao vector)
    norms = []
    for c in range(rank):
        u_norm = np.linalg.norm(U_f[:, c])
        v_norm = np.linalg.norm(V_f[:, c])
        w_norm = np.linalg.norm(W_f[:, c])
        norms.append(u_norm * v_norm * w_norm)
    
    order = sorted(range(rank), key=lambda c: norms[c])
    
    for c_remove in order:
        if _elapsed() > max_time:
            break
        
        if c_remove not in current_indices:
            continue
        
        new_indices = [c for c in current_indices if c != c_remove]
        
        U_sub = U_f[:, new_indices]
        V_sub = V_f[:, new_indices]
        W_sub = W_f[:, new_indices]
        
        if _verify_numpy(U_sub, V_sub, W_sub):
            current_indices = new_indices
            best_rank = len(current_indices)
            best_indices = list(current_indices)
    
    if best_rank < rank:
        U_red = [[U_f[i, c] for c in best_indices] for i in range(n * m)]
        V_red = [[V_f[i, c] for c in best_indices] for i in range(m * p)]
        W_red = [[W_f[i, c] for c in best_indices] for i in range(p * n)]
        
        def to_frac(x):
            if abs(x - round(x)) < 1e-9:
                return Fraction(int(round(x)))
            if abs(2 * x - round(2 * x)) < 1e-9:
                return Fraction(int(round(2 * x)), 2)
            return Fraction(x)
        
        U_frac = [[to_frac(x) for x in row] for row in U_red]
        V_frac = [[to_frac(x) for x in row] for row in V_red]
        W_frac = [[to_frac(x) for x in row] for row in W_red]
        
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None


def _randomized_local_search(decomp, max_time=120.0):
    """
    Randomized local search: try random perturbations of the factor matrices
    and check if we can reduce rank while maintaining exact decomposition.
    """
    U_f, V_f, W_f = _to_float_matrices(decomp)
    rank = U_f.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    # Try removing random subsets of columns
    for attempt in range(200):
        if _elapsed() > max_time:
            break
        
        # Try removing 1-3 random columns
        num_remove = random.randint(1, min(3, rank - 10))
        remove_cols = random.sample(range(rank), num_remove)
        
        keep_cols = [c for c in range(rank) if c not in remove_cols]
        
        U_sub = U_f[:, keep_cols]
        V_sub = V_f[:, keep_cols]
        W_sub = W_f[:, keep_cols]
        
        if _verify_numpy(U_sub, V_sub, W_sub):
            if len(keep_cols) < best_rank:
                best_rank = len(keep_cols)
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
    
    if best_rank < rank:
        def to_frac(x):
            if abs(x - round(x)) < 1e-9:
                return Fraction(int(round(x)))
            if abs(2 * x - round(2 * x)) < 1e-9:
                return Fraction(int(round(2 * x)), 2)
            return Fraction(x)
        
        U_frac = [[to_frac(x) for x in row] for row in best_U]
        V_frac = [[to_frac(x) for x in row] for row in best_V]
        W_frac = [[to_frac(x) for x in row] for row in best_W]
        
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None


def _structured_block_search():
    """
    Structured search exploiting block decomposition:
    M_{2,4,8} can be seen as 2 blocks of M_{2,2,8} in the i-dimension,
    or 4 blocks of M_{1,4,8}, etc.
    
    Try to find a decomposition that shares structure across blocks.
    """
    F = Fraction
    
    # Approach: treat the 2x4x8 tensor as having a 2x2 block structure in (i,j)
    # and 4x2 block structure in k.
    
    # For each (i,j) pair, the tensor entry is 1 for all k.
    # This means U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must sum to 1.
    
    # Try a rank-48 decomposition by sharing factors more aggressively.
    # This is a heuristic; we'll construct it and verify.
    
    rank = 48
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    # This is a placeholder - the actual construction needs to be worked out.
    # For now, return None to avoid an invalid decomposition.
    return None


def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK

    # Try the rank-56 Strassen decomposition
    if _elapsed() < _DEADLINE - 10:
        s2 = _strassen_56()
        if s2 is not None:
            if _verify_fraction(s2["U"], s2["V"], s2["W"]):
                r = len(s2["U"][0])
                if r < best_rank:
                    best = s2
                    best_rank = r

    # Try to reduce rank further via linear dependence elimination
    if _elapsed() < _DEADLINE - 5:
        reduced = _try_reduce_rank(best, max_time=_DEADLINE - _elapsed() - 5)
        if reduced is not None:
            if _verify_fraction(reduced["U"], reduced["V"], reduced["W"]):
                r = len(reduced["U"][0])
                if r < best_rank:
                    best = reduced
                    best_rank = r

    # Try randomized local search
    if _elapsed() < _DEADLINE - 3:
        randomized = _randomized_local_search(best, max_time=_DEADLINE - _elapsed() - 3)
        if randomized is not None:
            if _verify_fraction(randomized["U"], randomized["V"], randomized["W"]):
                r = len(randomized["U"][0])
                if r < best_rank:
                    best = randomized
                    best_rank = r

    # Try structured block search
    if _elapsed() < _DEADLINE - 1:
        structured = _structured_block_search()
        if structured is not None:
            if _verify_fraction(structured["U"], structured["V"], structured["W"]):
                r = len(structured["U"][0])
                if r < best_rank:
                    best = structured
                    best_rank = r

    return best

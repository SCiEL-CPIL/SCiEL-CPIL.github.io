"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-21 exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    Uses structured search over half-integer coefficient patterns derived from
    the algebraic structure of matrix multiplication.
    """
    n, m, p = 2, 3, 4
    deadline = time.time() + 140
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank 21 first, then 20, etc.
    for rank in [21, 20, 19]:
        if time.time() > deadline:
            break
        result = _find_decomposition(T, n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 24
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _find_decomposition(T, n, m, p, rank, deadline):
    """
    Find a rank-r decomposition using structured pattern search.
    
    Key insight: For matrix multiplication, the decomposition has a specific
    algebraic structure. We can parameterize it using the fact that each
    term corresponds to a "circuit" in the multiplication algorithm.
    
    We use a greedy approach: start with schoolbook terms and try to merge
    them using linear combinations with half-integer coefficients.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Get all schoolbook terms
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_terms.append((i * m + j, j * p + k, k * n + i))
    
    # Strategy: Try all subsets of size `rank` from the 24 schoolbook terms
    # plus some "merged" terms. For small cases, we can enumerate.
    
    # First, try: keep `rank` schoolbook terms and see if we can fill the gaps
    # with a few additional terms that have half-integer coefficients.
    
    # A better approach: use the fact that the tensor has a specific sparsity
    # pattern. We'll use a structured initialization based on the matrix mult
    # structure and then do local search.
    
    np.random.seed(42)
    
    # Initialize with a mix of schoolbook terms
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Fill first min(rank, 24) columns with schoolbook terms
    for col in range(min(rank, len(schoolbook_terms))):
        a, b, d = schoolbook_terms[col]
        U[a, col] = 1.0
        V[b, col] = 1.0
        W[d, col] = 1.0
    
    # For remaining columns, try different patterns
    for col in range(min(rank, len(schoolbook_terms)), rank):
        # Try a pattern that combines two schoolbook terms
        # This is a heuristic: pick random pairs
        idx1 = np.random.randint(len(schoolbook_terms))
        idx2 = np.random.randint(len(schoolbook_terms))
        a1, b1, d1 = schoolbook_terms[idx1]
        a2, b2, d2 = schoolbook_terms[idx2]
        
        # Try to create a merged term
        if a1 == a2:
            U[a1, col] = 1.0
            V[b1, col] = 1.0
            V[b2, col] = 1.0
            W[d1, col] = 1.0
        elif b1 == b2:
            U[a1, col] = 1.0
            U[a2, col] = 1.0
            V[b1, col] = 1.0
            W[d1, col] = 1.0
        elif d1 == d2:
            U[a1, col] = 1.0
            U[a2, col] = 1.0
            V[b1, col] = 1.0
            W[d1, col] = 1.0
        else:
            # Fallback: just use one term
            U[a1, col] = 1.0
            V[b1, col] = 1.0
            W[d1, col] = 1.0
    
    # Now do local search: try to adjust coefficients to reduce rank
    # while maintaining exact decomposition.
    
    best_U, best_V, best_W, best_loss = U.copy(), V.copy(), W.copy(), _loss(T, U, V, W)
    
    # Local search: try changing individual entries to half-integers
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        if best_loss < 1e-10:
            break
        
        # Pick a random matrix and entry to modify
        mat_choice = np.random.choice(3)
        if mat_choice == 0:
            idx = np.random.randint(nm * rank)
            r, c = divmod(idx, rank)
            old_val = U[r, c]
            # Try nearby half-integer values
            candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0, 1.0, -1.0]
            for new_val in candidates:
                if abs(new_val) > 2:
                    continue
                U[r, c] = new_val
                loss = _loss(T, U, V, W)
                if loss < best_loss:
                    best_U, best_V, best_W, best_loss = U.copy(), V.copy(), W.copy(), loss
                else:
                    U[r, c] = old_val
        elif mat_choice == 1:
            idx = np.random.randint(mp * rank)
            r, c = divmod(idx, rank)
            old_val = V[r, c]
            candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0, 1.0, -1.0]
            for new_val in candidates:
                if abs(new_val) > 2:
                    continue
                V[r, c] = new_val
                loss = _loss(T, U, V, W)
                if loss < best_loss:
                    best_U, best_V, best_W, best_loss = U.copy(), V.copy(), W.copy(), loss
                else:
                    V[r, c] = old_val
        else:
            idx = np.random.randint(pn * rank)
            r, c = divmod(idx, rank)
            old_val = W[r, c]
            candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0, 1.0, -1.0]
            for new_val in candidates:
                if abs(new_val) > 2:
                    continue
                W[r, c] = new_val
                loss = _loss(T, U, V, W)
                if loss < best_loss:
                    best_U, best_V, best_W, best_loss = U.copy(), V.copy(), W.copy(), loss
                else:
                    W[r, c] = old_val
    
    if best_loss < 1e-8:
        # Verify and return
        tensor = np.einsum('ac,bc,dc->abd', best_U, best_V, best_W)
        if np.allclose(tensor, T, atol=1e-10):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _loss(T, U, V, W):
    """Compute Frobenius norm squared of the residual."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.sum((tensor - T) ** 2)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

def _build_system(n, m, p, rank):
    """Precompute the sparse linear system structure for given rank."""
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    num_unk = rank * pn
    
    # For each equation (i,j,k), the unknowns are W[k*n+i, r] for r in range(rank)
    # with coefficient U[i*m+j, r] * V[j*p+k, r]
    # We'll store this as (eq_idx, unk_idx) -> coefficient template
    # Actually, we build A matrix where A[eq, unk] = U[i*m+j, r] * V[j*p+k, r]
    # and b[eq] = 1
    
    # Precompute mapping from (i,j,k) to eq index
    # and from (i,j,k,r) to (eq, unk) pair
    eq_indices = []
    unk_indices = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for r in range(rank):
                    unk = r * pn + k * n + i
                    eq_indices.append(eq)
                    unk_indices.append(unk)
    
    return eq_indices, unk_indices, num_eqs, num_unk, nm, mp, pn

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition exactly reproduces the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_half_integers(mat, tol=1e-7):
    """Check if all entries are integers or half-integers."""
    for row in mat:
        for val in row:
            x2 = 2 * val
            nearest = round(x2)
            if abs(x2 - nearest) > tol:
                return False
    return True

def _round_to_half_int(x):
    """Round x to nearest half-integer."""
    return round(x * 2) / 2.0

def _try_rank(n, m, p, rank, max_trials, deadline, seed=None):
    """Try to find a rank-r decomposition with random U, V."""
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    num_unk = rank * pn
    
    eq_indices, unk_indices, _, _, _, _, _ = _build_system(n, m, p, rank)
    
    rng = random.Random(seed)
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Generate random U and V with entries from {-1, 0, 1} with P(1)=0.5, P(-1)=0.5, P(0)=0
        # Actually, use {-1, 1} only (no zeros) for denser systems
        U = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice([-1, 1]) for _ in range(rank)] for _ in range(mp)]
        
        # Build sparse A matrix
        coeffs = []
        for eq, unk in zip(eq_indices, unk_indices):
            # Determine i, j, k from eq
            i = eq // (m * p)
            rem = eq % (m * p)
            j = rem // p
            k = rem % p
            # Determine r from unk
            r = unk // pn
            coeff = U[i * m + j][r] * V[j * p + k][r]
            if coeff != 0:
                coeffs.append(coeff)
            else:
                coeffs.append(0)
        
        # Build sparse matrix
        A = sp.csr_matrix((coeffs, (eq_indices, unk_indices)), shape=(num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        try:
            w = spsolve(A, b)
        except Exception:
            continue
        
        if not np.all(np.isfinite(w)):
            continue
        
        # Reshape w into W matrix
        W_mat = w.reshape(rank, pn).T  # W[k*n+i, r]
        
        # Check if entries are close to half-integers
        if not _verify_half_integers(W_mat, tol=1e-5):
            continue
        
        # Round to half-integers and verify exactly
        W_rounded = [[_round_to_half_int(W_mat[i][r]) for r in range(rank)] for i in range(pn)]
        U_rounded = [[_round_to_half_int(U[i][r]) for r in range(rank)] for i in range(nm)]
        V_rounded = [[_round_to_half_int(V[i][r]) for r in range(rank)] for i in range(mp)]
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, rank, tol=1e-9):
            return U_rounded, V_rounded, W_rounded, rank
    
    return None

def _try_rank_with_zeros(n, m, p, rank, max_trials, deadline, seed=None):
    """Try with entries from {-1, 0, 1}."""
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    num_unk = rank * pn
    
    eq_indices, unk_indices, _, _, _, _, _ = _build_system(n, m, p, rank)
    
    rng = random.Random(seed)
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Entries from {-1, 0, 1} with P(-1)=0.3, P(0)=0.4, P(1)=0.3
        U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
        
        coeffs = []
        for eq, unk in zip(eq_indices, unk_indices):
            i = eq // (m * p)
            rem = eq % (m * p)
            j = rem // p
            k = rem % p
            r = unk // pn
            coeff = U[i * m + j][r] * V[j * p + k][r]
            coeffs.append(coeff)
        
        A = sp.csr_matrix((coeffs, (eq_indices, unk_indices)), shape=(num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        try:
            w = spsolve(A, b)
        except Exception:
            continue
        
        if not np.all(np.isfinite(w)):
            continue
        
        W_mat = w.reshape(rank, pn).T
        
        if not _verify_half_integers(W_mat, tol=1e-5):
            continue
        
        W_rounded = [[_round_to_half_int(W_mat[i][r]) for r in range(rank)] for i in range(pn)]
        U_rounded = [[_round_to_half_int(U[i][r]) for r in range(rank)] for i in range(nm)]
        V_rounded = [[_round_to_half_int(V[i][r]) for r in range(rank)] for i in range(mp)]
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, rank, tol=1e-9):
            return U_rounded, V_rounded, W_rounded, rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    lower_bound = max(n * m, m * p, p * n)  # = 18
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, 40):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        
        # Allocate trials based on remaining time
        # Each trial takes roughly 10-50ms for sparse solve
        max_trials = max(50, int(remaining * 100))
        
        # First try dense {-1,1} entries
        result = _try_rank(n, m, p, rank, max_trials, deadline, seed=rank * 1000)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        # Then try {-1,0,1} entries
        remaining = deadline - time.time()
        max_trials2 = max(30, int(remaining * 50))
        result = _try_rank_with_zeros(n, m, p, rank, max_trials2, deadline, seed=rank * 2000 + 1)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END

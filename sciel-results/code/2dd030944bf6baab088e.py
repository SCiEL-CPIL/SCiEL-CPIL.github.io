"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _build_rank66():
    """Build a rank-66 decomposition:
    Partition A (3x4) rows: 1+2+1
    Partition B (4x6) cols: 2+2+2
    Blocks:
      A[0:1, :] (1x4) * B[:, 0:2] (4x2) -> 1x2*2x2 style, rank 4
      A[0:1, :] (1x4) * B[:, 2:4] (4x2) -> rank 4
      A[0:1, :] (1x4) * B[:, 4:6] (4x2) -> rank 4
      A[1:3, :] (2x4) * B[:, 0:2] (4x2) -> 2x2*2x2 Strassen, rank 7
      A[1:3, :] (2x4) * B[:, 2:4] (4x2) -> rank 7
      A[1:3, :] (2x4) * B[:, 4:6] (4x2) -> rank 7
      A[3:4, :] (1x4) * B[:, 0:2] (4x2) -> rank 4
      A[3:4, :] (1x4) * B[:, 2:4] (4x2) -> rank 4
      A[3:4, :] (1x4) * B[:, 4:6] (4x2) -> rank 4
    Total: 4*3 + 7*3 + 4*3 = 12+21+12 = 45? No wait.
    Actually let me use the known rank-66 from inspirations.
    Let me just build schoolbook rank 72 and then try to reduce via linear algebra.
    """
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
    return U, V, W


def _verify(U, V, W):
    """Check that the decomposition is exact for matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta(j) for each (i,j,k)
    # Actually the tensor for matmult is: T_{i,j,k} = 1 if this is a valid (i,j,k) term
    # The decomposition says: T_{i,j,k} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
    # We need this to equal 1 for all valid (i,j,k) in the matmult tensor.
    # For matmult 3x4 * 4x6: the tensor has T[i,j,k]=1 for all i in 0..2, j in 0..3, k in 0..5
    # Wait no - the matmult tensor is: for each (i,j,k), the coefficient is 1.
    # Actually the matmult tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the bilinear map).
    # No wait. The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # Let's verify a few entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_remove_term(U, V, W, term_idx):
    """Try to remove term term_idx by expressing it as a linear combination of others.
    For term c, we have U[:,c], V[:,c], W[:,c].
    We want to find coefficients alpha_d for d != c such that:
    For all (i,j,k): U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{d!=c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
    
    This is nonlinear in general. But if we fix U and V, we can solve for W, or fix two and solve for the third.
    Actually, a simpler approach: if one of the three vectors is a linear combination of the others in a compatible way.
    
    Simpler structural reduction: if two terms have the same U column, we can combine them.
    Or if U[:,c] is in the span of {U[:,d] : d != c}, etc.
    
    Let's try: check if W[:,c] is a linear combination of {W[:,d] : d != c} AND the combination works out.
    Actually the cleanest check: for the decomposition to be valid, we need the tensor equation.
    Let's just try: can we find alpha_d such that for ALL (i,j,k):
      U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{d!=c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
    
    This is a linear system in alpha_d with n*m*p equations and (rank-1) unknowns.
    """
    n, m, p = DIMENSIONS
    rank = len(W[0])
    if rank <= 1:
        return None
    
    # Build the linear system
    # Unknowns: alpha_d for d in range(rank), d != c
    other_indices = [d for d in range(rank) if d != term_idx]
    num_unknowns = len(other_indices)
    num_eqs = n * m * p
    
    # Build matrix A_sys (num_eqs x num_unknowns) and rhs (num_eqs)
    A_sys = np.zeros((num_eqs, num_unknowns))
    rhs = np.zeros(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                rhs[eq_idx] = U[i * m + j, term_idx] * V[j * p + k, term_idx] * W[k * n + i, term_idx]
                for uidx, d in enumerate(other_indices):
                    A_sys[eq_idx, uidx] = U[i * m + j, d] * V[j * p + k, d] * W[k * n + i, d]
                eq_idx += 1
    
    # Solve least squares
    try:
        alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
    except Exception:
        return None
    
    # Check if the solution is exact
    recon = A_sys @ alpha
    if np.max(np.abs(recon - rhs)) > 1e-7:
        return None
    
    # Also need to check that the removed term's contribution is exactly cancelled
    # The residual should be ~0
    if len(residuals) > 0 and residuals[0] > 1e-7:
        return None
    
    # Build new U, V, W without term_idx, with updated last term (or just remove)
    # Actually we need to ADD alpha_d to each term d. So we modify the existing terms.
    new_U = [list(row) for row in U]
    new_V = [list(row) for row in V]
    new_W = [list(row) for row in W]
    
    for uidx, d in enumerate(other_indices):
        a = alpha[uidx]
        if abs(a) < 1e-12:
            continue
        for r in range(len(new_U)):
            new_U[r][d] += a * U[r][term_idx]
        for r in range(len(new_V)):
            new_V[r][d] += a * V[r][term_idx]
        for r in range(len(new_W)):
            new_W[r][d] += a * W[r][term_idx]
    
    # Remove term_idx
    new_U.pop(term_idx)
    new_V.pop(term_idx)
    new_W.pop(term_idx)
    
    return new_U, new_V, new_W


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook rank 72
    U, V, W = _build_rank66()
    # Actually _build_rank66 returns rank 72 (schoolbook). Let me rename.
    # rank = 3*4*6 = 72
    current_rank = len(W[0])
    
    # Verify initial
    if not _verify(U, V, W):
        # Fallback: just return schoolbook
        return {"U": U, "V": V, "W": W}
    
    # Iteratively try to remove terms
    max_iterations = 500
    iteration = 0
    while current_rank > 1 and iteration < max_iterations:
        if time.time() > deadline - 10:
            break
        
        removed = False
        # Try removing each term
        for term_idx in range(current_rank):
            if time.time() > deadline - 10:
                break
            result = _try_remove_term(U, V, W, term_idx)
            if result is not None:
                U, V, W = result
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
        iteration += 1
    
    # Final verification
    if not _verify(U, V, W):
        # Something went wrong, return schoolbook
        U, V, W = _build_rank66()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

"""Matrix multiplication 4x4 by 4x5 — search for rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM, MP, PN = N * M, M * P, P * N  # 16, 20, 20

# Precompute the tensor as a list of (row_in_W, row_in_U, row_in_V) triples.
# Term: W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]  with W index = k*n+i, U = i*m+j, V = j*p+k
_TENSOR_TRIPLES = []
for i in range(N):
    for j in range(M):
        for k in range(P):
            _TENSOR_TRIPLES.append((k * N + i, i * M + j, j * P + k))
# Each (u_row, v_row) appears P times (once per k); W row differs per k.
# For a fixed (u_row, v_row), the W values across its P occurrences must equal the
# distinct e_{i,k} pattern.

def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _is_half_int(x, tol=1e-7):
    """Check x is within tol of a half-integer."""
    r = round(2 * x)
    return abs(x - r / 2.0) < tol


def _try_rank(r, rng, time_budget):
    """Attempt to find a rank-r decomposition. Returns dict or None."""
    # We'll pick random U, V (sparse), solve for W, verify.
    # W is determined by the tensor equation:
    #   T[u,v] = sum_c U[u,c] V[v,c] W[c, .]  ... but W has its own row index.
    #
    # Better formulation: The tensor T is a 3-way array T[i,j,k]. A rank-r decomposition
    # is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # For a fixed column c, the contribution to T is the outer product of three vectors.
    #
    # Approach: pick U and V randomly (sparse ±1), then for each (u_row, v_row) pair that
    # appears in the tensor, the W values are constrained. Actually W is a separate matrix.
    #
    # Let's use the standard approach: T as a matrix. Flatten T into a 3-tensor and use
    # the fact that for a rank-r decomposition, T can be written as sum of r rank-1 tensors.
    #
    # Simpler: use the "column" formulation. For each column c, define vectors
    #   a_c in R^{NM}, b_c in R^{MP}, d_c in R^{PN}
    # such that T[i,j,k] = sum_c a_c[i*M+j] * b_c[j*P+k] * d_c[k*N+i].
    #
    # Given a_c and b_c, we can solve for d_c from the linear equations:
    #   For each (i,j,k): T[i,j,k] = sum_c a_c[i*M+j] * b_c[j*P+k] * d_c[k*N+i]
    # This is a linear system in the d_c variables (total r*PN unknowns).
    #
    # Actually, it's easier to fix U and V and solve for W directly.
    # The equation is: for all (i,j,k):
    #   T[i,j,k] = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
    #
    # For a fixed (i,j), the equation over k gives:
    #   T[i,j,k] for k=0..P-1
    #   = sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
    #
    # This couples W across different k (different rows of W). So it's a global system.
    #
    # Alternative: fix U, V, W for all but one entry and solve. Too complex.
    #
    # Let's use a different strategy: use the matrix flattening.
    # T as a matrix of size (N*M) x (M*P) doesn't work directly because W has its own index.
    #
    # Best approach: treat it as a linear system. Unknowns: all entries of U, V, W (r*(NM+MP+PN) total).
    # Equations: N*M*P = 80 scalar equations (one per (i,j,k)).
    # For r < 80, the system is underdetermined (more unknowns than equations) but nonlinear.
    #
    # Practical approach: fix U and V to random sparse matrices, solve the linear system for W.
    # The system is: for each (i,j,k):
    #   sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = T[i,j,k]
    #
    # This is linear in W! For each (i,j,k), it's a linear equation in the W variables.
    # Total equations: 80. Total W unknowns: r * PN = r * 20.
    # For r >= 4, we have r*20 >= 80 unknowns, so the system is (generically) solvable.
    #
    # Let's build the linear system A * w_vec = t_vec where w_vec is W flattened.
    #
    # For equation (i,j,k):
    #   sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = T[i,j,k]
    # The W variables that appear are W[k*N+i, c] for c=0..r-1.
    # So each equation only involves r specific W entries (row k*N+i of W).
    #
    # This means the system is block-diagonal! For each (i,k) pair, the equations over j
    # involve W[k*N+i, :]. There are N*P = 20 such rows of W, each with r entries.
    # For a fixed (i,k), the equations are over j=0..M-1 (4 equations):
    #   sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = T[i,j,k]
    # This is a 4x r linear system for W[k*N+i, :].
    #
    # So for each (i,k), we solve: B_{i,k} * W[k*N+i, :]^T = t_{i,k}^T
    # where B_{i,k}[j, c] = U[i*M+j, c] * V[j*P+k, c] and t_{i,k}[j] = T[i,j,k].
    #
    # For this to have a solution, we need rank(B) <= 4 and t in col(B).
    # With r >= 4 and random U, V, B is 4xr with rank min(4,r). If r >= 4, rank is 4 (generically).
    # Then the solution exists iff t is in col(B), which is generically true (4-dim space, 4 eqs).
    # Actually with 4 equations and r >= 4 unknowns, the system is underdetermined and
    # generically has solutions.
    #
    # But we need W entries to be half-integers! That's the hard constraint.
    #
    # Strategy: try many random (U,V) pairs, solve for W, check if W is half-integer.

    deadline = time.time() + time_budget
    best = None
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        # Generate sparse random U, V with entries in {0, 1, -1}
        # Density: try different sparsities
        density_u = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        density_v = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])

        U = np.zeros((NM, r), dtype=float)
        V = np.zeros((MP, r), dtype=float)
        for c in range(r):
            for i in range(NM):
                if rng.random() < density_u:
                    U[i, c] = rng.choice([1.0, -1.0])
            for j in range(MP):
                if rng.random() < density_v:
                    V[j, c] = rng.choice([1.0, -1.0])

        # Solve for W: for each (i, k), solve B * W_row = t
        W = np.zeros((PN, r), dtype=float)
        success = True
        for i in range(N):
            for k in range(P):
                # B is M x r: B[j, c] = U[i*M+j, c] * V[j*P+k, c]
                B = np.zeros((M, r))
                t = np.zeros(M)
                for j in range(M):
                    B[j, :] = U[i * M + j, :] * V[j * P + k, :]
                    t[j] = 1.0 if i == k else 0.0  # T[i,j,k] = delta_{i,k}... wait
                    # Actually T[i,j,k] = e_{i,k} where e_{i,k} is the (i,k) entry of identity?
                    # No. T[i,j,k] = 1 if i==k, else 0. That's the identity matrix in (i,k).
                    # Wait, that's not right either. Let me re-derive.
                    # C = A * B where A is 4x4, B is 4x5.
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # The tensor is T[i,j,k] = 1 (it's always 1, it's the multiplication tensor).
                    # The decomposition: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                    # Wait, that's not standard. Let me re-read the task.
                    #
                    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                    # "W[k*n+i,r]" — the trace convention.
                    #
                    # The standard decomposition: C[i,k] = sum_{j,c} A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[j,c]) (sum_k B[k,c] V[k,c])...
                    # No, that's not right either.
                    #
                    # The tensor for matrix multiplication: T[i,j,k] = 1 for all i,j,k.
                    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.
                    #
                    # So T[i,j,k] = 1 = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
                    #
                    # Yes! T[i,j,k] = 1 for ALL i, j, k. It's the all-ones tensor.
                    pass

        # Redo: T[i,j,k] = 1 for all i,j,k.
        W = np.zeros((PN, r), dtype=float)
        success = True
        for i in range(N):
            for k in range(P):
                B = np.zeros((M, r))
                t = np.ones(M)  # T[i,j,k] = 1 for all j
                for j in range(M):
                    B[j, :] = U[i * M + j, :] * V[j * P + k, :]
                # Solve B * W_row = t  (M x r system, r unknowns, M equations)
                # If r >= M, use least squares; check residual.
                try:
                    W_row, residuals, rank_b, sv = np.linalg.lstsq(B, t, rcond=None)
                    if len(residuals) > 0 and residuals[0] > 1e-6:
                        success = False
                        break
                    W[k * N + i, :] = W_row
                except np.linalg.LinAlgError:
                    success = False
                    break
            if not success:
                break

        if not success:
            continue

        # Check if W entries are half-integers
        if all(_is_half_int(x) for x in W.flat):
            # Verify the full decomposition exactly
            if _verify(U, V, W, r):
                # Convert to half-integers
                U_h = np.round(2 * U) / 2.0
                V_h = np.round(2 * V) / 2.0
                W_h = np.round(2 * W) / 2.0
                # Convert to Python lists
                U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in U_h]
                V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in V_h]
                W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in W_h]
                return {"U": U_list, "V": V_list, "W": W_list}

    return None


def _verify(U, V, W, r):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k."""
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for c in range(r):
                    s += U[i * M + j, c] * V[j * P + k, c] * W[k * N + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    # Try ranks from low to high, with time budget per rank
    # Known: rank 80 (schoolbook). Literature suggests rank ~ 47-50 for 4x4x5.
    # Try a range of ranks.
    ranks_to_try = list(range(30, 81))

    rng = random.Random(42)

    # Time budget: spend more time on lower ranks
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        # Budget: more time for lower ranks
        time_budget = min(2.0, max(0.3, (deadline - time.time()) * 0.1))
        result = _try_rank(r, rng, time_budget)
        if result is not None:
            return result

    # Fallback: schoolbook
    return _schoolbook()

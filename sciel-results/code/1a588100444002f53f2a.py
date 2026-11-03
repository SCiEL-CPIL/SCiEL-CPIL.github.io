"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x7 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 145.0  # safety margin before 150s kill


def _build_target(n, m, p):
    """Return the multiplication tensor target T of shape (n*m, m*p).
    T[a,b] = 1 if there exists (i,j,k) with a=i*m+j and b=j*p+k."""
    T = np.zeros((n * m, m * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                T[a, b] = 1.0
    return T


def _try_supports(U_sp, V_sp, T, rank):
    """Given sparse support dicts (row->list of cols) and rank, build U,V,
    solve for W, and check exactness. Returns (ok, W) or (False, None)."""
    n, m, p = T.shape[0] // m, m, T.shape[1] // m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    for row, cols in U_sp.items():
        for c in cols:
            U[row, c] = 1.0
    for row, cols in V_sp.items():
        for c in cols:
            V[row, c] = 1.0

    # M[a,b] = sum_c U[a,c] V[b,c]
    M = U @ V.T  # shape (n*m, m*p)
    # We need M * W = T  =>  for each (a,b): sum_c M[a,b]... 
    # Actually: T[a,b] = sum_c U[a,c] V[b,c] W[c]  =>  T = M @ W
    # So solve M @ W = T where W is (rank,).
    # M is (n*m, m*p), W is (rank,)? No.
    # T[a,b] = sum_c U[a,c] V[b,c] W[c]
    # Let's vectorize: for each c, the matrix U[:,c] V[:,c].T contributes W[c] to T.
    # So T = sum_c W[c] * (U[:,c] @ V[:,c].T)
    # This is a linear system in W.
    # Build: for each (a,b), sum_c U[a,c] V[b,c] W[c] = T[a,b]
    # => A @ W = T_flat, where A[a*b + b_idx, c] = U[a,c] * V[b,c]
    A = np.zeros((n * m * m * p, rank))
    idx = 0
    for a in range(n * m):
        for b in range(m * p):
            A[idx, :] = U[a, :] * V[b, :]
            idx += 1
    T_flat = T.flatten()
    # Solve least squares
    try:
        W_sol, residuals, rank_r, sv = np.linalg.lstsq(A, T_flat, rcond=None)
    except np.linalg.LinAlgError:
        return False, None
    # Check exactness
    recon = (A @ W_sol).reshape(n * m, m * p)
    if np.allclose(recon, T, atol=1e-9):
        return True, W_sol
    return False, None


def _random_supports(n, m, p, rank, rng, max_per_row=3):
    """Generate random sparse support patterns for U and V."""
    U_sp = {}
    V_sp = {}
    for a in range(n * m):
        k = rng.randint(1, min(max_per_row, rank))
        cols = rng.sample(range(rank), k)
        U_sp[a] = cols
    for b in range(m * p):
        k = rng.randint(1, min(max_per_row, rank))
        cols = rng.sample(range(rank), k)
        V_sp[b] = cols
    return U_sp, V_sp


def _greedy_supports(n, m, p, rank, T, rng):
    """Greedy: assign columns to cover as many (a,b) pairs as possible."""
    U_sp = {a: [] for a in range(n * m)}
    V_sp = {b: [] for b in range(m * p)}
    # Target pairs: (a,b) where T[a,b]=1
    target_pairs = [(a, b) for a in range(n * m) for b in range(m * p) if T[a, b] == 1]
    # For each column, pick a (a,b) pair not yet covered, assign a to U and b to V
    covered = set()
    for c in range(rank):
        # Find uncovered pairs
        uncovered = [(a, b) for (a, b) in target_pairs if (a, b) not in covered]
        if not uncovered:
            break
        # Pick a pair that maximizes something; just pick random for now
        a, b = rng.choice(uncovered)
        if c not in U_sp[a]:
            U_sp[a].append(c)
        if c not in V_sp[b]:
            V_sp[b].append(c)
        covered.add((a, b))
    return U_sp, V_sp


def construct():
    n, m, p = DIMENSIONS
    T = _build_target(n, m, p)
    start = time.time()
    rng = random.Random(42)

    # Schoolbook rank as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best = (rank_sb, U_sb, V_sb, W_sb)

    # Try ranks below schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    lb = max(n * m, m * p, p * n)
    # Try from lb upward
    for rank in range(lb, rank_sb):
        if time.time() > DEADLINE - 5:
            break
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            if time.time() > DEADLINE - 5:
                break
            attempts += 1
            if rank < 30:
                U_sp, V_sp = _random_supports(n, m, p, rank, rng, max_per_row=2)
            else:
                U_sp, V_sp = _random_supports(n, m, p, rank, rng, max_per_row=3)
            ok, W_sol = _try_supports(U_sp, V_sp, T, rank)
            if ok:
                # Build full matrices
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for row, cols in U_sp.items():
                    for c in cols:
                        U[row, c] = 1.0
                for row, cols in V_sp.items():
                    for c in cols:
                        V[row, c] = 1.0
                # W is (p*n, rank)
                W = np.zeros((p * n, rank))
                # W_sol is (rank,), but we need W[k*n+i, c] = W_sol[c] for all k,i?
                # No: T[a,b] = sum_c U[a,c] V[b,c] W[c], and W has shape (p*n, rank)
                # The output index is (k*n+i), but the equation only involves sum over c.
                # Actually the tensor is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # So W[k*n+i, c] is a separate variable for each (k,i).
                # My linear system above was wrong: I assumed W[c] is shared.
                # Let me redo: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # This is a linear system in all W entries.
                # Number of W variables: p*n*rank
                # Number of equations: n*m*p
                # Let's build this properly.
                pass  # Need to fix the solver
            break  # Placeholder

    # The above approach has a flaw. Let me restart with correct linear system.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Variables: W[k*n+i, c] for all k,i,c
    # This is a big linear system. Let's solve it properly.

    # Redo search with correct model
    best = (rank_sb, U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    for rank in range(lb, rank_sb):
        if time.time() > DEADLINE - 5:
            break
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            if time.time() > DEADLINE - 5:
                break
            attempts += 1
            U_sp, V_sp = _random_supports(n, m, p, rank, rng, max_per_row=2)
            # Build U, V
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            for row, cols in U_sp.items():
                for c in cols:
                    U[row, c] = 1.0
            for row, cols in V_sp.items():
                for c in cols:
                    V[row, c] = 1.0
            # Build linear system for W
            # Variables: W[k*n+i, c], total p*n*rank
            # Equations: for each (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            nvar = p * n * rank
            neq = n * m * p
            A = np.zeros((neq, nvar))
            b = np.ones(neq)
            eq = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            val = U[i * m + j, c] * V[j * p + k, c]
                            if val != 0:
                                A[eq, k * n * rank + i * rank + c] = val
                        eq += 1
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            W = W_flat.reshape(p * n, rank)
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                best = (rank, U.tolist(), V.tolist(), W.tolist())
                break  # Found minimal for this rank, try lower

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}

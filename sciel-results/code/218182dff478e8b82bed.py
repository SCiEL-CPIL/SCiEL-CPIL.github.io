"""Maximum Cut via vectorized Parallel Tempering (Replica Exchange) + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_from_x(A, x):
    """Cut size = 0.5 * sum_u A[u] @ x[u]  (for x in {+1,-1})."""
    return 0.5 * float(np.dot(A @ x, x))

def _ls_flip(A, x, max_passes=500, time_budget=None, start_time=None):
    """Fast vectorized single-flip local search.
    Returns (x, cut, ax) where ax = A @ x (maintained incrementally)."""
    n = len(x)
    x = x.astype(np.int8).copy()
    ax = A @ x  # float array

    passes = 0
    while passes < max_passes:
        passes += 1
        # gain[u] = 2*ax[u]*x[u] ; flip if gain < 0
        gain = 2.0 * ax * x
        flips = gain < 0
        if not np.any(flips):
            break
        # Apply all beneficial flips at once
        # New ax: for each flipped u, x[u] -> -x[u]; ax changes by -2*A[:,u]*x[u]
        idx = np.where(flips)[0]
        # A[:, idx] has shape (n, k); x[idx] shape (k,)
        # ax_new = ax - 2 * (A[:, idx] @ x[idx])
        ax -= 2.0 * (A[:, idx] @ x[idx])
        x[idx] = -x[idx]
        if time_budget is not None and time.time() > time_budget:
            break
    cut = _cut_from_x(A, x)
    return x, cut, ax

def _ls_flip_2(A, x, max_passes=30, time_budget=None):
    """2-flip local search: try all pairs; flip if improves. O(n^2) per pass."""
    n = len(x)
    x = x.astype(np.int8).copy()
    ax = A @ x
    best_cut = _cut_from_x(A, x)

    passes = 0
    while passes < max_passes:
        passes += 1
        improved = False
        # For pair (i,j): flipping both changes ax by -2*(A[:,i]*x[i] + A[:,j]*x[j])
        # We evaluate gain in cut.
        # cut = 0.5 * x^T A x. Flipping i and j:
        #   new x has x_i->-x_i, x_j->-x_j
        #   delta = x^T A (dx) + dx^T A dx  where dx has -2x_i at i, -2x_j at j
        # Simpler: compute new cut directly from updated ax.
        for i in range(n):
            for j in range(i+1, n):
                # tentative
                old_xi, old_xj = x[i], x[j]
                x[i] = -old_xi
                x[j] = -old_xj
                ax_new = ax - 2.0*(A[:,i]*old_xi + A[:,j]*old_xj)
                new_cut = 0.5 * float(np.dot(ax_new, x))
                if new_cut > best_cut + 1e-9:
                    ax = ax_new
                    best_cut = new_cut
                    improved = True
                else:
                    x[i] = old_xi
                    x[j] = old_xj
        if not improved:
            break
        if time_budget is not None and time.time() > time_budget:
            break
    return x, best_cut, ax

def _spectral_init(A):
    """Spectral (Fiedler) initialization. Returns x in {+1,-1}."""
    n = len(A)
    deg = A.sum(axis=1)
    D = np.diag(deg)
    L = D - A
    try:
        w, V = np.linalg.eigh(L)
        fiedler = V[:, 1]
        x = np.where(fiedler >= 0, 1, -1).astype(np.int8)
    except Exception:
        x = np.random.choice([-1, 1], n).astype(np.int8)
    return x

def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    A_in = np.asarray(adjacency)
    n = A_in.shape[0]
    # Symmetrize and binarize
    A = ((A_in + A_in.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    best_x = None
    best_cut = -1.0

    def _consider(x, cut):
        nonlocal best_x, best_cut
        if cut > best_cut + 1e-9:
            best_cut = cut
            best_x = x.copy()

    # --- Phase 1: Parallel Tempering ---
    n_replicas = 8
    # Temperatures: T0=0 (greedy), increasing
    base_T = 0.5
    T_list = [0.0] + [base_T * (2.0 ** i) for i in range(n_replicas - 1)]
    # T_list[0]=0 means pure greedy for that replica

    # Initialize replicas: one spectral, rest random
    replicas = []
    cuts = []
    axes = []
    for r in range(n_replicas):
        if r == 0:
            x = _spectral_init(A)
        else:
            x = np.random.choice([-1, 1], n).astype(np.int8)
        # Short local search to settle
        x, cut, ax = _ls_flip(A, x, max_passes=100, time_budget=deadline - 1.0)
        replicas.append(x)
        cuts.append(cut)
        axes.append(ax)
        _consider(x, cut)

    # Parallel tempering loop
    rng = np.random
    pt_passes = 0
    max_pt_passes = 2000
    while pt_passes < max_pt_passes:
        pt_passes += 1
        if time.time() > deadline - 30.0:
            break

        # Each replica does a few LS steps (vectorized)
        for r in range(n_replicas):
            x, cut, ax = _ls_flip(A, replicas[r], max_passes=20, time_budget=deadline - 30.0)
            replicas[r] = x
            cuts[r] = cut
            axes[r] = ax
            _consider(x, cut)

        # Replica exchange: try swapping adjacent replicas
        for r in range(n_replicas - 1):
            r1, r2 = r, r + 1
            T1, T2 = T_list[r1], T_list[r2]
            E1, E2 = cuts[r1], cuts[r2]
            # Energy difference for swap: (E1 - E2)*(1/T2 - 1/T1)  (with T=0 handling)
            if T1 == 0.0 and T2 == 0.0:
                continue
            if T1 == 0.0:
                # Replica 1 is greedy; only swap if it strictly improves
                if E2 > E1:
                    # swap
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    cuts[r1], cuts[r2] = cuts[r2], cuts[r1]
                    axes[r1], axes[r2] = axes[r2], axes[r1]
                    _consider(replicas[r1], cuts[r1])
                    _consider(replicas[r2], cuts[r2])
                continue
            if T2 == 0.0:
                if E1 > E2:
                    replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                    cuts[r1], cuts[r2] = cuts[r2], cuts[r1]
                    axes[r1], axes[r2] = axes[r2], axes[r1]
                    _consider(replicas[r1], cuts[r1])
                    _consider(replicas[r2], cuts[r2])
                continue
            # Both T > 0
            dE = (E1 - E2) * (1.0 / T2 - 1.0 / T1)
            if dE <= 0 or rng.random() < np.exp(-dE):
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                cuts[r1], cuts[r2] = cuts[r2], cuts[r1]
                axes[r1], axes[r2] = axes[r2], axes[r1]
                _consider(replicas[r1], cuts[r1])
                _consider(replicas[r2], cuts[r2])

    # --- Phase 2: Polish the best with 2-flip moves ---
    if best_x is not None and time.time() < deadline - 5.0:
        x2, cut2, _ = _ls_flip_2(A, best_x, max_passes=20, time_budget=deadline - 2.0)
        _consider(x2, cut2)
        # Final single-flip polish
        x3, cut3, _ = _ls_flip(A, best_x, max_passes=200, time_budget=deadline - 1.0)
        _consider(x3, cut3)

    # Convert x in {+1,-1} to labels in {0,1}
    if best_x is None:
        best_x = np.random.choice([0, 1], n).astype(np.int8)
        return best_x
    labels = ((best_x == 1).astype(np.int8))
    return labels
# EVOLVE-BLOCK-END

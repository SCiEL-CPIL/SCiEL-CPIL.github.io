"""Spectral-initialized Iterated Local Search for QAP wil100."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def _compute_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align high-flow facilities with low-distance
    locations using eigenvector-based rank matching, with random perturbation.
    """
    # Eigenvectors of flow matrix (largest magnitude)
    try:
        va, _ = eigsh(a, k=min(10, n - 1), which='LM')
        fa = np.abs(va).sum(axis=1)
    except Exception:
        fa = a.sum(axis=1)
    try:
        vb, _ = eigsh(b, k=min(10, n - 1), which='LM')
        fb = np.abs(vb).sum(axis=1)
    except Exception:
        fb = b.sum(axis=1)

    # Rank facilities by flow importance (descending)
    # Rank locations by distance centrality (ascending = central = good)
    fac_rank = np.argsort(-fa, kind='stable')
    loc_rank = np.argsort(fb, kind='stable')

    p = np.empty(n, dtype=np.int64)
    p[fac_rank] = loc_rank

    # Add small random perturbation to break ties and diversity
    for _ in range(3):
        i = rng.integers(n)
        j = rng.integers(n)
        p[i], p[j] = p[j], p[i]

    return p


def _delta_2opt(a, b, p, i, j):
    """
    Compute cost delta for swapping facilities i and j in permutation p.
    Let x = p[i], y = p[j].
    After swap: p[i]=y, p[j]=x.
    
    Delta = sum over all k != i,j of:
      a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
      + a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y])
    Plus the (i,j) and (j,i) terms which change:
      old: a[i,j]*b[x,y] + a[j,i]*b[y,x]
      new: a[i,j]*b[y,x] + a[j,i]*b[x,y]
    """
    x = p[i]
    y = p[j]
    
    # Vectorized delta computation
    # For all k != i, j:
    # contribution from i: a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
    # contribution from j: a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y])
    
    # Build index arrays excluding i and j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    pk = p[ks]
    
    # b[y, pk] - b[x, pk]
    diff_yx = b[y, pk] - b[x, pk]
    # b[pk, y] - b[pk, x]
    diff_xy = b[pk, y] - b[pk, x]
    
    delta = (a[i, ks] * diff_yx + a[ks, i] * diff_xy +
             a[j, ks] * (-diff_yx) + a[ks, j] * (-diff_xy))
    
    # Self terms (i,j) and (j,i)
    old_ij = a[i, j] * b[x, y] + a[j, i] * b[y, x]
    new_ij = a[i, j] * b[y, x] + a[j, i] * b[x, y]
    delta += new_ij - old_ij
    
    return int(delta)


def _local_search_2opt(a, b, p, deadline, rng, max_passes=50):
    """
    Vectorized 2-opt local search. Returns improved permutation and cost.
    Uses first-improvement with random restart within passes.
    """
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    no_improve = 0
    max_no_improve = 3
    
    while time.time() < deadline:
        improved = False
        for _ in range(max_passes):
            if time.time() >= deadline:
                break
            # Randomize starting point for this pass
            start = rng.integers(n)
            for i in range(n):
                if time.time() >= deadline:
                    break
                ii = (start + i) % n
                # Try swapping ii with a random set of partners
                partners = rng.permutation(n)
                for j in partners:
                    if j == ii:
                        continue
                    delta = _delta_2opt(a, b, p, ii, int(j))
                    if delta < 0:
                        p[ii], p[j] = p[j], p[ii]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        break  # first improvement, move on
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
        if not improved:
            break
    
    return best_p, best_cost


def _perturb(p, n, strength, rng):
    """Apply random perturbation (multiple swaps) to escape local optima."""
    p = p.copy()
    for _ in range(strength):
        i = rng.integers(n)
        j = rng.integers(n)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]]."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral initialization
    p = _spectral_init(a, b, n, rng)
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Local search from spectral start
    if time.time() < deadline - 10.0:
        p, current_cost = _local_search_2opt(a, b, p, deadline - 10.0, rng)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 3: Iterated Local Search
    strength = 2
    stall_count = 0
    max_stall = 5
    
    while time.time() < deadline - 5.0:
        # Perturb current best
        p_pert = _perturb(best_p, n, strength, rng)
        
        # Local search from perturbed solution
        p_ls, cost_ls = _local_search_2opt(a, b, p_pert, deadline - 5.0, rng)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            stall_count = 0
            # Reduce perturbation strength on improvement
            strength = max(1, strength - 1)
        else:
            stall_count += 1
            # Increase perturbation strength on stall
            strength = min(8, strength + 1)
        
        if stall_count >= max_stall:
            # Try a fresh spectral initialization
            if time.time() < deadline - 15.0:
                p_fresh = _spectral_init(a, b, n, rng)
                p_ls, cost_ls = _local_search_2opt(a, b, p_fresh, deadline - 5.0, rng)
                if cost_ls < best_cost:
                    best_cost = cost_ls
                    best_p = p_ls.copy()
                    stall_count = 0
                    strength = 2
                else:
                    stall_count = 0  # reset to avoid infinite loop
                    strength = 3
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def _cyclic3_delta(a, b, p, i, j, k):
    """
    Delta for cyclic permutation: p[i]->p[j], p[j]->p[k], p[k]->p[i].
    New positions: p'[i]=p[j], p'[j]=p[k], p'[k]=p[i].
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    
    # For each facility f, its old location is p[f], new location is p'[f]
    # Delta = sum_f a[f,:] * (b[new_loc_f, :] - b[old_loc_f, :])
    #        + sum_f a[:,f] * (b[:, new_loc_f] - b[:, old_loc_f])
    #        - cross terms handled by the full formula
    
    # Vectorized: compute the change in cost
    # Cost = sum_{f,g} a[f,g] * b[p[f], p[g]]
    # New cost = sum_{f,g} a[f,g] * b[p'[f], p[g']]
    
    # Change only involves facilities i, j, k and their interactions with all others.
    # Let's compute:
    # delta = sum_g a[i,g]*(b[pj,p[g]]-b[pi,p[g]]) + a[j,g]*(b[pk,p[g]]-b[pj,p[g]]) + a[k,g]*(b[pi,p[g]]-b[pk,p[g]])
    #       + sum_f a[f,i]*(b[p[f],pj]-b[p[f],pi]) + a[f,j]*(b[p[f],pk]-b[p[f],pj]) + a[f,k]*(b[p[f],pi]-b[p[f],pk])
    #       - correction for double-counted (i,j),(j,k),(k,i) pairs
    
    # More efficient: compute the full contribution change
    # Old contributions involving i,j,k:
    # sum_g a[i,g]*b[pi,p[g]] + sum_g a[j,g]*b[pj,p[g]] + sum_g a[k,g]*b[pk,p[g]]
    # sum_f a[f,i]*b[p[f],pi] + sum_f a[f,j]*b[p[f],pj] + sum_f a[f,k]*b[p[f],pk]
    # (but a[i,i]*b[pi,pi] etc are included in both, so we need to be careful)
    
    # Let me just compute old and new costs for the affected subset
    # Actually, let's use the vectorized approach properly.
    
    # The change in total cost when we cycle (i,j,k):
    # new_p[i] = pj, new_p[j] = pk, new_p[k] = pi
    
    # Delta = sum_{f,g} a[f,g] * (b[new_p[f], new_p[g]] - b[p[f], p[g]])
    # Only terms where f in {i,j,k} or g in {i,j,k} change.
    
    # Let S = {i, j, k}
    # Delta = sum_{f in S, g} a[f,g]*(b[new_p[f],p[g]] - b[p[f],p[g]])
    #       + sum_{f, g in S} a[f,g]*(b[p[f],new_p[g]] - b[p[f],p[g]])
    #       - sum_{f in S, g in S} a[f,g]*(b[new_p[f],new_p[g]] - b[p[f],p[g]])  # subtract double count
    # Wait, that's not right either. Let me just do it directly.
    
    # Simplest vectorized approach:
    # delta_row = sum over g of a[i,g]*(b[pj,p[g]]-b[pi,p[g]]) + a[j,g]*(b[pk,p[g]]-b[pj,p[g]]) + a[k,g]*(b[pi,p[g]]-b[pk,p[g]])
    # delta_col = sum over f of a[f,i]*(b[p[f],pj]-b[p[f],pi]) + a[f,j]*(b[p[f],pk]-b[p[f],pj]) + a[f,k]*(b[p[f],pi]-b[p[f],pk])
    # But this double-counts the (i,j), (j,k), (k,i) interactions in both row and col sums.
    # The (i,j) term: in row sum: a[i,j]*(b[pj,pj]-b[pi,pj]), in col sum: a[i,j]*(b[pj,pj]-b[pi,pj]) -- wait no.
    
    # Let me be more careful. The total cost is:
    # C = sum_{f,g} a[f,g] * b[p[f], p[g]]
    
    # When we change p[i], p[j], p[k], the terms that change are:
    # 1. Terms where f=i or f=j or f=k (row terms)
    # 2. Terms where g=i or g=j or g=k (col terms)
    # But terms where both f and g are in {i,j,k} are counted in both, so subtract once.
    
    # Row contribution change:
    # For f=i: sum_g a[i,g]*(b[pj,p[g]] - b[pi,p[g]])
    # For f=j: sum_g a[j,g]*(b[pk,p[g]] - b[pj,p[g]])
    # For f=k: sum_g a[k,g]*(b[pi,p[g]] - b[pk,p[g]])
    
    # Col contribution change:
    # For g=i: sum_f a[f,i]*(b[p[f],pj] - b[p[f],pi])
    # For g=j: sum_f a[f,j]*(b[p[f],pk] - b[p[f],pj])
    # For g=k: sum_f a[f,k]*(b[p[f],pi] - b[p[f],pk])
    
    # Double-counted (both f and g in {i,j,k}):
    # (i,i): a[i,i]*(b[pj,pj]-b[pi,pj]) + a[i,i]*(b[pj,pj]-b[pi,pj]) - a[i,i]*(b[pj,pj]-b[pi,pi])
    # This is getting complicated. Let me just compute it directly for the 3x3 submatrix plus interactions with rest.
    
    # Actually, the cleanest vectorized way:
    # Let old_p = p, new_p = p with cycle applied
    # delta = sum_{f,g} a[f,g] * (b[new_p[f], new_p[g]] - b[old_p[f], old_p[g]])
    # But this is O(n^2) to compute fully.
    
    # Better: decompose
    # delta = sum_{f in S} sum_g a[f,g]*(b[new_p[f],p[g]]-b[p[f],p[g]])
    #       + sum_g sum_{f in S} a[f,g]*(b[p[f],new_p[g]]-b[p[f],p[g]])
    #       - sum_{f in S} sum_{g in S} a[f,g]*(b[new_p[f],new_p[g]]-b[p[f],p[g]])
    # Wait no, that over-subtracts. Let me think again.
    
    # C_new - C_old = sum_{f,g} a[f,g]*b[new_p[f],new_p[g]] - sum_{f,g} a[f,g]*b[p[f],p[g]]
    
    # = sum_{f,g} a[f,g]*(b[new_p[f],new_p[g]] - b[p[f],p[g]])
    
    # Split into 4 cases: f in S or not, g in S or not.
    # Case 1: f not in S, g not in S: b[new_p[f],new_p[g]] = b[p[f],p[g]], contribution 0.
    # Case 2: f in S, g not in S: sum_{f in S, g not in S} a[f,g]*(b[new_p[f],p[g]]-b[p[f],p[g]])
    # Case 3: f not in S, g in S: sum_{f not in S, g in S} a[f,g]*(b[p[f],new_p[g]]-b[p[f],p[g]])
    # Case 4: f in S, g in S: sum_{f in S, g in S} a[f,g]*(b[new_p[f],new_p[g]]-b[p[f],p[g]])
    
    # So:
    # delta = sum_{f in S} a[f, :][~in_S] * (b[new_p[f], :][~in_S] - b[p[f], :][~in_S])
    #       + sum_{g in S} a[:, g][:~in_S] * (b[:, new_p[g]][:~in_S] - b[:, p[g]][:~in_S])
    #       + sum_{f in S, g in S} a[f,g]*(b[new_p[f],new_p[g]]-b[p[f],p[g]])
    
    # This is O(n) for the first two terms (sum over ~n elements for 3 facilities) and O(1) for the last.
    
    S = np.array([i, j, k])
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    not_S = ~in_S
    
    new_pi = pj
    new_pj = pk
    new_pk = pi
    
    # Term 2: f in S, g not in S
    # For f=i: a[i, not_S] @ (b[pj, not_S] - b[pi, not_S])
    # For f=j: a[j, not_S] @ (b[pk, not_S] - b[pj, not_S])
    # For f=k: a[k, not_S] @ (b[pi, not_S] - b[pk, not_S])
    t2 = (np.dot(a[i, not_S], b[pj, not_S] - b[pi, not_S]) +
          np.dot(a[j, not_S], b[pk, not_S] - b[pj, not_S]) +
          np.dot(a[k, not_S], b[pi, not_S] - b[pk, not_S]))
    
    # Term 3: f not in S, g in S
    # For g=i: a[not_S, i] @ (b[not_S, pj] - b[not_S, pi])
    # For g=j: a[not_S, j] @ (b[not_S, pk] - b[not_S, pj])
    # For g=k: a[not_S, k] @ (b[not_S, pi] - b[not_S, pk])
    t3 = (np.dot(a[not_S, i], b[not_S, pj] - b[not_S, pi]) +
          np.dot(a[not_S, j], b[not_S, pk] - b[not_S, pj]) +
          np.dot(a[not_S, k], b[not_S, pi] - b[not_S, pk]))
    
    # Term 4: f in S, g in S (3x3)
    old_33 = np.array([[b[pi, pi], b[pi, pj], b[pi, pk]],
                        [b[pj, pi], b[pj, pj], b[pj, pk]],
                        [b[pk, pi], b[pk, pj], b[pk, pk]]])
    new_33 = np.array([[b[pj, pj], b[pj, pk], b[pj, pi]],
                        [b[pk, pj], b[pk, pk], b[pk, pi]],
                        [b[pi, pj], b[pi, pk], b[pi, pi]]])
    a_33 = a[np.ix_([i, j, k], [i, j, k])]
    t4 = np.sum(a_33 * (new_33 - old_33))
    
    return int(t2 + t3 + t4)


def _local_search_2swap(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-swap local search."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return best_cost, best_p
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break
    return best_cost, best_p


def _local_search_3swap(a, b, p, deadline, best_cost, best_p, rng):
    """Randomized 3-swap local search: sample random triples, accept if improving."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    # Precompute for speed
    while time.time() < deadline:
        # Sample random triple
        idx = rng.integers(0, n, size=3)
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        if i == j or j == k or i == k:
            continue
        
        d = _cyclic3_delta(a, b, p, i, j, k)
        if d < 0:
            # Apply cycle: p[i]->p[j], p[j]->p[k], p[k]->p[i]
            pi = p[i]
            p[i] = p[j]
            p[j] = p[k]
            p[k] = pi
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
    return best_cost, best_p


def _anneal_with_3swap(a, b, p, deadline, best_cost, best_p, rng, t_start=3000.0, t_min=1.0, cooling=0.99998):
    """Simulated annealing using both 2-swap and 3-swap moves."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    T = t_start
    use_3swap = 0.3  # 30% of moves are 3-swaps
    
    while time.time() < deadline:
        if T < t_min:
            T = t_start
            p = best_p.copy()
            cur = best_cost
            continue
        
        if rng.random() < use_3swap:
            # 3-swap move
            idx = rng.integers(0, n, size=3)
            i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
            if i == j or j == k or i == k:
                continue
            d = _cyclic3_delta(a, b, p, i, j, k)
            if d < 0 or rng.random() < np.exp(-d / T):
                pi = p[i]
                p[i] = p[j]
                p[j] = p[k]
                p[k] = pi
                cur += d
                if cur < best_cost:
                    best_cost = cur
                    best_p = p.copy()
        else:
            # 2-swap move
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
            d = _swap_delta(a, b, p, i, j)
            if d < 0 or rng.random() < np.exp(-d / T):
                p[i], p[j] = p[j], p[i]
                cur += d
                if cur < best_cost:
                    best_cost = cur
                    best_p = p.copy()
        
        T *= cooling
    
    return best_cost, best_p


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # --- Constructive start: rank matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # --- Phase 1: 2-swap local search ---
    ls1_deadline = deadline - 80.0
    best_cost, best_p = _local_search_2swap(a, b, p, ls1_deadline, best_cost, best_p)
    
    # --- Phase 2: 3-swap local search to escape 2-swap optima ---
    ls2_deadline = deadline - 30.0
    best_cost, best_p = _local_search_3swap(a, b, best_p, ls2_deadline, best_cost, best_p, rng)
    
    # --- Phase 3: Another 2-swap pass after 3-swaps ---
    ls3_deadline = deadline - 20.0
    best_cost, best_p = _local_search_2swap(a, b, best_p, ls3_deadline, best_cost, best_p)
    
    # --- Phase 4: Simulated annealing with mixed 2/3-swap moves ---
    if time.time() < deadline - 10.0:
        anneal_deadline = deadline - 5.0
        best_cost, best_p = _anneal_with_3swap(a, b, best_p, anneal_deadline, best_cost, best_p, rng)
    
    # --- Phase 5: Final 2-swap + 3-swap polish ---
    if time.time() < deadline - 2.0:
        best_cost, best_p = _local_search_2swap(a, b, best_p, deadline - 1.0, best_cost, best_p)
        if time.time() < deadline - 1.0:
            best_cost, best_p = _local_search_3swap(a, b, best_p, deadline - 0.5, best_cost, best_p, rng)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

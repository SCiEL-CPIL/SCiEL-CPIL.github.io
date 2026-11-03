"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) STRUCTURAL CHANGE: Model as maximum independent set on support graph.
    #    Nodes: all C(11,4) = 330 four-element subsets of {0,...,10}.
    #    Edge between two supports if |intersection| >= 3 (i.e., they share 3+ coordinates).
    #    For weight-4 vectors with values in {+1,-1}, two vectors with supports A,B
    #    have inner product = sum over A∩B of (sign_a * sign_b).
    #    If |A∩B| <= 2, we can always choose signs so that inner product <= 2 = m//2.
    #    Actually for |A∩B| <= 2, max possible inner product is 2 (if both signs match on all shared).
    #    So condition: inner product <= 2, which is always satisfiable when |A∩B| <= 2.
    #    When |A∩B| >= 3, max inner product is 3 or 4, which may violate the constraint.
    #    
    #    Key insight: if we pick an independent set of supports (no two share >= 3 coords),
    #    then for EACH support we can include ALL 16 sign variants, because:
    #    - Within same support: two vectors differ in at least one sign, so inner product
    #      = 4 - 2*(number of sign flips). Min inner product within same support = -4.
    #      Max inner product within same support = 2 (differ in exactly one sign).
    #      Wait: if they differ in exactly 1 sign, inner product = 3*1 + 1*(-1) = 2. OK <= 2.
    #      If they differ in 0 signs, they're the same vector (excluded).
    #      So all 16 sign variants of the same support are mutually compatible!
    #    - Between different supports A, B with |A∩B| <= 2: for any sign choices,
    #      inner product = sum over A∩B of (s_a * s_b), which is at most |A∩B| <= 2. OK.
    #    
    #    Therefore: any independent set of supports of size S gives 16*S valid vectors!
    # ------------------------------------------------------------------

    # Generate all C(11,4) = 330 supports as bitmasks
    all_supports = list(itertools.combinations(range(D), 4))
    n_sup = len(all_supports)  # 330
    support_bits = np.zeros(n_sup, dtype=np.uint32)
    for idx, sup in enumerate(all_supports):
        mask = 0
        for c in sup:
            mask |= (1 << c)
        support_bits[idx] = mask

    # Build incompatibility matrix: edge if |intersection| >= 3
    # intersection_size(i,j) = popcount(support_bits[i] & support_bits[j])
    # For 4-element sets, intersection can be 0,1,2,3,4. Incompatible if >= 3.
    # Use vectorized approach with NumPy
    # Convert to uint32 and compute pairwise AND then popcount
    # For 330x330, we can do it in chunks
    incompatible = np.zeros((n_sup, n_sup), dtype=bool)
    
    # Compute intersection sizes using bit manipulation
    # For each pair, popcount of AND
    # Use a lookup table for popcount of 11-bit values
    popcount_11 = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_11[i] = bin(i).count('1')
    
    # Vectorized: for each i, compute AND with all j, then popcount
    bits_u32 = support_bits.astype(np.uint32)
    for i in range(n_sup):
        if is_deadline():
            break
        ands = bits_u32 & bits_u32[i]
        # Convert uint32 to int for indexing into popcount table (only lower 11 bits matter)
        idx_arr = ands.astype(np.int32) & 0x7FF  # mask to 11 bits
        inter_sizes = popcount_11[idx_arr]
        incompatible[i, :] = inter_sizes >= 3
    
    # Symmetrize (should already be symmetric)
    # incompatible = incompatible | incompatible.T
    
    # Diagonal is True (self-intersection = 4 >= 3), that's fine for independent set
    
    # ------------------------------------------------------------------
    # 3) Simulated Annealing for Maximum Independent Set on this graph
    #    State: binary vector x of length n_sup, x[i]=1 means support i is in the set.
    #    Objective: maximize sum(x) subject to no two selected nodes are adjacent.
    #    Penalty approach: energy = -sum(x) + penalty * (number of violated edges among selected)
    #    Or: use a "cost" = number of conflicts, minimize (conflicts - alpha * |S|)
    # ------------------------------------------------------------------
    
    def solve_mis_sa(n_nodes, adj_matrix, time_budget, seed=42):
        """Simulated annealing for max independent set.
        adj_matrix[i,j] = True if i and j are adjacent (incompatible).
        Returns the best independent set found as a boolean array."""
        rng = np.random.default_rng(seed)
        
        # Initial solution: greedy random
        x = np.zeros(n_nodes, dtype=bool)
        remaining = np.arange(n_nodes)
        np.random.seed(seed)
        np.random.shuffle(remaining)
        for i in remaining:
            if not np.any(adj_matrix[i, x]):
                x[i] = True
        
        best_x = x.copy()
        best_score = x.sum()
        
        # Energy function: -|S| + lambda * conflicts
        # conflicts = number of edges with both endpoints in S
        # For efficiency, track conflicts incrementally
        
        def count_conflicts(x_arr):
            """Count edges where both endpoints are selected."""
            if not np.any(x_arr):
                return 0
            sub = adj_matrix[np.ix_(x_arr, x_arr)]
            return int(np.sum(sub)) // 2  # each edge counted twice
        
        def energy(x_arr, lam):
            return -x_arr.sum() + lam * count_conflicts(x_arr)
        
        # SA parameters
        T_start = 5.0
        T_end = 0.01
        n_iter_budget = 200000  # we'll do as many as fit in time_budget
        
        t_start = time.perf_counter()
        lam = 1.0  # penalty weight for conflicts
        
        for it in range(n_iter_budget):
            if time.perf_counter() - t_start > time_budget:
                break
            if is_deadline():
                break
            
            # Annealing schedule
            frac = it / n_iter_budget
            T = T_start * (T_end / T_start) ** frac
            
            # Pick a random node to flip
            i = rng.integers(n_nodes)
            
            # Compute delta energy for flipping node i
            if x[i]:
                # Removing i: gain +1 in |S|, lose conflicts involving i
                neighbors_in_S = np.sum(adj_matrix[i, x])
                delta_E = -1 + lam * neighbors_in_S  # energy change (remove i)
                # Actually: new energy = old energy - (-1) - lam * (conflicts removed)
                # Let me think more carefully.
                # E = -|S| + lam * conflicts
                # Removing i: |S| decreases by 1, so -|S| increases by 1
                # conflicts: edges from i to other selected nodes are removed
                # So conflicts decrease by neighbors_in_S
                # delta_E = (+1) + lam * (-neighbors_in_S) = 1 - lam * neighbors_in_S
                delta_E = 1 - lam * neighbors_in_S
            else:
                # Adding i: |S| increases by 1, so -|S| decreases by 1
                # conflicts: edges from i to other selected nodes are added
                neighbors_in_S = np.sum(adj_matrix[i, x])
                delta_E = -1 + lam * neighbors_in_S
            
            # Accept if delta_E < 0 or with probability exp(-delta_E / T)
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                x[i] = not x[i]
                
                # Check if current solution is a valid independent set
                if not np.any(adj_matrix[np.ix_(x, x)]):
                    cur_size = x.sum()
                    if cur_size > best_score:
                        best_score = cur_size
                        best_x = x.copy()
        
        return best_x, best_score

    # Run SA with multiple seeds and time budgets
    sa_time_budget = min(90.0, DEADLINE_S - elapsed() - 5.0)
    if sa_time_budget > 5.0:
        # Split time among a few runs
        n_runs = 3
        per_run_time = sa_time_budget / n_runs
        
        for run_seed in [42, 123, 999]:
            if is_deadline() or elapsed() + 5.0 > DEADLINE_S:
                break
            try:
                sa_x, sa_k = solve_mis_sa(n_sup, incompatible, per_run_time, seed=run_seed)
                # Convert support indices to actual vectors
                if sa_k > 0:
                    selected_supports = all_supports[sa_x]
                    # For each support, generate all 16 sign variants
                    vecs = []
                    for sup in selected_supports:
                        for signs in itertools.product([1, -1], repeat=4):
                            v = np.zeros(D, dtype=np.int64)
                            for c, s in zip(sup, signs):
                                v[c] = s
                            vecs.append(v)
                    if vecs:
                        vec_arr = np.array(vecs, dtype=np.int64)
                        update_best(vec_arr)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 4) Also try a quick greedy on the support graph for comparison
    # ------------------------------------------------------------------
    if not is_deadline() and elapsed() < DEADLINE_S - 10.0:
        # Greedy: repeatedly pick the node with fewest neighbors among remaining
        x_greedy = np.zeros(n_sup, dtype=bool)
        in_set = np.zeros(n_sup, dtype=bool)
        available = np.ones(n_sup, dtype=bool)
        
        while np.any(available):
            if is_deadline():
                break
            # For each available node, count neighbors in current set
            # Pick the one with min conflicts (prefer 0)
            for i in np.where(available)[0]:
                if not np.any(incompatible[i, x_greedy]):
                    x_greedy[i] = True
                    in_set[i] = True
                    available[i] = False
                    # Remove neighbors from available
                    available[incompatible[i, :]] = False
                    break
            else:
                # No node with 0 conflicts; pick one with fewest
                min_conf = np.inf
                best_i = -1
                for i in np.where(available)[0]:
                    c = np.sum(incompatible[i, x_greedy])
                    if c < min_conf:
                        min_conf = c
                        best_i = i
                if best_i >= 0 and min_conf == 0:
                    x_greedy[best_i] = True
                    in_set[best_i] = True
                    available[best_i] = False
                    available[incompatible[best_i, :]] = False
                else:
                    break
        
        if x_greedy.sum() > 0:
            selected_supports = all_supports[x_greedy]
            vecs = []
            for sup in selected_supports:
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for c, s in zip(sup, signs):
                        v[c] = s
                    vecs.append(v)
            if vecs:
                vec_arr = np.array(vecs, dtype=np.int64)
                update_best(vec_arr)

    # ------------------------------------------------------------------
    # 5) Safety: return the best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best

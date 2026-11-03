"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Strategy: Simulated Annealing on the support-selection problem.
    Instead of greedy MIS on the conflict graph, we use SA to find a large independent set
    of supports (where no two selected supports share 3 indices). This allows escaping
    local optima that greedy methods get stuck in.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Generate all 4-element supports as bitmasks
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    
    n_sup = len(supports)  # C(11,4) = 330
    
    # 2. Precompute conflict matrix: conflict[i,j] = True if supports[i] & supports[j] has bit_count == 3
    # Two 4-supports conflict if they share exactly 3 indices (intersection size 3)
    # For an independent set, we want NO two selected supports to conflict.
    # So we build a conflict graph and find a large independent set.
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports[i] & supports[j]
            if inter.bit_count() == 3:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # 3. Simulated Annealing to find a large independent set
    # State: binary vector of length n_sup indicating which supports are selected
    # Objective: maximize number of selected supports subject to no conflicts
    # Penalty: number of conflicting pairs in the selected set
    
    def _evaluate_selection(sel):
        """Return (count, penalty) for a selection mask."""
        count = sel.sum()
        # Penalty = number of conflicting pairs both selected
        # Use matrix multiplication trick: for selected vector s, penalty = 0.5 * s^T C s
        # where C is the conflict matrix
        penalty = int(np.dot(sel, conflict @ sel) // 2)
        return count, penalty
    
    def _score(sel):
        """Score = count - large_penalty * penalty"""
        count, penalty = _evaluate_selection(sel)
        return count - 1000 * penalty
    
    # Initialize with a greedy solution
    sel = np.zeros(n_sup, dtype=int)
    alive = set(range(n_sup))
    while alive:
        # Pick node with min degree in alive
        v = min(alive, key=lambda x: np.sum(conflict[x, list(alive)]))
        sel[v] = 1
        # Remove v and its neighbors from alive
        neighbors = np.where(conflict[v])[0]
        alive.difference_update(neighbors)
        alive.discard(v)
    
    best_sel = sel.copy()
    best_score = _score(best_sel)
    current_score = best_score
    
    # SA parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    max_iter = 500000
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        if T < T_min:
            break
        
        # Random move: flip one bit
        v = rng.integers(0, n_sup)
        sel[v] = 1 - sel[v]
        
        new_score = _score(sel)
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_sel = sel.copy()
        
        T *= alpha
    
    # 4. Post-process: remove any remaining conflicts from best_sel
    sel = best_sel.copy()
    # Iteratively remove nodes that have conflicts
    changed = True
    while changed:
        changed = False
        for i in range(n_sup):
            if sel[i] == 1:
                # Check if i has any conflict with another selected node
                neighbors = np.where(conflict[i] & sel)[0]
                neighbors = neighbors[neighbors != i]
                if len(neighbors) > 0:
                    # Remove i or one of its neighbors
                    # Remove i to keep it simple
                    sel[i] = 0
                    changed = True
    
    # 5. Generate vectors for selected supports
    vectors = []
    for idx in range(n_sup):
        if sel[idx] == 1:
            mask = supports[idx]
            inds = [i for i in range(D) if (mask >> i) & 1]
            # For each support, add all 16 sign combinations? No, that would create conflicts.
            # Actually, for a fixed support, different sign assignments give vectors with
            # dot product in {-4, -2, 0, 2, 4} with each other (same support).
            # If we pick two vectors with the same support and dot product 4, they're identical.
            # If dot product 2, distance^2 = 8 - 4 = 4 >= 4. OK.
            # If dot product -2, distance^2 = 8 + 4 = 12. OK.
            # If dot product 0, distance^2 = 8. OK.
            # So we CAN include multiple sign variants of the same support!
            # But wait, the support is the same, so they share all 4 indices.
            # The conflict was defined for DIFFERENT supports sharing 3 indices.
            # Same support sharing 4 indices: max dot product is 4 (identical).
            # We should avoid identical vectors.
            # For same support, different sign patterns:
            # Two sign patterns differ in k positions. Dot product = 4 - 2k.
            # k=0: dot=4 (identical) - avoid
            # k=1: dot=2 - OK
            # k=2: dot=0 - OK
            # k=3: dot=-2 - OK
            # k=4: dot=-4 - OK
            # So we can include all 16 sign patterns except we need to avoid duplicates.
            # Actually, each sign pattern is a distinct vector. We can include all 16.
            # But we need to check cross-support conflicts too.
            
            # For now, let's just add one vector per support to keep it simple.
            # We can add more later.
            v = np.zeros(D, dtype=np.int64)
            for idx_val in inds:
                v[idx_val] = 1
            vectors.append(v)
    
    # Actually, let's be smarter. For each selected support, we can add multiple sign variants.
    # But we need to ensure all pairwise distances are >= max norm.
    # Let's first collect all candidate vectors (all sign variants of selected supports),
    # then greedily select a subset that satisfies the distance constraint.
    
    all_candidates = []
    for idx in range(n_sup):
        if sel[idx] == 1:
            mask = supports[idx]
            inds = [i for i in range(D) if (mask >> i) & 1]
            for sign_mask in range(16):
                v = np.zeros(D, dtype=np.int64)
                for j, idx_val in enumerate(inds):
                    if sign_mask & (1 << j):
                        v[idx_val] = 1
                    else:
                        v[idx_val] = -1
                all_candidates.append(v)
    
    # Also add shell-2 vectors (norm^2 = 2)
    for i in range(D):
        for j in range(i + 1, D):
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            all_candidates.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            all_candidates.append(v2)
    
    if len(all_candidates) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    candidates = np.array(all_candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    max_norm_sq = norms_sq.max()
    
    # Greedily select vectors satisfying: for all selected u, v: ||u-v||^2 >= max_norm_sq
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v >= max_norm_sq
    # So u.v <= (||u||^2 + ||v||^2 - max_norm_sq) / 2
    
    selected = []
    alive_cand = list(range(n_cand))
    
    # Sort by norm descending to prioritize larger vectors first
    order = np.argsort(-norms_sq)
    alive_cand = [order[i] for i in range(n_cand)]
    
    while alive_cand:
        if time.time() > deadline - 2:
            break
        v_idx = alive_cand[0]
        selected.append(v_idx)
        # Remove candidates that conflict with v_idx
        u = candidates[v_idx]
        u_norm_sq = norms_sq[v_idx]
        # For each remaining candidate w, check if ||u-w||^2 >= max_norm_sq
        # ||u-w||^2 = u_norm_sq + w_norm_sq - 2 u.w >= max_norm_sq
        # u.w <= (u_norm_sq + w_norm_sq - max_norm_sq) / 2
        remaining = alive_cand[1:]
        if len(remaining) > 0:
            dots = candidates[remaining] @ u
            w_norms_sq = norms_sq[remaining]
            threshold = (u_norm_sq + w_norms_sq - max_norm_sq) / 2.0
            valid = dots <= threshold
            alive_cand = [remaining[i] for i in range(len(remaining)) if valid[i]]
        else:
            alive_cand = []
    
    result = candidates[selected]
    return result

# EVOLVE-BLOCK-END

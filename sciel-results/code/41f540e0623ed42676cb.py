"""Rewrite: Branch-and-bound MIS with greedy coloring bounds on 330-node support graph."""
import time
import numpy as np
from itertools import combinations

D = 11

def _greedy_color_bound(cand_set, color_of):
    """Greedy sequential coloring on cand_set. Returns (color_count, color_of)."""
    color_of.clear()
    c = 0
    for v in cand_set:
        # smallest color not used by any colored neighbor
        used = set()
        for nb in color_of:
            if cand_set[nb] and nb != v:
                used.add(color_of[nb])
        while c in used:
            c += 1
        color_of[v] = c
        c += 1
    return c

def _branch_and_bound(adj, n, time_limit):
    """
    Exact MIS via branch-and-bound with greedy coloring upper bound.
    adj: list of sets (neighbors)
    n: number of vertices
    Returns: best independent set as set of vertex indices.
    """
    best_set = set()
    best_size = 0
    color_of = {}
    
    # Order vertices by degree ascending (heuristic: low degree first for better bounds)
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: deg[i])
    pos = {v: i for i, v in enumerate(order)}
    # Re-index: new vertex i = order[i]
    remap = {order[i]: i for i in range(n)}
    adj_new = [set() for _ in range(n)]
    for i in range(n):
        for nb in adj[i]:
            adj_new[remap[i]].add(remap[nb])
    
    cand = [True] * n  # candidate pool
    cand_set = set(range(n))
    
    def bb(cand_arr, cand_set_local, current_set, current_size):
        nonlocal best_set, best_size
        if time.time() > time_limit:
            return
        # Upper bound: greedy coloring on cand_set_local
        if not cand_set_local:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        # Bound
        bound = _greedy_color_bound(cand_set_local, color_of)
        if current_size + bound <= best_size:
            return
        if current_size + len(cand_set_local) <= best_size:
            return
        # Choose pivot vertex: the one with highest color in greedy coloring
        # Actually pick the vertex with max color
        pivot = max(cand_set_local, key=lambda v: color_of[v])
        
        # Branch 1: include pivot
        new_cand_set = cand_set_local - {pivot} - adj_new[pivot]
        new_current_set = current_set | {pivot}
        bb([cand_arr[v] for v in range(n)] , new_cand_set, new_current_set, current_size + 1)
        
        if time.time() > time_limit:
            return
        
        # Branch 2: exclude pivot
        new_cand_set = cand_set_local - {pivot}
        bb(None, new_cand_set, current_set, current_size)
    
    bb(cand, set(range(n)), set(), 0)
    
    # Map back to original indices
    inv_remap = {v: u for u, v in remap.items()}
    return {inv_remap[v] for v in best_set}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s
    
    # Generate all 4-element supports as bitmasks
    supports = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    n_sup = len(supports)  # C(11,4) = 330
    
    # Build conflict graph: two supports conflict if they share exactly 3 indices
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports[i] & supports[j]
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Solve MIS via branch-and-bound
    mis_set = _branch_and_bound(adj, n_sup, deadline - 3)
    
    # Generate all candidate vectors: for each MIS support, all 16 sign patterns
    all_candidates = []
    for idx in sorted(mis_set):
        mask = supports[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                v[idx_val] = 1 if (sign_mask >> j) & 1 else -1
            all_candidates.append(v)
    
    # Add shell-2 vectors (weight-2) for extra density
    for i in range(D):
        for j in range(i + 1, D):
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            all_candidates.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            all_candidates.append(v2)
    
    if not all_candidates:
        return np.zeros((0, D), dtype=np.int64)
    
    candidates = np.array(all_candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Deduplicate
    candidates = np.unique(candidates, axis=0)
    n_cand = len(candidates)
    
    norms_sq = np.sum(candidates ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())
    
    # Greedy selection: sort by norm desc, add if compatible with all selected
    order = np.argsort(-norms_sq)
    selected = []
    # Precompute compatibility: for each pair (a,b), is ||a-b||^2 >= max_norm_sq?
    # That is: norms_sq[a] + norms_sq[b] - 2 * candidates[a]·candidates[b] >= max_norm_sq
    # i.e., candidates[a]·candidates[b] <= (norms_sq[a] + norms_sq[b] - max_norm_sq) / 2
    
    # For speed, build conflict matrix among candidates
    # n_cand up to 330*16 + 110 = 5390 -- too big for O(n^2) dot products in Python
    # Use vectorized numpy: compute Gram matrix in chunks
    
    # Actually, just do greedy with vectorized dot product against selected set
    # But selected set grows, so we need to track which candidates are still valid.
    
    alive = list(order)
    selected_idx = []
    
    # Precompute dot products of all candidates with each other in chunks for the greedy step
    # Better: for each candidate we pick, compute its dot product with all alive candidates at once.
    
    while alive:
        if time.time() > deadline - 2:
            break
        v_idx = alive[0]
        u = candidates[v_idx]
        u_norm_sq = norms_sq[v_idx]
        selected_idx.append(v_idx)
        
        if len(alive) > 1:
            remaining = alive[1:]
            # Dot products: candidates[remaining] @ u
            dots = candidates[remaining] @ u
            w_norms_sq = norms_sq[remaining]
            threshold = (u_norm_sq + w_norms_sq - max_norm_sq) / 2.0
            valid = dots <= threshold
            alive = [remaining[i] for i in range(len(remaining)) if valid[i]]
        else:
            alive = []
    
    result = candidates[selected_idx]
    return result

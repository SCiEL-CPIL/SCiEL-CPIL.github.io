"""Rewritten program for kissing number in dimension 11.

Strategy:
  - Build a larger, structured candidate pool: all vectors with squared norm 2, 3, 4, 5
    (i.e. weights 2,3,4,5 with coordinates in {-1,0,1}), plus a few weight-6 vectors
    with a single 2 to explore the "max norm = 4" boundary more tightly.
  - The constraint is: for every pair i != j,  ||v_i - v_j||^2 >= max_j ||v_j||^2.
    Since we include weight-5 vectors (norm_sq=5), the max norm_sq in the pool is 5.
    But to keep the problem tractable and the kissing configuration valid, we
    normalize the *maximum norm* to be the largest norm present in the final set.
    So the constraint is: dist_sq[i,j] >= max_norm_sq for all i!=j.
    We will compute max_norm_sq dynamically from the selected set.

    However, a simpler and standard approach: fix the target minimum distance
    to be sqrt(4) = 2, i.e. require dist_sq >= 4. This means we only need
    max_norm_sq <= 4 in the selected set. So we should NOT include weight-5
    vectors in the final answer if we want dist_sq >= 4 to be the binding
    constraint. Instead, let's include weight-2,3,4 vectors only, and require
    dist_sq >= 4.

    Actually, re-reading the task: "minimum pairwise distance is at least the
    maximum vector norm". If we include a vector with norm_sq = 5, then max_norm
    = sqrt(5), so we need all pairwise distances >= sqrt(5), i.e. dist_sq >= 5.
    This is a stricter constraint. So including weight-5 vectors makes it harder.

    Best approach: use only weight-2, 3, 4 vectors (max norm_sq = 4), and require
    dist_sq >= 4. This is the standard kissing number setup with equal radius.

  - Use a bipartite matching / maximum flow approach if the conflict graph is
    bipartite. For weight-4 vectors with dot product > 2, the conflict graph
    may not be bipartite in general. So we'll use a greedy + local search
    hybrid, but with a much better candidate generation and a more efficient
    conflict check.

  - Key improvement: use a smarter candidate ordering and a better local search
    with 2-for-1 and 1-for-2 swaps, plus a final polish pass.

Let me focus on:
  1. Efficient candidate generation (vectorized where possible)
  2. Fast conflict matrix computation
  3. Multi-start greedy with different randomizations
  4. Aggressive local search with multiple swap types
"""

import numpy as np
import itertools
import time
import random

D = 11
MAX_NORM_SQ = 4  # We require dist_sq >= 4, so max norm_sq in set must be <= 4

def _generate_candidates():
    """Generate all candidate vectors with squared norm 2, 3, or 4."""
    candidates = []
    norms_sq = []
    
    # Weight 2: choose 2 positions, assign ±1
    for pos in itertools.combinations(range(D), 2):
        for s in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[pos[0]] = s[0]
            v[pos[1]] = s[1]
            candidates.append(v)
            norms_sq.append(2)
    
    # Weight 3: choose 3 positions, assign ±1
    for pos in itertools.combinations(range(D), 3):
        for s in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[pos[0]] = s[0]
            v[pos[1]] = s[1]
            v[pos[2]] = s[2]
            candidates.append(v)
            norms_sq.append(3)
    
    # Weight 4: choose 4 positions, assign ±1
    for pos in itertools.combinations(range(D), 4):
        for s in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            v[pos[0]] = s[0]
            v[pos[1]] = s[1]
            v[pos[2]] = s[2]
            v[pos[3]] = s[3]
            candidates.append(v)
            norms_sq.append(4)
    
    return np.array(candidates, dtype=np.int64), np.array(norms_sq, dtype=np.int64)

def _compute_conflicts(candidates, norms_sq):
    """
    Compute the conflict matrix: conflicts[i,j] = True if dist_sq(i,j) < MAX_NORM_SQ.
    dist_sq(i,j) = norms_sq[i] + norms_sq[j] - 2 * dot(v_i, v_j)
    """
    n = len(candidates)
    dots = candidates @ candidates.T  # (n, n)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < MAX_NORM_SQ
    np.fill_diagonal(conflicts, False)
    return conflicts

def _greedy_select(candidates, norms_sq, conflicts, seed, deadline):
    """
    Greedy independent set selection with randomized ordering.
    Returns list of selected indices.
    """
    random.seed(seed)
    n = len(candidates)
    order = list(range(n))
    random.shuffle(order)
    
    selected = []
    selected_set = set()
    available = set(range(n))
    
    for v in order:
        if v not in available:
            continue
        # Check if v conflicts with any already selected
        conflict = False
        for s in selected:
            if conflicts[v, s]:
                conflict = True
                break
        if not conflict:
            selected.append(v)
            selected_set.add(v)
            # Remove all neighbors of v from available
            for u in range(n):
                if u != v and conflicts[v, u] and u in available:
                    available.discard(u)
            available.discard(v)
        
        if time.time() > deadline - 1.0:
            break
    
    return selected

def _greedy_min_degree(candidates, norms_sq, conflicts, seed, deadline):
    """
    Greedy selection: repeatedly pick the vertex with minimum degree in the
    remaining available subgraph.
    """
    random.seed(seed)
    n = len(candidates)
    available = set(range(n))
    selected = []
    
    # Precompute neighbor sets
    neighbors = [set(np.where(conflicts[i])[0].tolist()) for i in range(n)]
    
    while available and time.time() < deadline - 1.0:
        # Find vertex with min degree among available
        best_v = None
        min_deg = float('inf')
        
        # If too many available, sample
        if len(available) > 200:
            sample = random.sample(list(available), 200)
        else:
            sample = list(available)
        
        for v in sample:
            deg = sum(1 for u in available if conflicts[v, u])
            if deg < min_deg:
                min_deg = deg
                best_v = v
        
        if best_v is None:
            break
        
        selected.append(best_v)
        available.discard(best_v)
        for u in list(available):
            if conflicts[best_v, u]:
                available.discard(u)
    
    return selected

def _local_search(selected, candidates, norms_sq, conflicts, deadline):
    """
    Local search: try to add vertices, then do 1-for-k swaps.
    """
    S = set(selected)
    n = len(candidates)
    improved = True
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # 1. Try adding any vertex that doesn't conflict
        for v in range(n):
            if v in S:
                continue
            ok = True
            for s in S:
                if conflicts[v, s]:
                    ok = False
                    break
            if ok:
                S.add(v)
                improved = True
        
        # 2. Try 1-for-k swaps: remove one, add two or more
        if not improved and time.time() < deadline - 2.0:
            for v in list(S):
                test_S = S - {v}
                
                # Find all vertices compatible with test_S
                candidates_for_add = []
                for u in range(n):
                    if u in test_S or u == v:
                        continue
                    ok = True
                    for s in test_S:
                        if conflicts[u, s]:
                            ok = False
                            break
                    if ok:
                        candidates_for_add.append(u)
                
                if len(candidates_for_add) < 2:
                    continue
                
                # Greedily pick an independent subset from candidates_for_add
                avail = set(candidates_for_add)
                added = []
                while avail:
                    best_u = None
                    min_deg = float('inf')
                    for u in avail:
                        deg = sum(1 for w in avail if conflicts[u, w])
                        if deg < min_deg:
                            min_deg = deg
                            best_u = u
                    if best_u is None:
                        break
                    added.append(best_u)
                    avail.discard(best_u)
                    for w in list(avail):
                        if conflicts[best_u, w]:
                            avail.discard(w)
                    if len(added) >= 3:
                        break
                
                if len(added) >= 2:
                    S = test_S | set(added)
                    improved = True
    
    return list(S)

def _local_search_2for1(selected, candidates, norms_sq, conflicts, deadline):
    """
    Additional local search: 2-for-1 swaps (remove two, add one).
    This can sometimes help escape local optima.
    """
    S = set(selected)
    n = len(candidates)
    improved = True
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        for v1 in list(S):
            for v2 in list(S):
                if v1 == v2:
                    continue
                test_S = S - {v1, v2}
                
                # Find vertices compatible with test_S but not in test_S
                for u in range(n):
                    if u in test_S:
                        continue
                    ok = True
                    for s in test_S:
                        if conflicts[u, s]:
                            ok = False
                            break
                    if ok:
                        # Check that u doesn't conflict with v1 or v2 (it's being added)
                        # Actually u is new, so just check with test_S
                        S_new = test_S | {u}
                        if len(S_new) > len(S):
                            S = S_new
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate candidates
    candidates, norms_sq = _generate_candidates()
    n_candidates = len(candidates)
    
    # Compute conflicts
    conflicts = _compute_conflicts(candidates, norms_sq)
    
    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.5
    trial = 0
    
    while time.time() < greedy_deadline:
        trial += 1
        # Alternate between different greedy strategies
        if trial % 2 == 0:
            sel = _greedy_select(candidates, norms_sq, conflicts, 42 + trial, deadline)
        else:
            sel = _greedy_min_degree(candidates, norms_sq, conflicts, 42 + trial, deadline)
        
        if len(sel) > len(best_selected):
            best_selected = sel
        
        # Quick local search on this result
        if time.time() < deadline - 10.0:
            refined = _local_search(sel, candidates, norms_sq, conflicts, deadline)
            if len(refined) > len(best_selected):
                best_selected = refined
    
    # Phase 2: Thorough local search on the best result
    if best_selected and time.time() < deadline - 5.0:
        refined = _local_search(best_selected, candidates, norms_sq, conflicts, deadline)
        if len(refined) > len(best_selected):
            best_selected = refined
    
    # Phase 3: Try 2-for-1 swaps
    if best_selected and time.time() < deadline - 2.0:
        refined = _local_search_2for1(best_selected, candidates, norms_sq, conflicts, deadline)
        if len(refined) > len(best_selected):
            best_selected = refined
    
    # Phase 4: Final polish - one more full local search
    if best_selected and time.time() < deadline - 1.0:
        refined = _local_search(best_selected, candidates, norms_sq, conflicts, deadline)
        if len(refined) > len(best_selected):
            best_selected = refined
    
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]

# EVOLVE-BLOCK-END

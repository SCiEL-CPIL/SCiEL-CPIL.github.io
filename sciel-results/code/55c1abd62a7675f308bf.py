import numpy as np
import time
import itertools
from collections import defaultdict

def _build_conflict_graph(candidates, d):
    """
    Build adjacency list for conflict graph.
    Two vectors conflict if their squared distance < max_norm_sq (=4 for norm-4 vectors).
    For norm-4 vectors: ||u-v||² = 8 - 2*u.v, so conflict iff u.v > 2.
    """
    n = len(candidates)
    # Compute all pairwise dot products
    # candidates is (n, d) array
    # dot[i,j] = candidates[i] . candidates[j]
    dots = candidates @ candidates.T  # (n, n)
    
    # Conflict if dot > 2
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dots[i, j] > 2:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj

def _max_independent_set_bnb(adj, n, time_limit, start_time):
    """
    Branch and bound for maximum independent set.
    Uses coloring upper bound on the remaining candidates.
    """
    best = set()
    best_size = 0
    
    def color_bound(candidates_set):
        """
        Greedy coloring to get an upper bound on the size of MIS in the subgraph.
        Returns the number of colors used, which is an upper bound on the independent set size.
        """
        if not candidates_set:
            return 0
        remaining = list(candidates_set)
        colors = {}
        color_count = 0
        while remaining:
            color_count += 1
            # Greedy: pick vertices, assign current color if no conflict with same-color vertices
            color_class = []
            new_remaining = []
            for v in remaining:
                can_color = True
                for u in color_class:
                    if u in adj[v]:
                        can_color = False
                        break
                if can_color:
                    color_class.append(v)
                else:
                    new_remaining.append(v)
            remaining = new_remaining
        return color_count
    
    def bnb(available, current_set):
        nonlocal best, best_size
        if time.time() - start_time > time_limit:
            return
        
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best = set(current_set)
            return
        
        # Upper bound: current_size + color_bound(available)
        ub = len(current_set) + color_bound(available)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph
        v = max(available, key=lambda x: len(adj[x] & available))
        
        # Branch 1: include v
        new_available = available - {v} - adj[v]
        bnb(new_available, current_set | {v})
        
        # Branch 2: exclude v
        bnb(available - {v}, current_set)
    
    bnb(set(range(n)), set())
    return best

def _greedy_independent_set(adj, n, order):
    """
    Greedy: process vertices in given order, add if no conflict with already added.
    """
    result = []
    added = set()
    for v in order:
        if v in added:
            continue
        # Check if v conflicts with any in result
        conflict = False
        for u in result:
            if u in adj[v]:
                conflict = True
                break
        if not conflict:
            result.append(v)
            added.add(v)
    return result

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # Safety margin
    
    # Strategy 1: Norm-squared-2 vectors (guaranteed valid, k=220)
    # All vectors with exactly two ±1 entries
    cands_r2 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    cands_r2.append(v)
    cands_r2 = np.array(cands_r2, dtype=np.int64)
    best_k = len(cands_r2)
    best_arr = cands_r2.copy()
    
    # Strategy 2: Norm-squared-4 vectors
    # Type A: ±2 on one coordinate (22 vectors)
    # Type B: ±1 on four coordinates (5280 vectors)
    cands_r4 = []
    # Type A
    for i in range(d):
        for s in [-2, 2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            cands_r4.append(v)
    # Type B
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            cands_r4.append(v)
    cands_r4 = np.array(cands_r4, dtype=np.int64)
    n4 = len(cands_r4)
    
    # Build conflict graph for norm-4 vectors
    # Conflict if dot product > 2
    t0 = time.time()
    if time.time() < deadline - 100:
        dots = cands_r4 @ cands_r4.T
        adj = [set() for _ in range(n4)]
        for i in range(n4):
            for j in range(i + 1, n4):
                if dots[i, j] > 2:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Try greedy with multiple orderings first (fast)
        best_greedy = []
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 100:
                break
            order = list(range(n4))
            np.random.shuffle(order)
            result = _greedy_independent_set(adj, n4, order)
            if len(result) > len(best_greedy):
                best_greedy = result
        
        if len(best_greedy) > best_k:
            best_k = len(best_greedy)
            best_arr = cands_r4[best_greedy]
        
        # Try branch-and-bound on a subset if time permits
        # B&B on full graph is too slow; try on a smaller induced subgraph
        # Or try B&B with aggressive pruning
        if time.time() < deadline - 50 and len(best_greedy) < best_k + 100:
            # Try B&B with time limit
            remaining_time = min(200, deadline - time.time() - 30)
            if remaining_time > 10:
                bnb_result = _max_independent_set_bnb(adj, n4, remaining_time, time.time())
                if len(bnb_result) > best_k:
                    best_k = len(bnb_result)
                    best_arr = cands_r4[list(bnb_result)]
    
    # Final verification
    if len(best_arr) == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    k = best_arr.shape[0]
    norms = np.sum(best_arr**2, axis=1)
    max_norm = int(np.max(norms))
    
    valid = True
    for i in range(k):
        if k < 2:
            break
        for j in range(i + 1, k):
            diff = best_arr[i] - best_arr[j]
            dist_sq = int(np.sum(diff * diff))
            if dist_sq < max_norm:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return cands_r2
    
    return np.around(best_arr).astype(np.int64)

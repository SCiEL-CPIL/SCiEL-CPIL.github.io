"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _build_conflict_graph(N=128):
    """
    Build a conflict graph where edge (u,v) exists iff there exists w != u,v
    such that gcd(u,v) == gcd(u,w) == gcd(v,w).
    Return adjacency as list of sets.
    """
    adj = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            guv = gcd(u, v)
            # Check if there exists w such that gcd(u,w)==guv and gcd(v,w)==guv
            # This means w is a multiple of guv, and gcd(u/guv, w/guv) == 1, gcd(v/guv, w/guv) == 1
            ug = u // guv
            vg = v // guv
            # w must be a multiple of guv, say w = guv * k
            # Need gcd(ug, k) == 1 and gcd(vg, k) == 1
            # And w in [1,N], w != u, w != v
            found = False
            for k in range(1, N // guv + 1):
                w = guv * k
                if w == u or w == v:
                    continue
                if gcd(ug, k) == 1 and gcd(vg, k) == 1:
                    found = True
                    break
            if found:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _branch_and_bound_mis(adj, N, deadline):
    """
    Maximum Independent Set via branch and bound.
    Uses Tomita's algorithm with color-based upper bound.
    """
    best_size = 0
    best_set = []
    
    # Precompute degree ordering for better branching
    # We'll use a simple approach: process vertices in order of decreasing degree
    
    def dfs(candidates, current_set, depth):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: if len(current_set) + len(candidates) <= best_size, prune
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Color bound: greedy coloring of candidate subgraph for upper bound
        # Simple upper bound: just use len(candidates)
        # Better: use a greedy coloring
        cand_list = list(candidates)
        # Order by degree within candidates
        cand_list.sort(key=lambda x: len(adj[x] & candidates), reverse=True)
        
        # Greedy coloring
        color = {}
        colors_used = 0
        for v in cand_list:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for nb in adj[v]:
                if nb in color:
                    neighbor_colors.add(color[nb])
            c = 1
            while c in neighbor_colors:
                c += 1
            color[v] = c
            colors_used = max(colors_used, c)
        
        # Upper bound from coloring
        if len(current_set) + colors_used <= best_size:
            return
        
        # Branch on a vertex
        # Choose vertex with max degree in candidates
        v = cand_list[0]
        
        # Branch 1: include v
        current_set.append(v)
        new_candidates = candidates - {v} - adj[v]
        dfs(new_candidates, current_set, depth + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        new_candidates = candidates - {v}
        dfs(new_candidates, current_set, depth + 1)
    
    all_vertices = set(range(1, N + 1))
    dfs(all_vertices, [], 0)
    
    return best_set, best_size

def _greedy_mis(adj, N, deadline):
    """
    Greedy Maximum Independent Set: repeatedly pick vertex with min degree,
    remove it and its neighbors.
    """
    candidates = set(range(1, N + 1))
    selected = []
    
    while candidates and time.time() < deadline:
        # Pick vertex with minimum degree in candidate subgraph
        min_deg = float('inf')
        v = None
        for u in candidates:
            deg = len(adj[u] & candidates)
            if deg < min_deg:
                min_deg = deg
                v = u
        if v is None:
            break
        selected.append(v)
        candidates -= adj[v]
        candidates.discard(v)
    
    return selected

def _local_search_mis(adj, N, current_set, deadline):
    """
    Local search: try to add vertices, or swap to improve.
    """
    current = set(current_set)
    best = list(current)
    best_size = len(best)
    all_numbers = set(range(1, N + 1))
    
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a vertex not in current set
        for v in all_numbers - current:
            # Check if v is independent from all in current
            if not (adj[v] & current):
                current.add(v)
                best = list(current)
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add two
        for v in list(current):
            if time.time() > deadline:
                break
            trial = current - {v}
            added = 0
            for u in all_numbers - trial:
                if u == v:
                    continue
                if not (adj[u] & trial):
                    trial.add(u)
                    added += 1
                    if added == 2:
                        break
            if added >= 2 and len(trial) > best_size:
                current = trial
                best = list(current)
                best_size = len(best)
                improved = True
                break
            elif added == 1 and len(trial) > best_size:
                current = trial
                best = list(current)
                best_size = len(best)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    N = 128
    
    # Build conflict graph
    adj = _build_conflict_graph(N)
    
    # Get initial solution via greedy MIS
    best_set = _greedy_mis(adj, N, deadline)
    
    # Try multiple greedy orderings for MIS
    if time.time() < deadline:
        # Greedy with different vertex ordering
        for order_func in [
            lambda x: x,
            lambda x: -x,
            lambda x: (-len(adj[x]), x),
            lambda x: (len(adj[x]), x),
        ]:
            if time.time() > deadline:
                break
            candidates = set(range(1, N + 1))
            selected = []
            sorted_cands = sorted(candidates, key=order_func)
            for v in sorted_cands:
                if v not in candidates:
                    continue
                # Check independence
                if not (adj[v] & set(selected)):
                    selected.append(v)
                    candidates.discard(v)
            if len(selected) > len(best_set):
                best_set = selected
    
    # Local search to improve
    if time.time() < deadline:
        best_set = _local_search_mis(adj, N, best_set, deadline)
    
    # Try branch and bound if time permits
    if time.time() < deadline - 5:
        bb_set, bb_size = _branch_and_bound_mis(adj, N, deadline)
        if bb_size > len(best_set):
            best_set = bb_set
    
    # Verify validity
    def is_valid(s):
        s = set(s)
        for a, b, c in combinations(sorted(s), 3):
            if gcd(a, b) == gcd(b, c) == gcd(c, a):
                return False
        return True
    
    if not is_valid(best_set):
        # Fallback: simple greedy
        selected = []
        for x in range(1, N + 1):
            if all(len({gcd(a, b), gcd(a, x), gcd(b, x)}) > 1 for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected
    
    return best_set
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _biclique_edges(left, right):
    """Return the set of edge keys covered by a biclique."""
    s = set()
    for u in left:
        for v in right:
            s.add(_edge_key(u, v))
    return s

def _is_valid_biclique(left, right, adj, remaining=None):
    """Check if (left, right) is a valid biclique: all cross pairs are edges."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _find_max_biclique(adj, remaining, deadline):
    """
    Find the largest biclique in the remaining edge set using a 
    multi-seed approach with iterative expansion.
    """
    if not remaining:
        return None
    
    n = P['n']
    
    # Compute remaining degrees
    deg = {}
    for (u, v) in remaining:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    # Sort vertices by remaining degree (descending)
    sorted_v = sorted(deg.keys(), key=lambda x: -deg[x])
    
    best = None
    best_size = 0
    
    # Strategy 1: Try all pairs of high-degree vertices as seeds
    top_k = min(20, len(sorted_v))
    for i in range(top_k):
        if time.time() > deadline:
            break
        a = sorted_v[i]
        for j in range(i + 1, top_k):
            if time.time() > deadline:
                break
            b = sorted_v[j]
            key = _edge_key(a, b)
            if key not in remaining:
                continue
            
            # Seed biclique: left={a}, right={b}
            left = {a}
            right = {b}
            
            # Iterative expansion
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                # Try to add vertices to left
                for v in range(n):
                    if v in left or v in right:
                        continue
                    # v can join left if it connects to all of right via remaining edges
                    can_add = all(_edge_key(v, r) in remaining for r in right)
                    if can_add:
                        left.add(v)
                        changed = True
                # Try to add vertices to right
                for v in range(n):
                    if v in left or v in right:
                        continue
                    can_add = all(_edge_key(v, l) in remaining for l in left)
                    if can_add:
                        right.add(v)
                        changed = True
            
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best = (left, right)
    
    # Strategy 2: Try single vertex seeds with multiple right candidates
    for a in sorted_v[:12]:
        if time.time() > deadline:
            break
        # Find neighbors in remaining
        neighbors = [v for v in range(n) if v != a and _edge_key(a, v) in remaining]
        neighbors.sort(key=lambda x: -deg.get(x, 0))
        for b in neighbors[:6]:
            left = {a}
            right = {b}
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                for v in range(n):
                    if v in left or v in right:
                        continue
                    if all(_edge_key(v, r) in remaining for r in right):
                        left.add(v)
                        changed = True
                for v in range(n):
                    if v in left or v in right:
                        continue
                    if all(_edge_key(v, l) in remaining for l in left):
                        right.add(v)
                        changed = True
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best = (left, right)
    
    if best is None:
        e = next(iter(remaining))
        best = ({e[0]}, {e[1]})
    
    return best

def _greedy_partition(adj, deadline, seed=0):
    """Greedy: repeatedly find a large biclique among remaining edges."""
    random.seed(seed)
    remaining = set()
    for a, b in P['edges']:
        remaining.add(_edge_key(a, b))
    
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        
        bl = _find_max_biclique(adj, remaining, deadline)
        if bl is None:
            e = next(iter(remaining))
            bl = ({e[0]}, {e[1]})
        
        l, r = bl
        bicliques.append((list(l), list(r)))
        
        # Remove all cross edges from remaining
        for u in l:
            for v in r:
                remaining.discard(_edge_key(u, v))
    
    return bicliques

def _try_merge_two(b1, b2, adj):
    """Try to merge two bicliques. Returns merged biclique or None."""
    l1, r1 = set(b1[0]), set(b1[1])
    l2, r2 = set(b2[0]), set(b2[1])
    
    candidates = [
        (l1 | l2, r1 & r2),
        (l1 & l2, r1 | r2),
        (l1 | r2, r1 & l2),
        (l2 | r1, r2 & l1),
        (l1 | l2, r1 | r2),
        (l1 & l2, r1 & r2),
    ]
    
    for new_l, new_r in candidates:
        if not new_l or not new_r:
            continue
        if new_l & new_r:
            continue
        if not _is_valid_biclique(new_l, new_r, adj):
            continue
        # Check edge coverage
        new_edges = _biclique_edges(new_l, new_r)
        old_edges = _biclique_edges(l1, r1) | _biclique_edges(l2, r2)
        if new_edges == old_edges:
            return (list(new_l), list(new_r))
    
    return None

def _merge_pass(bicliques, adj, deadline):
    """Repeatedly try to merge pairs of bicliques."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                merged = _try_merge_two(bicliques[i], bicliques[j], adj)
                if merged:
                    bicliques[i] = merged
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _refine_biclique(bi, adj, remaining_all):
    """Try to refine a single biclique by moving vertices between sides."""
    l, r = set(bi[0]), set(bi[1])
    changed = True
    while changed:
        changed = False
        # Try moving a vertex from left to right
        for v in list(l):
            new_l = l - {v}
            new_r = r | {v}
            if not new_l or not new_r:
                continue
            if new_l & new_r:
                continue
            if _is_valid_biclique(new_l, new_r, adj):
                if _biclique_edges(new_l, new_r) == _biclique_edges(l, r):
                    l, r = new_l, new_r
                    changed = True
                    break
        if changed:
            continue
        # Try moving a vertex from right to left
        for v in list(r):
            new_l = l | {v}
            new_r = r - {v}
            if not new_l or not new_r:
                continue
            if new_l & new_r:
                continue
            if _is_valid_biclique(new_l, new_r, adj):
                if _biclique_edges(new_l, new_r) == _biclique_edges(l, r):
                    l, r = new_l, new_r
                    changed = True
                    break
    return (list(l), list(r))

def _refine_pass(bicliques, adj, deadline):
    """Refine each biclique individually."""
    for i in range(len(bicliques)):
        if time.time() > deadline:
            break
        bicliques[i] = _refine_biclique(bicliques[i], adj, set())
    return bicliques

def _to_result(bicliques):
    """Convert to the required format with proper types."""
    result = []
    for l, r in bicliques:
        # Ensure integers are Python int
        l = [int(v) for v in l]
        r = [int(v) for v in r]
        result.append([l, r])
    return result

def _validate_partition(bicliques, adj):
    """Validate that the partition is correct."""
    all_edges = set()
    for a, b in P['edges']:
        all_edges.add(_edge_key(a, b))
    
    covered = set()
    for l, r in bicliques:
        ls, rs = set(l), set(r)
        if not ls or not rs:
            return False
        if ls & rs:
            return False
        for u in ls:
            for v in rs:
                if v not in adj[u]:
                    return False
                covered.add(_edge_key(u, v))
    
    return covered == all_edges

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    adj = _build_adj()
    
    # Run greedy with multiple random restarts
    best_bicliques = None
    best_count = float('inf')
    
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        bl = _greedy_partition(adj, deadline - 15, seed=restart)
        if bl:
            bl = _refine_pass(bl, adj, deadline - 10)
            bl = _merge_pass(bl, adj, deadline - 5)
            bl = _refine_pass(bl, adj, deadline - 3)
            bl = _merge_pass(bl, adj, deadline - 1)
            if len(bl) < best_count:
                best_count = len(bl)
                best_bicliques = bl
    
    if best_bicliques is None:
        # Fallback: star partition (always valid)
        best_bicliques = [[u, [v for a, v in P['edges'] if a == u]] 
                         for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    # Final validation
    if not _validate_partition(best_bicliques, adj):
        # Fallback to star partition
        best_bicliques = [[u, [v for a, v in P['edges'] if a == u]] 
                         for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END

"""Triangle-free diameter-four augmentation via hub-based construction."""
# EVOLVE-BLOCK-START

import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_adjacency(existing_edges, added_edges):
    """Build adjacency list from existing + added edges."""
    n = 24
    adj = {i: set() for i in range(n)}
    for u, v in existing_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_triangle_free(adj):
    """Check triangle-free by verifying bipartiteness (since graph is bipartite if triangle-free for this structure)."""
    n = 24
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
    return True


def _diameter(adj):
    """Compute graph diameter via BFS from each vertex."""
    n = 24
    max_d = 0
    for start in range(n):
        dist = {start: 0}
        queue = [start]
        while queue:
            nxt = []
            for u in queue:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            queue = nxt
        if len(dist) < n:
            return -1
        max_d = max(max_d, max(dist.values()))
    return max_d


def _validate(added_edges):
    """Validate the added edges: triangle-free, connected, diameter <= 4, no duplicates, no existing edges."""
    n = 24
    base_edge_set = set()
    for u, v in P['edges']:
        base_edge_set.add((min(u, v), max(u, v)))
    
    added_edge_set = set()
    for u, v in added_edges:
        if not (isinstance(u, int) and isinstance(v, int)):
            return False
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in added_edge_set:
            return False
        added_edge_set.add(key)
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False
    
    return True


def construct():
    """
    Triangle-free diameter-four augmentation.
    
    The base graph is a path on 24 vertices. All added edges must be between
    even and odd vertices (to maintain triangle-free/bipartite property).
    
    Strategy:
    1. Use minimum dominating set on P_24^2 to find optimal hub placement
    2. For each candidate hub set, compute minimum edges via exact set cover
    3. Validate and return best solution
    """
    deadline = time.time() + 85
    n = 24
    
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    best = [list(e) for e in fallback]
    best_val = len(best)
    
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
    base_edge_set = set((min(u,v), max(u,v)) for u, v in P['edges'])
    
    def fast_validate(added_edges):
        """Fast validation: triangle-free, connected, diameter <= 4."""
        seen = set()
        for u, v in added_edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in base_edge_set or key in seen:
                return False
            seen.add(key)
        
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check
        for a in range(n):
            for b in adj[a]:
                if b > a:
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            return False
        
        # BFS diameter check (early exit if > 4)
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
            for i in range(n):
                if dist[i] == -1:
                    return False
        
        return True
    
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    def min_set_cover_exact(uncovered_elements, candidate_to_elements, max_select=None):
        """Exact minimum set cover via backtracking for small instances."""
        if not uncovered_elements:
            return []
        candidates = list(candidate_to_elements.keys())
        best_solution = None
        
        def backtrack(remaining, selected, idx):
            nonlocal best_solution
            if not remaining:
                if best_solution is None or len(selected) < len(best_solution):
                    best_solution = list(selected)
                return
            if max_select is not None and len(selected) >= max_select:
                return
            # Lower bound: need at least ceil(len(remaining) / max_coverage) more
            if best_solution is not None and len(selected) >= len(best_solution):
                return
            
            # Find the element with fewest covering candidates (most constrained)
            min_count = float('inf')
            min_elem = None
            for elem in remaining:
                count = sum(1 for c in candidates if elem in candidate_to_elements[c])
                if count < min_count:
                    min_count = count
                    min_elem = elem
                if count == 0:
                    return  # Can't cover this element
            
            # Try each candidate that covers min_elem
            for c in candidates:
                if min_elem in candidate_to_elements[c] and c not in selected:
                    new_remaining = remaining - candidate_to_elements[c]
                    selected.append(c)
                    backtrack(new_remaining, selected, idx + 1)
                    selected.pop()
        
        backtrack(set(uncovered_elements), [], 0)
        return best_solution
    
    def min_set_cover_greedy(uncovered_elements, candidate_to_elements):
        """Greedy set cover."""
        if not uncovered_elements:
            return []
        uncovered = set(uncovered_elements)
        selected = []
        candidates = list(candidate_to_elements.keys())
        while uncovered:
            best_c = None
            best_count = -1
            for c in candidates:
                if c in selected:
                    continue
                count = len(uncovered & candidate_to_elements[c])
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None or best_count == 0:
                return None
            selected.append(best_c)
            uncovered -= candidate_to_elements[best_c]
        return selected
    
    # Phase 1: Single hub search with exact set cover
    for h in even_verts:
        if time.time() > deadline:
            break
        # h (even) connects to odd vertices
        h_odd_neighbors = set()
        if h - 1 >= 0:
            h_odd_neighbors.add(h - 1)
        if h + 1 < n:
            h_odd_neighbors.add(h + 1)
        
        # For each even e != h, need an odd neighbor connected to h
        evens_needing = set()
        for e in even_verts:
            if e == h:
                continue
            e_odd_neighbors = set()
            if e - 1 >= 0:
                e_odd_neighbors.add(e - 1)
            if e + 1 < n:
                e_odd_neighbors.add(e + 1)
            if not (e_odd_neighbors & h_odd_neighbors):
                evens_needing.add(e)
        
        # For each odd o, which even vertices does it cover?
        odd_to_evens = {}
        for o in odd_verts:
            if o in h_odd_neighbors:
                continue
            evens_covered = set()
            for e in evens_needing:
                if e - 1 == o or e + 1 == o:
                    evens_covered.add(e)
            if evens_covered:
                odd_to_evens[o] = evens_covered
        
        if not evens_needing:
            config = []
        else:
            # Try exact first if small enough
            if len(odd_to_evens) <= 15:
                sol = min_set_cover_exact(evens_needing, odd_to_evens)
            else:
                sol = min_set_cover_greedy(evens_needing, odd_to_evens)
            if sol is None:
                continue
            config = [[h, o] for o in sol]
        
        if len(config) < best_val and fast_validate(config):
            best = config
            best_val = len(best)
    
    for h in odd_verts:
        if time.time() > deadline:
            break
        h_even_neighbors = set()
        if h - 1 >= 0:
            h_even_neighbors.add(h - 1)
        if h + 1 < n:
            h_even_neighbors.add(h + 1)
        
        odds_needing = set()
        for o in odd_verts:
            if o == h:
                continue
            o_even_neighbors = set()
            if o - 1 >= 0:
                o_even_neighbors.add(o - 1)
            if o + 1 < n:
                o_even_neighbors.add(o + 1)
            if not (o_even_neighbors & h_even_neighbors):
                odds_needing.add(o)
        
        even_to_odds = {}
        for e in even_verts:
            if e in h_even_neighbors:
                continue
            odds_covered = set()
            for o in odds_needing:
                if o - 1 == e or o + 1 == e:
                    odds_covered.add(o)
            if odds_covered:
                even_to_odds[e] = odds_covered
        
        if not odds_needing:
            config = []
        else:
            if len(even_to_odds) <= 15:
                sol = min_set_cover_exact(odds_needing, even_to_odds)
            else:
                sol = min_set_cover_greedy(odds_needing, even_to_odds)
            if sol is None:
                continue
            config = [[h, e] for e in sol]
        
        if len(config) < best_val and fast_validate(config):
            best = config
            best_val = len(best)
    
    # Phase 2: Two-hub search (one even, one odd)
    for h1 in even_verts:
        if time.time() > deadline:
            break
        for h2 in odd_verts:
            if time.time() > deadline:
                break
            
            # Check if h1 and h2 are path-adjacent
            adjacent = (min(h1,h2), max(h1,h2)) in base_edge_set
            
            # h1 (even) covers even vertices via odd neighbors
            h1_odd_neighbors = set()
            if h1 - 1 >= 0:
                h1_odd_neighbors.add(h1 - 1)
            if h1 + 1 < n:
                h1_odd_neighbors.add(h1 + 1)
            
            # h2 (odd) covers odd vertices via even neighbors
            h2_even_neighbors = set()
            if h2 - 1 >= 0:
                h2_even_neighbors.add(h2 - 1)
            if h2 + 1 < n:
                h2_even_neighbors.add(h2 + 1)
            
            # Even vertices needing coverage from h1
            evens_needing_h1 = set()
            for e in even_verts:
                if e == h1:
                    continue
                e_odd_neighbors = set()
                if e - 1 >= 0:
                    e_odd_neighbors.add(e - 1)
                if e + 1 < n:
                    e_odd_neighbors.add(e + 1)
                if not (e_odd_neighbors & h1_odd_neighbors):
                    evens_needing_h1.add(e)
            
            # Odd vertices needing coverage from h2
            odds_needing_h2 = set()
            for o in odd_verts:
                if o == h2:
                    continue
                o_even_neighbors = set()
                if o - 1 >= 0:
                    o_even_neighbors.add(o - 1)
                if o + 1 < n:
                    o_even_neighbors.add(o + 1)
                if not (o_even_neighbors & h2_even_neighbors):
                    odds_needing_h2.add(o)
            
            # Build set cover for h1
            odd_to_evens = {}
            for o in odd_verts:
                if o in h1_odd_neighbors:
                    continue
                evens_covered = set()
                for e in evens_needing_h1:
                    if e - 1 == o or e + 1 == o:
                        evens_covered.add(e)
                if evens_covered:
                    odd_to_evens[o] = evens_covered
            
            # Build set cover for h2
            even_to_odds = {}
            for e in even_verts:
                if e in h2_even_neighbors:
                    continue
                odds_covered = set()
                for o in odds_needing_h2:
                    if o - 1 == e or o + 1 == e:
                        odds_covered.add(o)
                if odds_covered:
                    even_to_odds[e] = odds_covered
            
            # Solve set covers
            h1_sol = min_set_cover_greedy(evens_needing_h1, odd_to_evens) if evens_needing_h1 else []
            h2_sol = min_set_cover_greedy(odds_needing_h2, even_to_odds) if odds_needing_h2 else []
            
            if h1_sol is None or h2_sol is None:
                continue
            
            total = len(h1_sol) + len(h2_sol) + (0 if adjacent else 1)
            if total >= best_val:
                continue
            
            config = []
            for o in h1_sol:
                config.append([h1, o])
            for e in h2_sol:
                config.append([h2, e])
            if not adjacent:
                config.append([h1, h2])
            
            if len(config) < best_val and fast_validate(config):
                best = config
                best_val = len(best)
    
    # Phase 3: Try removing edges from current best
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:i] + best[i+1:]
            if fast_validate(trial):
                best = trial
                best_val = len(best)
                changed = True
                break
    
    # Phase 4: Try replacing pairs of edges with single edges
    for i in range(len(best)):
        if time.time() > deadline:
            break
        for j in range(i+1, len(best)):
            if time.time() > deadline:
                break
            # Remove edges i and j, try adding one new edge
            remaining = best[:i] + best[i+1:j] + best[j+1:]
            # Try all possible single edges
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if time.time() > deadline:
                        break
                    key = (min(u,v), max(u,v))
                    if key in base_edge_set:
                        continue
                    if (u % 2) == (v % 2):
                        continue  # Must be bipartite
                    trial = remaining + [[u, v]]
                    if len(trial) < best_val and fast_validate(trial):
                        best = trial
                        best_val = len(best)
    
    # Final validation
    if not fast_validate(best):
        best = [list(e) for e in fallback]
    
    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END

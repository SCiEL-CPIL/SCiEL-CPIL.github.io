import time
import random
import sys

_DEADLINE = 115.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q):
    """Construct the polarity graph of the projective plane PG(2, q)."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    return adj

def _has_indep_11(adj):
    """Check if there is an independent set of size 11."""
    n = len(adj)
    if n < 11:
        return False
        
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)
    
    sys.setrecursionlimit(10000)
    
    def branch(remaining_mask, current_size):
        if current_size >= 11:
            return True
        if remaining_mask == 0:
            return False
        if current_size + remaining_mask.bit_count() < 11:
            return False
        
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = (remaining_mask & masks[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return False

        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _get_conflict_masks(adj, n):
    """
    Returns a list of bitmasks for the conflict graph.
    conflict_mask[i] has bit j set if i and j share a common neighbor in adj.
    """
    conflict_masks = [0] * n
    for k in range(n):
        neighbors = list(adj[k])
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i+1, len(neighbors)):
                v = neighbors[j]
                conflict_masks[u] |= (1 << v)
                conflict_masks[v] |= (1 << u)
                
    return conflict_masks

def _find_conflict_independent_set(conflict_masks, n, seed=0):
    """
    Find an independent set in the graph defined by conflict_masks.
    Independent set in conflict graph = set of vertices no two of which share a common neighbor.
    """
    remaining = (1 << n) - 1
    independent_set = []
    
    while remaining:
        best_v = -1
        best_deg = 10**9
        
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (conflict_masks[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            break
            
        independent_set.append(best_v)
        remaining &= ~(conflict_masks[best_v] | (1 << best_v))
        
    return independent_set

def _alpha_greedy_lb(adj):
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 115.0
    
    # 1. Start with PG(2,5) polarity graph (31 vertices)
    # This graph is C4-free and has alpha ~ 10.
    adj = _build_polarity_graph(5)
    n = len(adj)
    
    # Verify initial graph
    if _has_indep_11(adj):
        # Fallback to a known valid smaller graph if PG(2,5) fails (unlikely)
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # 2. Main expansion loop
    # Goal: Add vertices one by one.
    # When adding a new vertex 'v', we choose its neighborhood N(v) from existing vertices.
    # Constraint 1 (C4-free): N(v) must be an independent set in the "conflict graph".
    #   The conflict graph has edges between u, w if N(u) and N(w) are not disjoint.
    #   If u, w in N(v) share a neighbor x, then v-u-x-w-v is a C4.
    # Constraint 2 (No alpha 11): The new graph must not have an independent set of size 11.
    #   Adding a vertex v with neighborhood S.
    #   Any independent set in the new graph either:
    #   a) Includes v. Then it can include vertices from V \ (N(v) U {v}).
    #      This set is V \ N(v) (since v is not in N(v)).
    #      Let alpha_old be the max independent set size in old graph.
    #      Let alpha_excl_v be the max independent set size in G[V \ N(v)].
    #      Then alpha_new <= max(alpha_old, 1 + alpha_excl_v).
    #   b) Excludes v. Then it is an independent set in the old graph.
    #      So alpha_new <= max(alpha_old, 1 + alpha_excl_v).
    #   We want alpha_new < 11.
    #   Since alpha_old < 11 (we maintain this invariant), we need 1 + alpha_excl_v < 11 => alpha_excl_v <= 9.
    #   So we need to choose S (neighborhood of v) such that the subgraph induced by V \ S has alpha <= 9.
    #   Equivalently, S must be a "9-dominating" set in the sense that V\S has no independent set of size 10.
    #   Actually, simpler: Just check _has_indep_11 on the new graph.
    #   But that's expensive.
    #   Heuristic: S should be large and well-distributed to break up independent sets.
    #   Greedy approach for S:
    #   We want S to be an independent set in the conflict graph.
    #   We also want S to be "large" to reduce the size of V\S.
    #   Let's try to find a large independent set in the conflict graph.
    #   If we can find one of size k, then V\S has size n-k.
    #   If n-k is small enough, alpha(V\S) is likely small.
    
    while _elapsed() < 100.0:
        conflict_masks = _get_conflict_masks(adj, n)
        
        # Try to find a large independent set in conflict graph
        # Greedy: pick vertex with min degree in remaining conflict graph
        # We want to maximize the size of S to minimize V\S.
        # But we also need V\S to have low alpha.
        
        # Strategy:
        # 1. Find a maximal independent set S in conflict graph.
        # 2. Check if adding a vertex connected to S keeps alpha < 11.
        # 3. If yes, add it.
        # 4. If no, try to modify S (e.g., remove a vertex from S to increase V\S? No, that makes V\S larger.
        #    We want V\S to have low alpha.
        #    If alpha(V\S) is too high, we need S to "cover" more of the independent sets.
        #    This is hard to control directly.
        #    Let's just try random maximal independent sets in conflict graph.
        
        added = False
        for attempt in range(50):
            if _deadline_hit():
                break
            
            # Generate a random maximal independent set in conflict graph
            remaining = (1 << n) - 1
            S = []
            
            while remaining:
                # Pick a random vertex from remaining with low degree?
                # Or just random?
                # Low degree is better for maximality (greedy).
                # Let's use the greedy min-degree but with random tie-breaking.
                candidates = []
                min_deg = 10**9
                m = remaining
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    deg = (conflict_masks[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                S.append(v)
                remaining &= ~(conflict_masks[v] | (1 << v))
            
            if not S:
                continue
            
            # Create new graph
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(n)
            
            # Check C4 (should be satisfied by construction, but verify just in case)
            # Actually, construction guarantees no C4 involving the new vertex.
            # Old C4s are preserved.
            
            # Check alpha
            if not _has_indep_11(new_adj):
                # Valid!
                adj = new_adj
                n = new_n
                added = True
                best_n = n
                best_adj = [set(s) for s in adj]
                break
        
        if not added:
            # Try to find a different structure?
            # Maybe try removing an edge? No, we want to maximize n.
            # If we can't add a vertex, we're stuck.
            # Break to return best result.
            break
        
        # If we reach n=43, stop
        if n >= 43:
            break
    
    # Final verification
    if _has_indep_11(adj):
        # This shouldn't happen if we check every step, but just in case
        # Fall back to best_adj which was verified
        adj = best_adj
        n = len(adj)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}

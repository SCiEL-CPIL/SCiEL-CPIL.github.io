"""Triangle-free graphs without independent 10-sets (R(3,10) > n).

Strategy:
  1. Start from the known Andrasfai graph And(9) on 26 vertices.
  2. Extend by adding vertices one at a time, choosing neighbor sets that
     preserve triangle-freeness and keep alpha < 10.
  3. Use fast bitset-based validators for triangle-freeness and alpha.
  4. Use multiple heuristics and random restarts to find valid extensions.
  5. Deadline guard to stay within 150s wall-clock limit.
"""
import time
import random
import sys

DEADLINE = 140.0  # safety margin before 150s limit
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > DEADLINE

def _has_triangle(adj, n):
    """Check if graph with adjacency bitmasks has a triangle."""
    for i in range(n):
        ai = adj[i]
        # For each neighbor j of i, check if any common neighbor k exists
        j = ai
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            # common neighbors of i and j (excluding i and j themselves)
            common = ai & adj[jj] & ~(1 << i) & ~(1 << jj)
            if common:
                return True
            j ^= lb
    return False

def _max_independent_set_size(adj, n, limit=10):
    """Branch and bound: return size of max independent set, capped at limit."""
    best = [0]
    
    def _bbs(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Pick a vertex with highest degree in the candidate subgraph for better pruning
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        # Exclude v
        _bbs(chosen_count, rest)
        # Include v: new candidates = rest & ~adj[v]
        new_cands = rest & ~adj[v]
        _bbs(chosen_count + 1, new_cands)
    
    _bbs(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle(adj, n):
        return False
    if _max_independent_set_size(adj, n, limit=10) >= 10:
        return False
    return True

def _andrasfai_9():
    """Return adj bitmasks for And(9) on 26 vertices."""
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def _find_large_independent_set(adj, n, target_size, random_order=None):
    """Greedy: find an independent set of size >= target_size."""
    if random_order is None:
        random_order = list(range(n))
    random.shuffle(random_order)
    I = []
    for v in random_order:
        conflict = False
        for u in I:
            if adj[v] & (1 << u):
                conflict = True
                break
        if not conflict:
            I.append(v)
            if len(I) >= target_size:
                return I
    return I

def _try_add_vertex(adj, n, deadline=DEADLINE):
    """Try to add one vertex to the graph. Returns (new_adj, new_n) or (None, n)."""
    if _deadline_exceeded():
        return None, n
    
    # The new vertex's neighbors must form an independent set in G
    # (to avoid triangles). We want to choose N(new) such that:
    #   1. N(new) is an independent set in G (triangle-free)
    #   2. alpha(V \ N(new)) + 1 < 10, i.e., alpha(V \ N(new)) <= 8
    
    # Strategy: try various independent sets as N(new)
    # Larger N(new) means smaller V \ N(new), which helps alpha constraint
    # but we need N(new) to be independent.
    
    best_result = None
    
    for attempt in range(300):
        if _deadline_exceeded():
            break
        
        # Find a random independent set of moderate size
        target = random.randint(5, min(15, n))
        I = _find_large_independent_set(adj, n, target)
        
        if len(I) < 3:
            continue
        
        # New vertex n connects to all in I
        new_adj = list(adj)
        new_adj.append(0)
        for i in range(n):
            if i in I:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        
        if _is_valid(new_adj, n + 1):
            return new_adj, n + 1
    
    # Try smaller neighbor sets
    for attempt in range(200):
        if _deadline_exceeded():
            break
        
        target = random.randint(2, min(8, n))
        I = _find_large_independent_set(adj, n, target)
        
        if len(I) < 2:
            continue
        
        new_adj = list(adj)
        new_adj.append(0)
        for i in range(n):
            if i in I:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        
        if _is_valid(new_adj, n + 1):
            return new_adj, n + 1
    
    return None, n

def _try_edge_flip(adj, n):
    """Try flipping one edge to potentially enable future extensions.
    This is a local search move."""
    if _deadline_exceeded():
        return None
    
    # Try removing a random edge
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                all_edges.append((i, j))
    
    if not all_edges:
        return None
    
    u, v = random.choice(all_edges)
    new_adj = list(adj)
    new_adj[u] &= ~(1 << v)
    new_adj[v] &= ~(1 << u)
    
    if _is_valid(new_adj, n):
        return new_adj
    
    # Try adding a random non-edge
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                non_edges.append((i, j))
    
    if not non_edges:
        return None
    
    u, v = random.choice(non_edges)
    # Check triangle-freeness: u and v must not share a common neighbor
    common = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
    if common:
        return None
    
    new_adj = list(adj)
    new_adj[u] |= (1 << v)
    new_adj[v] |= (1 << u)
    
    if _is_valid(new_adj, n):
        return new_adj
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Start from And(9)
    adj, n = _andrasfai_9()
    best_n = n
    best_adj = list(adj)
    
    # Verify starting graph is valid
    if not _is_valid(adj, n):
        # Shouldn't happen, but fallback
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
    
    # Try to extend the graph
    for target_n in range(n + 1, 41):
        if _deadline_exceeded():
            break
        
        # Try multiple times to add a vertex
        added = False
        for attempt in range(50):
            if _deadline_exceeded():
                break
            
            new_adj, new_n = _try_add_vertex(adj, n)
            if new_adj is not None:
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = list(adj)
                added = True
                break
        
        if not added:
            # Try local search to modify the graph, then retry
            for ls_attempt in range(30):
                if _deadline_exceeded():
                    break
                
                flipped = _try_edge_flip(adj, n)
                if flipped is not None:
                    adj = flipped
                    # Now try to add a vertex again
                    new_adj, new_n = _try_add_vertex(adj, n)
                    if new_adj is not None:
                        adj = new_adj
                        n = new_n
                        best_n = n
                        best_adj = list(adj)
                        added = True
                        break
            
            if not added:
                # Try a fresh random restart from a perturbed version
                for restart in range(10):
                    if _deadline_exceeded():
                        break
                    
                    # Perturb: remove some random edges
                    perturbed = list(adj)
                    for _ in range(random.randint(2, 5)):
                        all_edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if perturbed[i] & (1 << j):
                                    all_edges.append((i, j))
                        if all_edges:
                            u, v = random.choice(all_edges)
                            perturbed[u] &= ~(1 << v)
                            perturbed[v] &= ~(1 << u)
                    
                    if _is_valid(perturbed, n):
                        adj = perturbed
                        new_adj, new_n = _try_add_vertex(adj, n)
                        if new_adj is not None:
                            adj = new_adj
                            n = new_n
                            best_n = n
                            best_adj = list(adj)
                            added = True
                            break
            
            if not added:
                break  # Can't extend further, stop
    
    # Return the best valid graph found
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

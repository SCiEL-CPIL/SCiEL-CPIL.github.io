"""Structural change: use a fast, deterministic 3-layer C6 test on 40-bit masks,
seed with 8 K5s, build greedily, then run a targeted 2-opt local search that
removes an edge and tries to replace it (and possibly add more) until C6-free.
"""
import time
import random

N = 40
ALL_MASK = (1 << N) - 1

def _c6_after_add(adj, u, v):
    """Return True if adding edge (u,v) creates a 6-cycle.
    We need a path of length 4 between u and v that avoids u and v internally.
    Path: u-a-b-c-d-v with all internal distinct and not u/v.
    """
    if u == v:
        return False
    # Neighbors of u excluding v (a candidates)
    a_candidates = adj[u] & ~(1 << v) & ~(1 << u)
    # Neighbors of v excluding u (d candidates)
    d_candidates = adj[v] & ~(1 << u) & ~(1 << v)
    
    # For each a in a_candidates:
    while a_candidates:
        lsb = a_candidates & -a_candidates
        a = lsb.bit_length() - 1
        a_candidates ^= lsb
        
        # b must be neighbor of a, not u, not v, not a
        b_candidates = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
        while b_candidates:
            lsb2 = b_candidates & -b_candidates
            b = lsb2.bit_length() - 1
            b_candidates ^= lsb2
            
            # c must be neighbor of b, not u, not v, not a, not b
            c_candidates = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
            while c_candidates:
                lsb3 = c_candidates & -c_candidates
                c = lsb3.bit_length() - 1
                c_candidates ^= lsb3
                
                # d must be neighbor of c, not u, not v, not a, not b, not c
                d_candidates_c = adj[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                if d_candidates_c & d_candidates:
                    return True
    return False

def _seed_8k5():
    """Return adjacency list as list of ints (bitmasks) for 8 disjoint K5s."""
    adj = [0] * N
    edges = []
    for block in range(8):
        base = block * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
    return adj, edges

def _greedy_add_edges(adj, edges, deadline, rng=None):
    """Greedily add edges that don't create C6, prioritizing low-degree pairs."""
    if rng is None:
        rng = random.Random(0)
    
    # Build list of candidate edges not yet present
    edge_set = set(edges)
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edge_set:
                deg_u = bin(adj[u]).count('1')
                deg_v = bin(adj[v]).count('1')
                candidates.append((deg_u + deg_v, u, v))
    
    candidates.sort(key=lambda x: x[0])
    
    added = 0
    for _, u, v in candidates:
        if time.time() > deadline:
            break
        if not _c6_after_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            edges.append((u, v))
            added += 1
            # Refresh candidates for edges involving u or v with new degrees
            # This is expensive; for simplicity we don't update in-place.
            # Instead we rely on local search later.
    
    return added

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _local_search(adj, edges, deadline, rng):
    """2-opt local search: remove an edge, try to replace it and add more."""
    edge_set = set(edges)
    current_edges = len(edges)
    best_edges = list(edges)
    best_count = current_edges
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        if not edge_set:
            break
        
        # Pick a random edge to remove
        rem_edge = rng.choice(list(edge_set))
        ru, rv = rem_edge
        
        # Remove it
        _remove_edge(adj, ru, rv)
        edge_set.discard(rem_edge)
        current_edges -= 1
        
        # Try to add back a different edge, and then try to add more
        improved = False
        
        # First, try to find an edge we can add
        # Try random non-edges, prioritizing those that might unlock more
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            u = rng.randint(0, N - 1)
            v = rng.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
            if _c6_after_add(adj, u, v):
                continue
            
            # Add this edge
            _add_edge(adj, u, v)
            edge_set.add((u, v))
            current_edges += 1
            improved = True
            
            # Now try to add more edges greedily (quick pass)
            for _ in range(20):
                if time.time() > deadline:
                    break
                u2 = rng.randint(0, N - 1)
                v2 = rng.randint(0, N - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if (u2, v2) in edge_set:
                    continue
                if _c6_after_add(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                edge_set.add((u2, v2))
                current_edges += 1
            
            break
        
        if improved:
            if current_edges > best_count:
                best_count = current_edges
                best_edges = list(edge_set)
                no_improve = 0
            else:
                # Revert to previous best
                adj = [0] * N
                edge_set = set(best_edges)
                for u, v in best_edges:
                    _add_edge(adj, u, v)
                current_edges = best_count
                no_improve += 1
        else:
            # Couldn't find replacement; put edge back
            _add_edge(adj, ru, rv)
            edge_set.add(rem_edge)
            current_edges += 1
            no_improve += 1
    
    # Reconstruct edge list from best
    best_adj = [0] * N
    best_edge_set = set(best_edges)
    for u, v in best_edges:
        _add_edge(best_adj, u, v)
    
    return best_adj, list(best_edge_set), best_count

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_edges_overall = []
    best_count_overall = 0
    
    # Multi-start with different seeds
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(42 + trial * 12345)
        adj, edges = _seed_8k5()
        
        # Greedy phase
        _greedy_add_edges(adj, edges, deadline - 10, rng)
        
        # Local search phase
        ls_deadline = min(deadline - 2, start_time + (deadline - start_time - 5) * (trial + 1) / 10)
        adj, edges, count = _local_search(adj, edges, ls_deadline, rng)
        
        if count > best_count_overall:
            best_count_overall = count
            best_edges_overall = list(edges)
    
    # Final verification
    final_adj = [0] * N
    final_edges = set(best_edges_overall)
    for u, v in final_edges:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    # Check no C6 by testing each edge removal
    has_c6 = False
    for u, v in list(final_edges):
        final_adj[u] &= ~(1 << v)
        final_adj[v] &= ~(1 << u)
        if _c6_after_add(final_adj, u, v):
            has_c6 = True
            break
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    if has_c6:
        # Fallback to 8 K5s
        fallback = []
        for block in range(8):
            base = block * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    fallback.append([base + i, base + j])
        return fallback
    
    return [[u, v] for u, v in best_edges_overall]

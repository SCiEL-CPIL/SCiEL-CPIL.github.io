"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_bitmask(adj_mask, n, u, v):
    """Fast check if adding edge (u,v) creates a C6 using bitmasks.
    
    A C6 is formed if there exists a simple path of length 5 from u to v:
    u - a - b - c - d - v
    
    We enumerate paths of length 5 from u to v using bitmask operations.
    """
    mask_u = adj_mask[u]
    mask_v = adj_mask[v]
    
    # Path: u - a - b - c - d - v
    # a in N(u), d in N(v), need path a-b-c-d of length 3 in G - {u,v}
    # Enumerate a in N(u)
    a_set = mask_u & ~(1 << v)  # a != v
    while a_set:
        a_bit = a_set & (-a_set)
        a = a_bit.bit_length() - 1
        a_set ^= a_bit
        
        # b in N(a) \ {u, v, a}
        b_set = adj_mask[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
        while b_set:
            b_bit = b_set & (-b_set)
            b = b_bit.bit_length() - 1
            b_set ^= b_bit
            
            # c in N(b) \ {u, v, a, b}
            c_set = adj_mask[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
            while c_set:
                c_bit = c_set & (-c_set)
                c = c_bit.bit_length() - 1
                c_set ^= c_bit
                
                # d in N(c) \ {u, v, a, b, c} AND d in N(v)
                d_set = adj_mask[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c) & mask_v
                if d_set:
                    return True
    return False

def _creates_c6_fast(adj, n, u, v):
    """Fast check if adding edge (u,v) creates a C6."""
    adj_u = adj[u]
    adj_v = adj[v]
    
    # Enumerate: u - a - b - c - d - v
    for a in adj_u:
        if a == v:
            continue
        adj_a = adj[a]
        for b in adj_a:
            if b == u or b == v or b == a:
                continue
            adj_b = adj[b]
            for c in adj_b:
                if c == u or c == v or c == a or c == b:
                    continue
                adj_c = adj[c]
                for d in adj_c:
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if d in adj_v:
                        return True
    return False

def _greedy_build(n, rng, deadline, use_bitmask=False):
    """Build a C6-free graph greedily with a given seed."""
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    # Generate candidate edges and sort by a heuristic
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                # Heuristic: prefer edges between vertices with lower degrees
                candidates.append((len(adj[u]) + len(adj[v]), rng.random(), u, v))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    check_fn = _creates_c6_bitmask if use_bitmask else _creates_c6_fast
    adj_arg = adj_mask if use_bitmask else adj
    
    for _, _, u, v in candidates:
        if time.time() > deadline:
            break
        if not check_fn(adj_arg, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    return edges, adj, adj_mask

def _local_search(edges, adj, adj_mask, n, deadline, rng, use_bitmask=False):
    """Local search: try swap moves (remove one edge, add another)."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_adj_mask = adj_mask.copy()
    
    # Precompute all possible edges
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add((u, v))
    
    no_improve = 0
    max_no_improve = 300
    
    check_fn = _creates_c6_bitmask if use_bitmask else _creates_c6_fast
    adj_arg = adj_mask if use_bitmask else adj
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edges:
            break
        removed = rng.choice(list(edges))
        ru, rv = removed
        
        # Try to find a replacement edge
        added = None
        # Try random non-edges
        for _ in range(150):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                continue
            if (u, v) == removed:
                continue
            if not check_fn(adj_arg, n, u, v):
                added = (u, v)
                break
        
        if added is None:
            no_improve += 1
            continue
        
        au, av = added
        
        # Perform the swap
        edges.discard(removed)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        adj_mask[ru] &= ~(1 << rv)
        adj_mask[rv] &= ~(1 << ru)
        
        edges.add(added)
        adj[au].add(av)
        adj[av].add(au)
        adj_mask[au] |= (1 << av)
        adj_mask[av] |= (1 << au)
        
        new_count = len(edges)
        
        if new_count >= best_count:
            if new_count > best_count:
                best_count = new_count
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
                best_adj_mask = adj_mask.copy()
                no_improve = 0
        else:
            # Revert
            edges.discard(added)
            adj[au].discard(av)
            adj[av].discard(au)
            adj_mask[au] &= ~(1 << av)
            adj_mask[av] &= ~(1 << au)
            edges.add(removed)
            adj[ru].add(rv)
            adj[rv].add(ru)
            adj_mask[ru] |= (1 << rv)
            adj_mask[rv] |= (1 << ru)
            no_improve += 1
    
    return best_edges, best_adj, best_adj_mask, best_count

def _simulated_annealing(edges, adj, adj_mask, n, deadline, rng):
    """Simulated annealing: flip edge variables to maximize edges while staying C6-free."""
    current_edges = edges.copy()
    current_adj = [s.copy() for s in adj]
    current_adj_mask = adj_mask.copy()
    
    best_edges = current_edges.copy()
    best_adj = [s.copy() for s in current_adj]
    best_adj_mask = current_adj_mask.copy()
    best_count = len(current_edges)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    no_improve = 0
    max_no_improve = 500
    
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add((u, v))
    
    while time.time() < deadline and T > T_min and no_improve < max_no_improve:
        # Pick a random non-edge to add
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in current_edges:
            continue
        
        # Check if adding (u,v) creates a C6
        if _creates_c6_bitmask(current_adj_mask, n, u, v):
            # Try removing a random edge and adding (u,v)
            if not current_edges:
                continue
            removed = rng.choice(list(current_edges))
            ru, rv = removed
            
            # Remove the edge first
            current_edges.discard(removed)
            current_adj[ru].discard(rv)
            current_adj[rv].discard(ru)
            current_adj_mask[ru] &= ~(1 << rv)
            current_adj_mask[rv] &= ~(1 << ru)
            
            # Check if we can add (u,v)
            if not _creates_c6_bitmask(current_adj_mask, n, u, v):
                current_edges.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_adj_mask[u] |= (1 << v)
                current_adj_mask[v] |= (1 << u)
                
                new_count = len(current_edges)
                if new_count > best_count:
                    best_count = new_count
                    best_edges = current_edges.copy()
                    best_adj = [s.copy() for s in current_adj]
                    best_adj_mask = current_adj_mask.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_edges.add(removed)
                current_adj[ru].add(rv)
                current_adj[rv].add(ru)
                current_adj_mask[ru] |= (1 << rv)
                current_adj_mask[rv] |= (1 << ru)
                no_improve += 1
        else:
            # Can add directly
            current_edges.add((u, v))
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_adj_mask[u] |= (1 << v)
            current_adj_mask[v] |= (1 << u)
            
            new_count = len(current_edges)
            if new_count > best_count:
                best_count = new_count
                best_edges = current_edges.copy()
                best_adj = [s.copy() for s in current_adj]
                best_adj_mask = current_adj_mask.copy()
                no_improve = 0
            else:
                no_improve += 1
        
        T *= cooling
    
    return best_edges, best_adj, best_adj_mask, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    best_adj_overall = []
    best_adj_mask_overall = []
    
    # Multi-start local search
    num_starts = 12
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 1000
        rng = random.Random(seed_start)
        edges, adj, adj_mask = _greedy_build(n, rng, deadline - 5, use_bitmask=True)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
            best_adj_overall = [s2.copy() for s2 in adj]
            best_adj_mask_overall = adj_mask.copy()
        
        # Local search with remaining time
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1) * 0.6)
        edges, adj, adj_mask, count = _local_search(edges, adj, adj_mask, n, ls_deadline, rng, use_bitmask=True)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
            best_adj_overall = [s2.copy() for s2 in adj]
            best_adj_mask_overall = adj_mask.copy()
        
        # Simulated annealing with remaining time
        sa_deadline = min(deadline - 2, start_time + time_per_start * (s + 1) * 0.9)
        edges, adj, adj_mask, count = _simulated_annealing(edges, adj, adj_mask, n, sa_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
            best_adj_overall = [s2.copy() for s2 in adj]
            best_adj_mask_overall = adj_mask.copy()
    
    # Final verification: make sure no C6 exists
    final_adj = [set() for _ in range(n)]
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    # Verify C6-free by checking all potential 6-cycles
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in final_adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge and check
            final_adj[u].discard(v)
            final_adj[v].discard(u)
            if _creates_c6_fast(final_adj, n, u, v):
                has_c6 = True
            final_adj[u].add(v)
            final_adj[v].add(u)
    
    if has_c6:
        # This shouldn't happen, but fallback to just the K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END

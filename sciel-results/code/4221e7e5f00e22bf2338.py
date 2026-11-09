"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_fast(adj_bits, n, u, v):
    """Fast C6 check using bitset operations.
    
    Path: u - a - b - c - d - v (all distinct)
    Uses bitset intersection to check existence of length-5 path.
    """
    nu = adj_bits[u]
    nv = adj_bits[v]
    
    # Exclude u and v from all intermediate vertices
    mask_uv = ~(1 << u) & ~(1 << v)
    a_set = nu & mask_uv
    
    while a_set:
        a = (a_set & (-a_set)).bit_length() - 1
        a_set &= a_set - 1
        
        na = adj_bits[a]
        # b must be in N(a), not u, v, or a
        mask_uva = mask_uv & ~(1 << a)
        b_set = na & mask_uva
        
        while b_set:
            b = (b_set & (-b_set)).bit_length() - 1
            b_set &= b_set - 1
            
            nb = adj_bits[b]
            # c must be in N(b), not u, v, a, or b
            mask_uvab = mask_uva & ~(1 << b)
            c_set = nb & mask_uvab
            
            while c_set:
                c = (c_set & (-c_set)).bit_length() - 1
                c_set &= c_set - 1
                
                nc = adj_bits[c]
                # d must be in N(c), not u, v, a, b, or c
                mask_uvabc = mask_uvab & ~(1 << c)
                
                # Check if any d in N(c) \ {u,v,a,b,c} is in N(v)
                if (nc & mask_uvabc) & nv:
                    return True
    
    return False

def _greedy_build(n, rng, deadline):
    """Build a C6-free graph greedily with a given seed."""
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    # Generate candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidates.append((u, v))
    
    # Use iterative improvement: repeatedly try to add edges
    # Sort by degree sum for initial ordering (prefer connecting low-degree vertices)
    candidates.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
    
    # Multiple passes to allow for changes
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Re-sort candidates by current degree sum
        candidates.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            if (u, v) in edges:
                continue
            if not _creates_c6_fast(adj_bits, n, u, v):
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                progress = True
    
    return edges, adj, adj_bits

def _local_search(edges, adj, adj_bits, n, deadline, rng):
    """Local search with 1-opt swaps and 2-opt moves (remove 1, add 2)."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_adj_bits = adj_bits[:]
    
    no_improve = 0
    max_no_improve = 150
    
    # Maintain a list of edges for fast random selection
    edge_list = list(edges)
    
    # Build list of all non-edges
    all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try pure add first
        added = None
        for _ in range(30):
            if not all_non_edges:
                break
            idx = rng.randrange(len(all_non_edges))
            u, v = all_non_edges[idx]
            if not _creates_c6_fast(adj_bits, n, u, v):
                added = (u, v)
                break
        
        if added is not None:
            au, av = added
            for i, e in enumerate(all_non_edges):
                if e == added:
                    all_non_edges.pop(i)
                    break
            edges.add(added)
            adj[au].add(av)
            adj[av].add(au)
            adj_bits[au] |= (1 << av)
            adj_bits[av] |= (1 << au)
            edge_list.append(added)
            no_improve = 0
            continue
        
        # Try 2-opt: remove 1 edge, add 2 non-edges
        two_opt_success = False
        if edge_list and len(all_non_edges) >= 2:
            removed_idx = rng.randrange(len(edge_list))
            removed = edge_list[removed_idx]
            ru, rv = removed
            
            # Temporarily remove the edge
            edges.discard(removed)
            adj[ru].discard(rv)
            adj[rv].discard(ru)
            adj_bits[ru] &= ~(1 << rv)
            adj_bits[rv] &= ~(1 << ru)
            
            # Try to add two non-edges
            added1 = None
            added2 = None
            candidates = all_non_edges[:]
            
            for _ in range(40):
                if not candidates:
                    break
                idx = rng.randrange(len(candidates))
                u, v = candidates[idx]
                if (u, v) == removed:
                    continue
                if not _creates_c6_fast(adj_bits, n, u, v):
                    added1 = (u, v)
                    # Add it
                    edges.add(added1)
                    adj[u].add(v)
                    adj[v].add(u)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    candidates.remove(added1)
                    break
            
            if added1 is not None:
                for _ in range(40):
                    if not candidates:
                        break
                    idx = rng.randrange(len(candidates))
                    u, v = candidates[idx]
                    if (u, v) == removed:
                        continue
                    if not _creates_c6_fast(adj_bits, n, u, v):
                        added2 = (u, v)
                        edges.add(added2)
                        adj[u].add(v)
                        adj[v].add(u)
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        candidates.remove(added2)
                        break
            
            if added2 is not None:
                # Success! We removed 1 and added 2, net gain of 1
                # Update edge_list
                edge_list[removed_idx] = added1
                edge_list.append(added2)
                
                # Update non-edges
                for i, e in enumerate(all_non_edges):
                    if e == added1:
                        all_non_edges.pop(i)
                        break
                for i, e in enumerate(all_non_edges):
                    if e == added2:
                        all_non_edges.pop(i)
                        break
                all_non_edges.append(removed)
                
                two_opt_success = True
                no_improve = 0
            else:
                # Failed, undo everything
                if added1 is not None:
                    edges.discard(added1)
                    u1, v1 = added1
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    adj_bits[u1] &= ~(1 << v1)
                    adj_bits[v1] &= ~(1 << u1)
                    edge_list.pop()  # Remove the added1 we temporarily added to edge_list
                    all_non_edges.append(added1)
                
                # Restore the removed edge
                edges.add(removed)
                adj[ru].add(rv)
                adj[rv].add(ru)
                adj_bits[ru] |= (1 << rv)
                adj_bits[rv] |= (1 << ru)
                no_improve += 1
        
        if not two_opt_success and not added:
            no_improve += 1
    
    return best_edges, best_adj, best_adj_bits, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start local search
    num_starts = 8
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 1000
        rng = random.Random(seed_start)
        
        # Greedy build with limited time
        greedy_deadline = min(deadline - 5, start_time + time_per_start * 0.25)
        edges, adj, adj_bits = _greedy_build(n, rng, greedy_deadline)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Local search with remaining time
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1))
        edges, adj, adj_bits, count = _local_search(edges, adj, adj_bits, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
    
    # Final verification: make sure no C6 exists
    final_adj = [set() for _ in range(n)]
    final_adj_bits = [0] * n
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
        final_adj_bits[u] |= (1 << v)
        final_adj_bits[v] |= (1 << u)
    
    # Verify C6-free by checking all edges
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
            final_adj_bits[u] &= ~(1 << v)
            final_adj_bits[v] &= ~(1 << u)
            if _creates_c6_fast(final_adj_bits, n, u, v):
                has_c6 = True
            final_adj[u].add(v)
            final_adj[v].add(u)
            final_adj_bits[u] |= (1 << v)
            final_adj_bits[v] |= (1 << u)
    
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

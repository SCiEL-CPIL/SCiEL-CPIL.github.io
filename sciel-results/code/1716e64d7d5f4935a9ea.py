import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(80):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    # If we found a solution for the current n, we can return it 
                    # unless we want to try higher n (but we are iterating down).
                    # So returning immediately is optimal if we are maximizing n.
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add(adj_sets, n, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    A C4 is u-x-y-v-u where x~u, x~y, y~v.
    For each x in N(u) (x != v), check if N(x) intersects N(v) \ {u}.
    If x~v, then x is in N(v), but we need another common neighbor.
    Actually: C4 exists iff there is x in N(u)\{v} and y in N(v)\{u} with x~y and x!=y.
    Equivalent: for some x in N(u)\{v}, (N(x) ∩ N(v)) contains an element other than possibly u.
    But if x~v, then v is in N(x) and x is in N(v). The C4 would be u-x-v-? -u, but v is already connected to u.
    Let me think again: C4 is a cycle of length 4: u-a-b-v-u where a~u, a~b, b~v, v~u.
    So for each a in N(u)\{v}, we need b in N(v)\{u} such that a~b and a!=b.
    If a~v, then b could be a? No, a!=b. So b must be another common neighbor.
    N(a) ∩ N(v) should contain some element b where b != a and b != u.
    If a~v, then v is in N(a). Is v in N(v)? No. So v is not in the intersection.
    So we just need N(a) ∩ N(v) to be non-empty after removing {a, u}.
    Actually simpler: for each a in N(u)\{v}, if (N(a) & N(v)) - {a, u} is non-empty, C4 exists.
    But we can simplify: if a~v, then a is in N(v). But a is not in N(a) (no self-loops).
    So N(a) & N(v) would include... let's just check if there's any common neighbor.
    The cleanest: for each a in N(u)\{v}, if len(N(a) & N(v)) > 0 and not (a in N(v) and len(N(a)&N(v))==1 and a in N(a)&N(v)):
    This is getting complicated. Let me use the standard check:
    C4 exists iff there exist a in N(u), b in N(v) such that a~b, a!=b, a!=v, b!=u.
    For each a in N(u)\{v}: if N(a) & (N(v)\{u}) is non-empty, C4 exists.
    """
    neighbors_u = adj_sets[u]
    neighbors_v = adj_sets[v]
    
    for a in neighbors_u:
        if a == v:
            continue
        # Check if a has a common neighbor with v (other than u and a itself)
        common = adj_sets[a] & neighbors_v
        # Remove u (trivially common if u~v, but we're adding u~v) and a (a not in N(a))
        if u in common:
            common.discard(u)
        # a is not in adj_sets[a] (no self-loop), so no need to remove a
        if common:
            return True
    return False


def _init_structured(n, rng):
    """
    Initialize with a structured C4-free graph using a polarity-like construction.
    Use a grid or finite field structure to get high minimum degree.
    """
    adj_sets = [set() for _ in range(n)]
    
    # Try a construction based on arranging vertices in a grid-like structure
    # For C4-free, we can use a polarity graph of a projective plane or similar
    
    # Simple approach: use a random regular-ish bipartite core + intra-group edges
    mid = n // 2
    group_a = list(range(mid))
    group_b = list(range(mid, n))
    
    rng.shuffle(group_a)
    rng.shuffle(group_b)
    
    # Connect A to B with controlled density to avoid C4
    # For C4-free bipartite, we need that no two vertices in A share two neighbors in B
    # Use a "permutation matrix" style: each vertex in A connects to a unique pattern
    
    target_deg = max(1, (n - 39) // 2)
    
    # Create a random bipartite graph where each vertex in A has degree ~target_deg
    # and no two vertices in A share more than 1 neighbor in B
    used_pairs = set()
    
    for a in group_a:
        b_candidates = list(group_b)
        rng.shuffle(b_candidates)
        connected = 0
        for b in b_candidates:
            if connected >= target_deg:
                break
            # Check if adding (a,b) would create a C4 in the bipartite part
            # C4 in bipartite: a1-b1-a2-b2-a1 where a1,a2 in A, b1,b2 in B
            # This happens if a1 and a2 share two neighbors in B
            # For now, just add and check later
            adj_sets[a].add(b)
            adj_sets[b].add(a)
            connected += 1
    
    # Add intra-A edges (matching-like to avoid C4)
    a_remaining = list(group_a)
    rng.shuffle(a_remaining)
    for i in range(0, len(a_remaining) - 1, 2):
        u, v = a_remaining[i], a_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    # Add intra-B edges
    b_remaining = list(group_b)
    rng.shuffle(b_remaining)
    for i in range(0, len(b_remaining) - 1, 2):
        u, v = b_remaining[i], b_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    return adj_sets


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Use structured initialization for better starting point
    adj_sets = _init_structured(n, rng)
    
    # Phase 1: Greedy fill - try to add remaining edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if u not in adj_sets[v]:
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
    
    # Phase 2: Local search to improve minimum degree
    max_iters = 3000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate degrees
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find vertices with minimum degree
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Strategy 1: Try to add an edge from v to a non-neighbor
        candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added = True
                break
        
        if added:
            continue
        
        # Strategy 2: Edge swap - remove an edge (a,b) and add (v,u) for some u
        all_current_edges = []
        for u in range(n):
            for w in adj_sets[u]:
                if u < w:
                    all_current_edges.append((u, w))
                        
        if not all_current_edges:
            break
        
        # Try multiple random swaps
        found_swap = False
        for _ in range(5):
            if time.time() > deadline:
                break
            
            a, b = rng.choice(all_current_edges)
            
            # Remove edge (a, b)
            adj_sets[a].remove(b)
            adj_sets[b].remove(a)
            
            # Try to add edges to v
            candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
            rng.shuffle(candidates)
            
            added_edges = []
            for u in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_add(adj_sets, n, u, v):
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    added_edges.append(u)
                    if len(added_edges) >= 2:
                        break
                        
            new_degrees = [len(s) for s in adj_sets]
            new_min = min(new_degrees)
            
            if new_min >= min_deg:
                # Accept the swap
                found_swap = True
                break
            else:
                # Revert
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                for u in added_edges:
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
        
        if not found_swap:
            # Try a 2-swap: remove two edges and add two new edges
            if len(all_current_edges) >= 2:
                a1, b1 = rng.choice(all_current_edges)
                a2, b2 = rng.choice(all_current_edges)
                if (a1, b1) == (a2, b2) or (a1, b1) == (b2, a2):
                    continue
                
                # Remove both edges
                adj_sets[a1].remove(b1)
                adj_sets[b1].remove(a1)
                adj_sets[a2].remove(b2)
                adj_sets[b2].remove(a2)
                
                # Try to add edges to v and another low-degree vertex
                min_verts2 = [i for i, d in enumerate([len(s) for s in adj_sets]) if d <= min_deg + 1]
                v2 = rng.choice(min_verts2) if min_verts2 else rng.choice(range(n))
                
                added_edges = []
                for vtx in [v, v2]:
                    candidates = [u for u in range(n) if u != vtx and u not in adj_sets[vtx]]
                    rng.shuffle(candidates)
                    for u in candidates:
                        if time.time() > deadline:
                            break
                        if not _check_c4_add(adj_sets, n, u, vtx):
                            adj_sets[u].add(vtx)
                            adj_sets[vtx].add(u)
                            added_edges.append((u, vtx))
                            if len(added_edges) >= 4:
                                break
                
                new_degrees = [len(s) for s in adj_sets]
                new_min = min(new_degrees)
                
                if new_min >= min_deg:
                    # Accept
                    pass
                else:
                    # Revert
                    adj_sets[a1].add(b1)
                    adj_sets[b1].add(a1)
                    adj_sets[a2].add(b2)
                    adj_sets[b2].add(a2)
                    for u, vtx in added_edges:
                        adj_sets[u].remove(vtx)
                        adj_sets[vtx].remove(u)
            
            # If no improvement after trying swaps, break to restart
            break

    # Final check
    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END

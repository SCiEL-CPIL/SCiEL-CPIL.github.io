"""
C6-free graph on 40 vertices – multi-start greedy with randomized restarts
and 2-swap local search, using fast bitset-based C6 detection.

Strategy:
  1. Build diverse starting graphs using different block decompositions
     (not just 8 K5 blocks; also try 5-clique + extras, random dense seeds).
  2. Greedy edge addition with multiple random orderings per seed.
  3. 2-swap local search (remove up to 2 edges, add up to 2) to escape
     1-swap local optima.
  4. Keep the best C6-free graph found within the time budget.
"""
import time
import random
import numpy as np

N = 40
FULL_MASK = (1 << N) - 1

def _build_adj(edges):
    """Build bitset adjacency list."""
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_c6_adj(adj):
    """
    Check if graph with bitset adjacency contains a C6.
    A C6 is a cycle of length 6: v0-v1-v2-v3-v4-v5-v0.
    Equivalent: exists edge (u,v) such that there's a path of length 4
    from u to v in the graph with that edge removed.
    
    For efficiency, we check for each non-edge (u,v) whether adding it
    would create a C6, i.e., whether there's a path of length 4 from u to v.
    But we need to check existing edges too. Actually, a C6 exists iff
    there's an edge (u,v) and a path of length 4 from u to v in G - (u,v).
    """
    for u in range(N):
        for v in range(u + 1, N):
            if not (adj[u] >> v) & 1:
                continue
            # Temporarily remove edge
            old_u = adj[u]
            old_v = adj[v]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Check path of length 4 from u to v
            # u - a - b - c - d - v
            # a in N(u), b in N(a)\{u}, c in N(b)\{u,a}, d in N(c)\{u,a,b}, v in N(d)\{u,a,b,c}
            
            found = False
            for a in range(N):
                if not (adj[u] >> a) & 1:
                    continue
                if a == v:
                    continue
                a_bit = 1 << a
                for b in range(N):
                    if b == u or b == a or b == v:
                        continue
                    if not (adj[a] >> b) & 1:
                        continue
                    b_bit = 1 << b
                    for c in range(N):
                        if c == u or c == a or c == b or c == v:
                            continue
                        if not (adj[b] >> c) & 1:
                            continue
                        c_bit = 1 << c
                        for d in range(N):
                            if d == u or d == a or d == b or d == c or d == v:
                                continue
                            if not (adj[c] >> d) & 1:
                                continue
                            if (adj[d] >> v) & 1:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                adj[u] = old_u
                adj[v] = old_v
                return True
            
            adj[u] = old_u
            adj[v] = old_v
    return False

def _would_create_c6(adj, u, v):
    """
    Check if adding edge (u,v) to the current graph (which doesn't have it)
    creates a C6. Equivalent: is there a path of length 4 from u to v in
    the current graph?
    """
    if (adj[u] >> v) & 1:
        return False  # edge already exists
    
    u_bit = 1 << u
    v_bit = 1 << v
    
    # Path of length 4: u - a - b - c - d - v
    # a in N(u), b in N(a)\{u,v}, c in N(b)\{u,a,v}, d in N(c)\{u,a,b,v}
    # and v in N(d)\{u,a,b,c}
    
    for a in range(N):
        if a == u or a == v:
            continue
        if not (adj[u] >> a) & 1:
            continue
        a_bit = 1 << a
        for b in range(N):
            if b == u or b == a or b == v:
                continue
            if not (adj[a] >> b) & 1:
                continue
            b_bit = 1 << b
            for c in range(N):
                if c == u or c == a or c == b or c == v:
                    continue
                if not (adj[b] >> c) & 1:
                    continue
                c_bit = 1 << c
                for d in range(N):
                    if d == u or d == a or d == b or d == c or d == v:
                        continue
                    if not (adj[c] >> d) & 1:
                        continue
                    if (adj[d] >> v) & 1:
                        return True
    return False

def _seed_k5_blocks():
    """8 disjoint K5 blocks: 80 edges."""
    edges = []
    for g in range(8):
        base = g * 5
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append([base + i, base + j])
    return edges

def _seed_random_dense(rng):
    """Random graph with ~90 edges, then remove C6s."""
    edges = []
    edge_set = set()
    # Randomly add edges until we have ~90 or can't add more
    attempts = 0
    while len(edges) < 90 and attempts < 2000:
        attempts += 1
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v or (u, v) in edge_set or (v, u) in edge_set:
            continue
        adj = _build_adj(edges)
        if _would_create_c6(adj, u, v):
            continue
        edges.append([u, v])
        edge_set.add((u, v))
        edge_set.add((v, u))
    return edges

def _greedy_add(edges, rng, deadline, max_extra=None):
    """
    Greedily add edges in random order until no more can be added.
    """
    adj = _build_adj(edges)
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Generate all possible edges not already present
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edge_set:
                candidates.append((u, v))
    
    rng.shuffle(candidates)
    
    added = 0
    for u, v in candidates:
        if time.time() > deadline:
            break
        if max_extra is not None and added >= max_extra:
            break
        if _would_create_c6(adj, u, v):
            continue
        edges.append([u, v])
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        added += 1
    
    return edges

def _local_search_2swap(edges, rng, deadline):
    """
    2-swap local search: remove up to 2 edges, add up to 2.
    Accept if C6-free and edge count doesn't decrease.
    """
    adj = _build_adj(edges)
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    best_count = len(edges)
    best_edges = list(edges)
    no_improve = 0
    max_no_improve = 300
    
    # Precompute candidate edges (all non-edges)
    all_non_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edge_set:
                all_non_edges.append((u, v))
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Decide swap size: 1-swap or 2-swap
        if rng.random() < 0.4 and len(edges) >= 2:
            swap_size = 2
        else:
            swap_size = 1
        
        # Remove random edges
        removed = []
        removed_indices = []
        for _ in range(swap_size):
            if not edges:
                break
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            removed.append((u, v))
            removed_indices.append(idx)
            edges.pop(idx)
            # Fix indices after pop
            for i in range(len(removed_indices)):
                if removed_indices[i] > idx:
                    removed_indices[i] -= 1
        
        # Try to add edges
        added = []
        rng.shuffle(all_non_edges)
        # Rebuild non-edges since we removed some
        for _ in range(swap_size * 3):  # try more candidates
            if time.time() > deadline:
                break
            if len(added) >= swap_size:
                break
            # Pick a random candidate
            u, v = all_non_edges[rng.randint(0, len(all_non_edges) - 1)]
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if _would_create_c6(adj, u, v):
                continue
            edges.append([u, v])
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            edge_set.add((v, u))
            added.append((u, v))
        
        current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edges)
            no_improve = 0
        elif current_count == best_count:
            no_improve += 1
        else:
            # Revert: remove added edges, restore removed edges
            for u, v in added:
                for i in range(len(edges) - 1, -1, -1):
                    if edges[i][0] == u and edges[i][1] == v:
                        edges.pop(i)
                        break
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.discard((u, v))
                edge_set.discard((v, u))
            for u, v in removed:
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
            no_improve += 1
    
    return best_edges

def _verify_c6_free(edges):
    """Final verification that the graph is C6-free."""
    adj = _build_adj(edges)
    return not _has_c6_adj(adj)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = random.Random(12345)
    
    best_edges = None
    best_count = 0
    
    # Multiple strategies for seeds
    seeds = []
    
    # Strategy 1: 8 K5 blocks
    seeds.append(_seed_k5_blocks())
    
    # Strategy 2: Random dense seeds (multiple)
    for _ in range(5):
        seeds.append(_seed_random_dense(rng))
    
    # Strategy 3: K5 blocks + random perturbation
    base = _seed_k5_blocks()
    adj = _build_adj(base)
    edge_set = set()
    for e in base:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    # Remove some random edges from K5 blocks
    perturbed = list(base)
    for _ in range(rng.randint(5, 15)):
        idx = rng.randint(0, len(perturbed) - 1)
        u, v = perturbed[idx]
        perturbed.pop(idx)
        edge_set.discard((u, v))
        edge_set.discard((v, u))
    # Add some random edges
    for _ in range(20):
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v or (u, v) in edge_set or (v, u) in edge_set:
            continue
        if _would_create_c6(adj, u, v):
            continue
        perturbed.append([u, v])
        edge_set.add((u, v))
        edge_set.add((v, u))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    seeds.append(perturbed)
    
    # Run greedy + local search on each seed
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        # Multiple greedy runs per seed with different orderings
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            seed_copy = [e[:] for e in seed]
            result = _greedy_add(seed_copy, rng, deadline - 10)
            
            if len(result) > best_count:
                if _verify_c6_free(result):
                    best_count = len(result)
                    best_edges = list(result)
            
            # Local search refinement
            if time.time() < deadline - 15 and len(result) >= best_count - 3:
                refined = _local_search_2swap(result, rng, deadline - 15)
                if len(refined) > best_count and _verify_c6_free(refined):
                    best_count = len(refined)
                    best_edges = list(refined)
    
    # Final verification
    if best_edges is None:
        # Fallback: return 8 K5 blocks
        return _seed_k5_blocks()
    
    if _verify_c6_free(best_edges):
        return best_edges
    
    # If verification fails, try to fix by removing edges that create C6s
    adj = _build_adj(best_edges)
    # Just return the 8 K5 blocks as fallback
    return _seed_k5_blocks()

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_START = time.time()
_LIMIT = 110.0  # safety margin under 120s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    """Build adjacency as list of ints (bitmasks)."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _greedy_clique_from(start, adj, remaining_mask, n):
    """Greedy: start from `start`, extend to a clique.
    remaining_mask: for each vertex, which other vertices are still connected via remaining edges.
    Actually, let's track remaining edges as a set of (u,v) pairs and build a subgraph."""
    pass

def _find_large_clique(n, adj, edge_set, max_tries=50):
    """Find a large clique using greedy with random restarts.
    adj: bitmask adjacency of original graph
    edge_set: set of (min(u,v), max(u,v)) for remaining edges
    Returns a list of vertices forming a clique, all edges of which are in edge_set.
    """
    best_clique = []
    best_size = 0
    
    # Build a set of remaining edges for quick lookup
    # For greedy clique building, we need: given a set of candidate vertices,
    # check if all pairs are connected in the remaining graph.
    
    # Greedy approach: pick a starting vertex, then greedily add vertices
    # that are connected to all current clique members in the remaining graph.
    
    for _ in range(max_tries):
        if not _deadline_ok():
            break
        # Pick a random vertex that has at least one remaining edge
        active_vertices = set()
        for u, v in edge_set:
            active_vertices.add(u)
            active_vertices.add(v)
        if not active_vertices:
            break
        start = random.choice(list(active_vertices))
        
        # Greedy extend: maintain candidate set
        clique = [start]
        # Candidates: vertices connected to start in remaining graph
        candidates = set()
        for u, v in edge_set:
            if u == start:
                candidates.add(v)
            elif v == start:
                candidates.add(u)
        
        # Greedily add vertices with highest degree within candidates
        while candidates:
            if not _deadline_ok():
                break
            # Pick vertex with max degree within candidates (in remaining graph)
            best_v = -1
            best_deg = -1
            for v in candidates:
                # Count how many candidates are connected to v in remaining graph
                deg = 0
                for w in candidates:
                    if w == v:
                        continue
                    a, b = min(v, w), max(v, w)
                    if (a, b) in edge_set:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0:
                break
            clique.append(best_v)
            # Update candidates: only those connected to best_v in remaining graph
            new_candidates = set()
            for w in candidates:
                if w == best_v:
                    continue
                a, b = min(best_v, w), max(best_v, w)
                if (a, b) in edge_set:
                    new_candidates.add(w)
            candidates = new_candidates
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = list(clique)
    
    return best_clique

def _partition_greedy(n, edges):
    """Partition edges into minimum number of cliques using greedy."""
    if not edges:
        return []
    
    adj = _build_adj(n, edges)
    edge_set = set(edges)
    partition = []
    
    while edge_set:
        if not _deadline_ok():
            # Fallback: each remaining edge as its own clique
            for u, v in edge_set:
                partition.append([u, v])
            break
        
        # Find a large clique in the remaining graph
        clique = _find_large_clique(n, adj, edge_set, max_tries=30)
        
        if not clique or len(clique) < 2:
            # Shouldn't happen if edge_set is non-empty, but safety
            # Pick any remaining edge
            u, v = next(iter(edge_set))
            partition.append([u, v])
            edge_set.discard((min(u,v), max(u,v)))
            continue
        
        # Verify it's a valid clique in remaining graph
        valid = True
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                a, b = min(clique[i], clique[j]), max(clique[i], clique[j])
                if (a, b) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: use just the first two
            u, v = clique[0], clique[1]
            clique = [u, v]
        
        partition.append(clique)
        
        # Remove all edges in this clique
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                a, b = min(clique[i], clique[j]), max(clique[i], clique[j])
                edge_set.discard((a, b))
    
    return partition

def _partition_improved(n, edges, time_budget=15.0):
    """Improved partition with local search / multiple passes."""
    # First do a greedy partition
    partition = _partition_greedy(n, edges)
    
    if not _deadline_ok():
        return partition
    
    # Try to improve: merge cliques if possible (if union is a clique)
    # Or: re-run greedy with different random seeds
    best_partition = partition
    best_len = len(partition)
    
    start_time = time.time()
    while (time.time() - start_time) < time_budget and _deadline_ok():
        # Try a fresh greedy with different random state
        random.seed()  # reseed
        p = _partition_greedy(n, edges)
        if len(p) < best_len:
            best_len = len(p)
            best_partition = p
    
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _random_edges(n, p, seed)
        if not _deadline_ok():
            # Fallback: each edge as its own clique
            results.append([[u, v] for u, v in edges])
            continue
        
        # Allocate time: more for larger graph
        if n == 32:
            time_budget = 20.0
        else:
            time_budget = 30.0
        
        partition = _partition_improved(n, edges, time_budget=time_budget)
        results.append(partition)
    
    return results
# EVOLVE-BLOCK-END

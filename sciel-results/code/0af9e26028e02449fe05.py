"""Clique partition of two fixed dense random graphs via complement matching + greedy refinement."""
# EVOLVE-BLOCK-START
import time
import random as _random
from collections import defaultdict

_T0 = time.time()
_DEADLINE = _T0 + 115.0


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_adjacency(edges, n):
    """Build adjacency list and bitmask from edge list."""
    adj = [[] for _ in range(n)]
    adj_mask = [0] * n
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj, adj_mask


def _max_clique_from_seed(seed_vertices, adj_mask, n, rng):
    """
    Greedily extend a seed set of vertices to a maximal clique.
    seed_vertices: list of vertices that form a clique.
    Returns the maximal clique as a list.
    """
    if not seed_vertices:
        return []
    
    C = list(seed_vertices)
    # Candidates: vertices connected to all in C
    cand_mask = (1 << n) - 1
    for v in C:
        cand_mask &= adj_mask[v]
    # Remove vertices already in C
    for v in C:
        cand_mask &= ~(1 << v)
    
    while cand_mask:
        # Find candidate with max degree within cand_mask
        best_c = -1
        max_deg = -1
        temp = cand_mask
        while temp:
            lsb = temp & -temp
            c = lsb.bit_length() - 1
            d = bin(adj_mask[c] & cand_mask).count('1')
            # Small random noise for diversity
            d += rng.random() * 0.01
            if d > max_deg:
                max_deg = d
                best_c = c
            temp ^= lsb
        
        if best_c == -1:
            break
        
        C.append(best_c)
        cand_mask &= adj_mask[best_c]
        cand_mask &= ~(1 << best_c)
    
    return C


def _partition_via_complement_matching(edges, n, rng):
    """
    Strategy: The complement graph has edges that are NOT in the original graph.
    A clique partition of G corresponds to a partition of V into cliques.
    Equivalently, we want to cover all edges of G with as few cliques as possible.
    
    Alternative view: We want to find a set of cliques whose union of edges is exactly E(G).
    
    Let's try a different structural approach:
    1. Compute the complement graph H = (V, E_bar) where E_bar = all pairs - E(G).
    2. A clique in G is an independent set in H.
    3. So we want to partition the edges of G into independent sets of H? No, that's not right.
       A clique in G is a set of vertices where all pairs are edges in G, i.e., no pair is an edge in H.
       So a clique in G is an independent set in H.
    4. We need to cover all edges of G with cliques (independent sets in H).
       This is equivalent to: for each edge (u,v) in G, it must be in exactly one clique.
    
    Let's try a greedy approach that prioritizes edges in the complement graph.
    Edges in H (non-edges in G) constrain which vertices can be in the same clique.
    
    Actually, let's try a simpler structural change:
    Use a local search / hill climbing on the partition.
    
    New strategy:
    1. Start with a greedy partition (as before).
    2. Apply local moves: move a vertex from one clique to another if it reduces the total number of cliques
       or allows merging of cliques.
    
    But let's try something more radical:
    Model as a graph coloring problem on the line graph? No, too complex.
    
    Let's try: Greedy with a different heuristic - pick the edge with the minimum number of common neighbors
    in the remaining graph (i.e., the "hardest" edge to cover). This might lead to smaller cliques
    but potentially a better overall partition.
    """
    if not edges:
        return []
    
    adj, adj_mask = _build_adjacency(edges, n)
    remaining = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining.add((u, v))
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    cliques = []
    
    while remaining:
        if time.time() > _DEADLINE - 2:
            for u, v in remaining:
                cliques.append([u, v])
            break
        
        # Find the edge with the minimum number of common neighbors (hardest to extend)
        best_u, best_v = -1, -1
        best_common = float('inf')
        
        for (u, v) in remaining:
            common = bin(adj_mask[u] & adj_mask[v]).count('1')
            # Subtract u and v themselves if they are counted
            # adj_mask[u] includes v, adj_mask[v] includes u
            # common counts vertices w such that (u,w) and (v,w) are in G
            # We want common neighbors excluding u and v
            # Actually, adj_mask[u] & adj_mask[v] gives vertices connected to both u and v
            # This includes u? No, adj_mask[u] has bits for neighbors of u, not u itself.
            # So common is the number of vertices w != u, w != v such that (u,w) and (v,w) are edges.
            
            # We want to pick the edge with fewest common neighbors (hardest to extend)
            # But we also want to consider degree for tie-breaking
            score = common
            # Add small random noise
            score += rng.random() * 0.1
            if score < best_common:
                best_common = score
                best_u, best_v = u, v
        
        if best_u == -1:
            break
        
        # Build clique starting with best_u, best_v
        C = _max_clique_from_seed([best_u, best_v], adj_mask, n, rng)
        
        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    if u > w:
                        u, w = w, u
                    remaining.discard((u, w))
                    adj_mask[u] &= ~(1 << w)
                    adj_mask[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
        else:
            # Fallback: take just the edge
            cliques.append([best_u, best_v])
            remaining.discard((best_u, best_v))
            adj_mask[best_u] &= ~(1 << best_v)
            adj_mask[best_v] &= ~(1 << best_u)
            deg[best_u] -= 1
            deg[best_v] -= 1
    
    return cliques


def _partition_via_max_degree(edges, n, rng):
    """
    Original strategy: pick edge with max sum of degrees.
    """
    if not edges:
        return []
    
    adj, adj_mask = _build_adjacency(edges, n)
    remaining = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining.add((u, v))
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    cliques = []
    
    while remaining:
        if time.time() > _DEADLINE - 2:
            for u, v in remaining:
                cliques.append([u, v])
            break
        
        best_u, best_v = -1, -1
        best_score = -1.0
        
        for (u, v) in remaining:
            score = deg[u] + deg[v] + rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_u, best_v = u, v
        
        if best_u == -1:
            break
        
        C = _max_clique_from_seed([best_u, best_v], adj_mask, n, rng)
        
        if len(C) >= 2:
            cliques.append(C)
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    if u > w:
                        u, w = w, u
                    remaining.discard((u, w))
                    adj_mask[u] &= ~(1 << w)
                    adj_mask[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
        else:
            cliques.append([best_u, best_v])
            remaining.discard((best_u, best_v))
            adj_mask[best_u] &= ~(1 << best_v)
            adj_mask[best_v] &= ~(1 << best_u)
            deg[best_u] -= 1
            deg[best_v] -= 1
    
    return cliques


def _validate_partition(cliques, edges):
    """Validate that cliques form a valid partition of edges."""
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                u, v = c[i], c[j]
                if u > v:
                    u, v = v, u
                e = (u, v)
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return len(seen_edges) == len(edges)


def _solve_instance(edges, n):
    """
    Solve a single instance with multiple strategies and restarts.
    """
    if not edges:
        return []
    
    best_cliques = None
    best_count = float('inf')
    
    # Try different strategies
    strategies = [
        ("min_common", _partition_via_complement_matching),
        ("max_degree", _partition_via_max_degree),
    ]
    
    restart_count = 0
    while time.time() < _DEADLINE - 5:
        restart_count += 1
        # Alternate between strategies
        strategy_name, strategy_fn = strategies[restart_count % len(strategies)]
        rng = _random.Random(20000 + restart_count * 7)
        
        cliques = strategy_fn(edges, n, rng)
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques
        
        if best_count < 12:
            if restart_count > 200:
                break
    
    if best_cliques is None:
        rng = _random.Random(0)
        best_cliques = _partition_via_max_degree(edges, n, rng)
    
    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)
    
    # Solve smaller instance first
    res1 = _solve_instance(edges1, 32)
    
    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64)
    
    return [res1, res2]
# EVOLVE-BLOCK-END

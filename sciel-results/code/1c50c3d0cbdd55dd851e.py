"""Clique partition of two fixed dense random graphs via optimized bitmask greedy with triangle-first heuristic."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 112.0  # Wall-clock guard (evaluator kills at 150s, keep 38s margin)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _encode_edge(u, v):
    if u > v:
        u, v = v, u
    return u * 128 + v  # n <= 64, so 128 is safe


def _decode_edge(e):
    u = e >> 7
    v = e & 127
    return u, v


def _popcount(x):
    """Fast popcount using bit operations."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def _greedy_partition_triangle_first(edges, n, rng):
    """
    Greedy clique partition that prioritizes triangle extraction.
    Strategy:
    1. Find triangles (3-cliques) and extract them first, as they cover 3 edges per clique.
    2. For remaining edges, use the standard greedy approach with common-neighbor heuristic.
    3. Use bitmask operations for speed.
    """
    if not edges:
        return []

    # Build adjacency as bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add(_encode_edge(u, v))

    remaining = set(edge_set)
    cliques = []

    # Track degrees in the remaining graph
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    def _extract_triangle():
        """Find and extract a triangle from remaining edges using edge-neighbor lists."""
        # Build list of edges for each vertex (only those still remaining)
        # We'll scan a sample of remaining edges and check for common neighbors
        if len(remaining) > 100:
            sample = rng.sample(list(remaining), 100)
        else:
            sample = list(remaining)
            
        for e in sample:
            u, v = _decode_edge(e)
            # Common neighbors of u and v in the current graph
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                # Check if all three edges exist in remaining
                e_uw = _encode_edge(u, w)
                e_vw = _encode_edge(v, w)
                if e_uw in remaining and e_vw in remaining:
                    # Found triangle (u, v, w)
                    # Remove all three edges
                    remaining.discard(e)
                    remaining.discard(e_uw)
                    remaining.discard(e_vw)
                    
                    # Update adjacency
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
                    
                    # Update degrees
                    deg[u] -= 2
                    deg[v] -= 2
                    deg[w] -= 2
                    
                    return [u, v, w]
                common ^= lsb
        return None

    def _extract_max_clique(start_u, start_v):
        """Extract a maximal clique starting from edge (start_u, start_v)."""
        C = [start_u, start_v]
        cand_mask = adj[start_u] & adj[start_v]
        cand_mask &= ~(1 << start_u)
        cand_mask &= ~(1 << start_v)
        
        while cand_mask:
            # Find candidate with max degree within cand_mask
            best_c = -1
            max_deg_c = -1
            temp_mask = cand_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                d = _popcount(adj[c] & cand_mask)
                if d > max_deg_c:
                    max_deg_c = d
                    best_c = c
                temp_mask ^= lsb

            if best_c == -1:
                break

            C.append(best_c)
            cand_mask &= adj[best_c]
            for w in C:
                cand_mask &= ~(1 << w)

        return C

    def _remove_clique_edges(C):
        """Remove all edges within clique C from remaining and update state."""
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                e = _encode_edge(a, b)
                if e in remaining:
                    remaining.discard(e)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    deg[a] -= 1
                    deg[b] -= 1

    # Phase 1: Extract triangles while they exist
    triangle_count = 0
    consecutive_fails = 0
    while remaining and triangle_count < 800:
        if time.time() > _DEADLINE - 5:
            break
        tri = _extract_triangle()
        if tri is None:
            consecutive_fails += 1
            if consecutive_fails >= 5:
                break
            continue
        consecutive_fails = 0
        cliques.append(tri)
        triangle_count += 1

    # Phase 2: Extract remaining edges using greedy max-clique
    while remaining:
        if time.time() > _DEADLINE - 3:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                u, v = _decode_edge(e)
                cliques.append([u, v])
            break

        # Find a good edge to start from: pick an edge with fewest common neighbors
        if len(remaining) > 20:
            sample = rng.sample(list(remaining), 20)
        else:
            sample = list(remaining)

        best_edge = None
        best_common = float('inf')
        for e in sample:
            u, v = _decode_edge(e)
            common = _popcount(adj[u] & adj[v])
            if common < best_common:
                best_common = common
                best_edge = e

        u, v = _decode_edge(best_edge)
        
        # Build clique starting with u and v
        C = _extract_max_clique(u, v)
        
        if len(C) >= 2:
            cliques.append(C)
            _remove_clique_edges(C)
        else:
            # Fallback: take one edge
            e = next(iter(remaining))
            a, b = _decode_edge(e)
            cliques.append([a, b])
            remaining.discard(e)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            deg[a] -= 1
            deg[b] -= 1

    return cliques


def _greedy_partition_standard(edges, n, rng):
    """
    Standard greedy clique partition using bitmasks.
    Strategy:
    1. Pick an edge with fewest common neighbors (hardest to cover) from a sample.
    2. Extend to a maximal clique using a max-degree heuristic within common neighbors.
    3. Remove all edges in the clique.
    """
    if not edges:
        return []

    # Build adjacency as bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add(_encode_edge(u, v))

    remaining = set(edge_set)
    cliques = []

    while remaining:
        if time.time() > _DEADLINE - 2:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                u, v = _decode_edge(e)
                cliques.append([u, v])
            break

        # Find a good edge to start from: pick an edge with fewest common neighbors
        if len(remaining) > 20:
            sample = rng.sample(list(remaining), 20)
        else:
            sample = list(remaining)

        best_edge = None
        best_common = float('inf')
        for e in sample:
            u, v = _decode_edge(e)
            common = _popcount(adj[u] & adj[v])
            if common < best_common:
                best_common = common
                best_edge = e

        u, v = _decode_edge(best_edge)
        
        # Build clique starting with u and v
        C = [u, v]
        cand_mask = adj[u] & adj[v]
        cand_mask &= ~(1 << u)
        cand_mask &= ~(1 << v)
        
        while cand_mask:
            # Find candidate with max degree within cand_mask
            best_c = -1
            max_deg_c = -1
            temp_mask = cand_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                d = _popcount(adj[c] & cand_mask)
                if d > max_deg_c:
                    max_deg_c = d
                    best_c = c
                temp_mask ^= lsb

            if best_c == -1:
                break

            C.append(best_c)
            cand_mask &= adj[best_c]
            for w in C:
                cand_mask &= ~(1 << w)

        if len(C) >= 2:
            cliques.append(C)
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = _encode_edge(a, b)
                    if e in remaining:
                        remaining.discard(e)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
        else:
            e = next(iter(remaining))
            a, b = _decode_edge(e)
            cliques.append([a, b])
            remaining.discard(e)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)

    return cliques


def _local_improve(cliques, edges, n, rng):
    """
    Local improvement: try to merge cliques or reassign edges to reduce count.
    """
    if len(cliques) < 2:
        return cliques

    # Build edge-to-clique mapping
    edge_to_clique = {}
    for ci, c in enumerate(cliques):
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                e = _encode_edge(c[i], c[j])
                edge_to_clique[e] = ci

    # Build adjacency for quick lookup
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    new_cliques = [list(c) for c in cliques]
    
    def _is_clique(vertices):
        """Check if a set of vertices forms a clique."""
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                if not (adj[vlist[i]] & (1 << vlist[j])):
                    return False
        return True

    improved = True
    while improved and time.time() < _DEADLINE - 10:
        improved = False
        
        # Strategy 1: Try merging pairs of cliques
        for i in range(len(new_cliques)):
            if time.time() > _DEADLINE - 10:
                break
            for j in range(i + 1, len(new_cliques)):
                if len(new_cliques[i]) + len(new_cliques[j]) > 14:
                    continue
                
                union_set = set(new_cliques[i]) | set(new_cliques[j])
                if _is_clique(union_set):
                    new_cliques[i] = list(union_set)
                    new_cliques.pop(j)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy 2: Try to add a vertex from one clique to another
        for i in range(len(new_cliques)):
            if time.time() > _DEADLINE - 10:
                break
            for j in range(len(new_cliques)):
                if i == j:
                    continue
                # Try adding each vertex from clique j to clique i
                for v in list(new_cliques[j]):
                    if v in new_cliques[i]:
                        continue
                    new_set = set(new_cliques[i]) | {v}
                    if _is_clique(new_set):
                        # Check that all edges from v to new_cliques[i] exist in edge_to_clique
                        # and belong to clique j
                        can_move = True
                        for u in new_cliques[i]:
                            e = _encode_edge(u, v)
                            if edge_to_clique.get(e) != j:
                                can_move = False
                                break
                        if can_move:
                            new_cliques[i] = list(new_set)
                            new_cliques[j].remove(v)
                            if len(new_cliques[j]) < 2:
                                new_cliques.pop(j)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return new_cliques


def _validate_partition(cliques, edges):
    """Validate that cliques form a valid partition of edges."""
    edge_set = set(_encode_edge(u, v) for u, v in edges)
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                e = _encode_edge(c[i], c[j])
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return len(seen_edges) == len(edges)


def _solve_instance(edges, n, time_budget):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    best_cliques = None
    best_count = float('inf')
    
    restart_count = 0
    start_time = time.time()
    
    while time.time() < _DEADLINE - 5:
        restart_count += 1
        rng = _random.Random(10000 + restart_count)
        
        # Alternate between triangle-first and standard greedy
        if restart_count % 2 == 0:
            cliques = _greedy_partition_triangle_first(edges, n, rng)
        else:
            cliques = _greedy_partition_standard(edges, n, rng)
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

        # Early stop if we reach a very good solution
        if best_count < 12:
            if restart_count > 50:
                break

    if best_cliques is None:
        rng = _random.Random(0)
        best_cliques = _greedy_partition_standard(edges, n, rng)
    
    # Try local improvement
    rng2 = _random.Random(42)
    improved = _local_improve(best_cliques, edges, n, rng2)
    if len(improved) < len(best_cliques):
        best_cliques = improved

    # Second pass of local improvement with different seed
    rng3 = _random.Random(123)
    improved2 = _local_improve(best_cliques, edges, n, rng3)
    if len(improved2) < len(best_cliques):
        best_cliques = improved2

    # Final validation
    if not _validate_partition(best_cliques, edges):
        rng4 = _random.Random(999)
        best_cliques = _greedy_partition_standard(edges, n, rng4)
    
    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first with 35% of budget
    res1 = _solve_instance(edges1, 32, 40.0)

    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64, 70.0)

    return [res1, res2]
# EVOLVE-BLOCK-END

"""Rewritten: structured clique partition for split graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Separate clique-internal edges from cross edges
        clique_edges = []
        cross_edges = []
        for (u, v) in edges:
            if u < k and v < k:
                clique_edges.append((u, v))
            else:
                cross_edges.append((u, v))

        # Partition clique vertices into groups to form cliques
        # Each group of size g covers g*(g-1)/2 internal edges
        # Cross edges are assigned to the group containing their clique vertex
        # Total cliques = sum over groups of (1 if group has any edge)
        # We want to minimize: number of non-empty groups (internal) + number of groups with cross edges

        # Strategy: partition [0, k) into contiguous groups of size g
        # Try several g values and pick the best
        best_cliques = None
        best_count = float('inf')

        # g should be at least 2 for a valid clique
        for g in range(2, k + 1):
            if time.time() - _t0 > _DEADLINE:
                break
            # Partition into groups
            groups = []
            for start in range(0, k, g):
                end = min(start + g, k)
                if end - start >= 2:
                    groups.append(list(range(start, end)))
                elif end - start == 1:
                    # Single vertex - merge with previous group
                    if groups:
                        groups[-1].append(start)
                    # else can't form a clique from 1 vertex, but it has no internal edges
                    # Its cross edges still need to be covered - we'll handle separately

            # For each group, check if it has any internal or cross edges
            # Build vertex -> group index mapping
            vert_to_group = {}
            for gi, grp in enumerate(groups):
                for v in grp:
                    vert_to_group[v] = gi

            # Count which groups have edges
            group_has_edge = [False] * len(groups)
            for (u, v) in clique_edges:
                if u in vert_to_group and v in vert_to_group:
                    group_has_edge[vert_to_group[u]] = True
            for (u, v) in cross_edges:
                # u < k, v >= k (or vice versa, but in our construction u < k)
                cu = u if u < k else v
                if cu in vert_to_group:
                    group_has_edge[vert_to_group[cu]] = True

            # Groups with no edges don't need to be listed
            # But every edge must be in exactly one clique
            # Internal edges: both endpoints in same group -> that group's clique
            # Cross edges: one endpoint in a group -> that group's clique
            # This works! Each edge is assigned to exactly one group.

            # But wait: we need each listed set to be a CLIQUE.
            # A group is a clique only if all its vertices are mutually adjacent.
            # In a split graph, clique vertices are all mutually adjacent, so any subset of clique vertices is a clique.
            # Cross edges: the group contains only clique vertices, so it's a valid clique.
            # But the cross edge (u, v) where v >= k: v is NOT in the group.
            # So the group is not a clique containing that edge!

            # I need to reconsider. A clique partition means each part is a clique in the original graph.
            # So if I put vertex v (independent set) into a group, the group must be a clique,
            # meaning all pairs in the group must be edges. But independent set vertices have no edges among themselves,
            # and may not connect to all clique vertices.

            # Better approach: each clique in the partition can contain:
            # - Some clique vertices (all mutually adjacent)
            # - Some independent vertices that are connected to ALL clique vertices in the clique
            # - At most one independent vertex if we don't want to check mutual independence (they're independent, so no edges between them, but that's fine - they just don't need to be adjacent to each other for the set to be... wait, a clique requires ALL pairs to be edges)

            # Actually, a clique requires every pair of vertices in the set to be an edge.
            # Independent set vertices have no edges between them, so a clique can contain at most one independent vertex.
            # And that independent vertex must be connected to all clique vertices in the set.

            # So each clique is: a subset of clique vertices + optionally one independent vertex connected to all of them.

            # Revised strategy:
            # 1. Partition clique vertices into groups
            # 2. For each cross edge (u, v) where u is in clique, v in independent set:
            #    Assign v to the group containing u, but only if v connects to ALL vertices in that group.
            #    If not, v must go in its own clique with just u (a 2-clique).

            # This is getting complex. Let me use a simpler approach:
            # - Make each cross edge a separate 2-clique {u, v}
            # - Partition clique edges into large cliques
            # This is valid but may not be optimal. Let's start with this and improve.

            pass  # Skip this complex approach for now

        # Simple approach: partition clique vertices into groups, each group is a clique.
        # Cross edges become individual 2-cliques.
        # Try different group sizes.
        for g in range(2, k + 1):
            if time.time() - _t0 > _DEADLINE:
                break
            groups = []
            for start in range(0, k, g):
                end = min(start + g, k)
                if end - start >= 2:
                    groups.append(tuple(range(start, end)))
                elif end - start == 1:
                    if groups:
                        groups[-1] = groups[-1] + (start,)
                    else:
                        groups.append((start,))

            # Only keep groups with size >= 2 that have at least one internal edge
            # Actually, every group of size >= 2 from clique vertices has internal edges
            cliques = []
            for grp in groups:
                if len(grp) >= 2:
                    cliques.append(list(grp))

            # Add cross edges as 2-cliques
            for (u, v) in cross_edges:
                cliques.append([u, v])

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]

        if best_cliques is None:
            # Fallback: each edge as a clique
            best_cliques = [[u, v] for (u, v) in edges]

        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - _t0 > _DEADLINE:
            # Fallback for remaining
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for (u, v) in edges])
        else:
            results.append(solve_one(*spec))

    return results
# EVOLVE-BLOCK-END

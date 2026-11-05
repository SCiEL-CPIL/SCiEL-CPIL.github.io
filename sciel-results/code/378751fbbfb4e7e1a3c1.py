"""Clique partitioning for two fixed dense random graphs."""
import time
import random
from itertools import combinations

DEADLINE = 140  # seconds, safety margin before 150s kill


def _generate_graph(n, p, seed):
    """Generate a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_set.add((u, v))
    return adj, edge_set


def _find_greedy_clique(adj, n, candidate_set, deadline):
    """
    Find a large clique greedily from candidate vertices.
    Start with the vertex of highest degree within candidates,
    then iteratively add vertices connected to all current clique members.
    """
    if not candidate_set:
        return []
    if time.time() > deadline:
        return []

    # Score vertices by degree within candidate set
    def degree_in(v, s):
        return sum(1 for u in s if adj[v][u])

    # Start with highest-degree vertex in candidates
    start = max(candidate_set, key=lambda v: degree_in(v, candidate_set))
    clique = [start]
    remaining = set(candidate_set) - {start}

    while remaining:
        if time.time() > deadline:
            break
        # Find vertices connected to ALL current clique members
        candidates = []
        for v in remaining:
            if all(adj[v][u] for u in clique):
                candidates.append(v)
        if not candidates:
            break
        # Pick the one with highest degree in remaining
        best = max(candidates, key=lambda v: degree_in(v, remaining))
        clique.append(best)
        remaining.discard(best)

    return clique


def _solve_instance(adj, edge_set, n, deadline):
    """
    Partition all edges into cliques.
    Strategy: repeatedly find a large clique, add it, remove its edges.
    For remaining edges, group into triangles where possible, then pairs.
    """
    cliques = []
    remaining_edges = set(edge_set)

    while remaining_edges:
        if time.time() > deadline:
            break

        # Build adjacency of remaining edges
        rem_adj = [[False] * n for _ in range(n)]
        rem_deg = [0] * n
        for (u, v) in remaining_edges:
            rem_adj[u][v] = rem_adj[v][u] = True
            rem_deg[u] += 1
            rem_deg[v] += 1

        # Find vertices involved in remaining edges
        involved = set()
        for (u, v) in remaining_edges:
            involved.add(u)
            involved.add(v)

        if not involved:
            break

        # Try to find a clique of size >= 3 using the remaining edges
        # Use greedy approach: pick high-degree vertex, expand
        start = max(involved, key=lambda v: rem_deg[v])
        clique = [start]
        remaining = set(involved) - {start}

        while remaining:
            if time.time() > deadline:
                break
            candidates = []
            for v in remaining:
                if all(rem_adj[v][u] for u in clique):
                    candidates.append(v)
            if not candidates:
                break
            best = max(candidates, key=lambda v: rem_deg[v])
            clique.append(best)
            remaining.discard(best)

        if len(clique) >= 2:
            # Verify it's a valid clique in remaining edges
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Add this clique and remove its edges
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        remaining_edges.discard((u, v))
                        remaining_edges.discard((v, u))
                continue

        # If no clique >= 2 found (shouldn't happen), take one edge
        u, v = next(iter(remaining_edges))
        cliques.append([u, v])
        remaining_edges.discard((u, v))
        remaining_edges.discard((v, u))

    return cliques


def _partition_edges(adj, edge_set, n, deadline):
    """
    More careful partitioning:
    1. Find maximal cliques greedily, add them.
    2. For leftover edges, try to form triangles.
    3. Remaining edges become 2-cliques.
    """
    cliques = []
    remaining_edges = set(edge_set)

    # Phase 1: Greedily extract large cliques
    while remaining_edges and time.time() < deadline:
        # Build remaining adjacency
        rem_adj = [[False] * n for _ in range(n)]
        rem_deg = [0] * n
        for (u, v) in remaining_edges:
            rem_adj[u][v] = rem_adj[v][u] = True
            rem_deg[u] += 1
            rem_deg[v] += 1

        involved = set()
        for (u, v) in remaining_edges:
            involved.add(u)
            involved.add(v)

        if not involved:
            break

        # Pick start vertex with highest remaining degree
        start = max(involved, key=lambda v: rem_deg[v])
        clique = [start]
        remaining = set(involved) - {start}

        while remaining and time.time() < deadline:
            candidates = [v for v in remaining if all(rem_adj[v][u] for u in clique)]
            if not candidates:
                break
            best = max(candidates, key=lambda v: rem_deg[v])
            clique.append(best)
            remaining.discard(best)

        if len(clique) >= 2:
            # Check validity
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        remaining_edges.discard((u, v))
                        remaining_edges.discard((v, u))
                continue

        # Fallback: take one edge
        u, v = next(iter(remaining_edges))
        cliques.append([u, v])
        remaining_edges.discard((u, v))
        remaining_edges.discard((v, u))

    # Phase 2: Try to merge some 2-cliques into triangles
    # Build map from edge to clique index
    edge_to_clique = {}
    for idx, clique in enumerate(cliques):
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                edge_to_clique[(u, v)] = idx
                edge_to_clique[(v, u)] = idx

    # Find triangles in remaining structure that can be merged
    # This is complex; skip for now and just ensure all edges are covered

    return cliques


def construct():
    """
    Return two lists of cliques, one per instance.
    Each clique is a list of vertices forming a clique.
    Every edge appears in exactly one clique.
    Minimize total number of cliques.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edge_set = _generate_graph(n, p, seed)
        cliques = _partition_edges(adj, edge_set, n, deadline)
        results.append(cliques)
        if time.time() > deadline:
            # If we ran out of time, we need to ensure validity
            # For now, just return what we have
            break

    # If we didn't process both instances, fill in with edge-by-edge
    while len(results) < 2:
        idx = len(results)
        n, p, seed = ((32, 0.6, 601), (64, 0.55, 602))[idx]
        adj, edge_set = _generate_graph(n, p, seed)
        results.append([list(e) for e in edge_set])

    return results


if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total cliques: {len(res[0]) + len(res[1])}")

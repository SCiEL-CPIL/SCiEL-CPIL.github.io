"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n = 40

    # Start with 8 disjoint K5's (104 edges)
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])

    edge_set = set(tuple(sorted(e)) for e in edges)

    def build_adjacency(edge_set):
        adj = [set() for _ in range(n)]
        for (a, b) in edge_set:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    adj = build_adjacency(edge_set)

    def has_c6(adj, new_edge):
        """Check if adding new_edge creates a C6 in the graph."""
        u, v = new_edge
        # A C6 containing edge (u,v) would be u-v-w-x-y-z-u
        # This means there must be a path of length 4 from u to v not using edge (u,v)
        # Or equivalently, the graph with edge (u,v) removed must have a path of length 4 between u and v

        # We need to check if there exist vertices w, x, y, z such that:
        # u-w, w-x, x-y, y-z, z-v are all edges, and {u,v,w,x,y,z} are distinct
        # (with w != v, z != u, etc.)

        # Path: u - w - x - y - z - v
        # Edges needed: (u,w), (w,x), (x,y), (y,z), (z,v)
        # All vertices distinct: u, v, w, x, y, z all different

        # For each possible w in adj[u]:
        for w in adj[u]:
            if w == v:
                continue
            # For each possible x in adj[w]:
            for x in adj[w]:
                if x in (u, v, w):
                    continue
                # For each possible y in adj[x]:
                for y in adj[x]:
                    if y in (u, v, w, x):
                        continue
                    # For each possible z in adj[y]:
                    for z in adj[y]:
                        if z in (u, v, w, x, y):
                            continue
                        # Check if z is adjacent to v
                        if v in adj[z]:
                            return True
        return False

    # Try to add edges greedily with multiple random orders
    all_possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    random.shuffle(all_possible)

    best_edges = list(edge_set)
    best_count = len(edge_set)

    # Try multiple random orderings to escape local optima
    for trial in range(5):
        if time.time() > deadline:
            break

        # Reset to current best
        current_edge_set = set(best_edges)
        current_adj = build_adjacency(current_edge_set)

        # Shuffle the order of edges to try
        candidate_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
        random.shuffle(candidate_edges)

        for (u, v) in candidate_edges:
            if time.time() > deadline:
                break
            if (u, v) in current_edge_set:
                continue
            # Check if adding this edge creates a C6
            if not has_c6(current_adj, (u, v)):
                current_edge_set.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)

        # Update best if we found a better solution
        if len(current_edge_set) > best_count:
            best_edges = list(current_edge_set)
            best_count = len(current_edge_set)

    # Convert back to list of [u, v]
    return [[a, b] for (a, b) in sorted(best_edges)]
# EVOLVE-BLOCK-END

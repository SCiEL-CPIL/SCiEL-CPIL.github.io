import time
import random

def construct():
    N = 15
    deadline = time.time() + 50  # safety margin before 60s limit

    def is_valid(edges):
        """Check if no 3-edge path has endpoints of equal degree."""
        deg = [0] * N
        adj = [set() for _ in range(N)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        # For every edge (u, v), check all s in N(u)\{v} and t in N(v)\{u}
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def violation_count(edges):
        """Count number of violating 3-edge paths."""
        deg = [0] * N
        adj = [set() for _ in range(N)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        count = 0
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
        return count

    def try_complete_greedy(seed_edges, edge_order=None):
        """Try to greedily add edges to seed_edges while maintaining validity."""
        if not is_valid(seed_edges):
            return seed_edges

        edges = list(seed_edges)
        edge_set = set(tuple(sorted(e)) for e in edges)

        if edge_order is None:
            edge_order = [(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) not in edge_set]
            random.shuffle(edge_order)

        for u, v in edge_order:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            new_edges = edges + [[u, v]]
            if is_valid(new_edges):
                edges = new_edges
                edge_set.add((min(u, v), max(u, v)))
        return edges

    best_edges = None
    best_len = 0

    # Start with empty graph and build up
    # Try multiple random orderings
    all_possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random ordering of edges
        order = all_possible_edges[:]
        random.shuffle(order)

        edges = []
        edge_set = set()
        for u, v in order:
            if (u, v) in edge_set:
                continue
            new_edges = edges + [[u, v]]
            if is_valid(new_edges):
                edges = new_edges
                edge_set.add((u, v))

        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]

        # Also try starting from some initial structures
        if num_restarts % 5 == 0:
            # Try star-like structures
            for center in range(N):
                star = [[center, v] for v in range(N) if v != center]
                completed = try_complete_greedy(star)
                if len(completed) > best_len:
                    best_len = len(completed)
                    best_edges = completed[:]

    # Convert to required format
    if best_edges is None:
        best_edges = [[0, v] for v in range(1, N)]

    return best_edges

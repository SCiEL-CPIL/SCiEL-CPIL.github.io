import random
import time
import sys

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    max_time = 110.0  # Safety margin

    # Precompute all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_possible_edges)}
    num_edges_total = len(all_possible_edges)

    # Represent graph as a set of edge indices for fast lookup
    def random_graph():
        all_edges = list(range(num_edges_total))
        random.shuffle(all_edges)
        return set(all_edges[:target_edges])

    def get_score(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for ei in edge_set:
            u, v = all_possible_edges[ei]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        max_sum = 0
        # Find all triangles
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def get_worst_triangle(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for ei in edge_set:
            u, v = all_possible_edges[ei]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        worst_tri = None
        max_s = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_s:
                                max_s = s
                                worst_tri = (u, v, w)
        return worst_tri, max_s

    best_score = 100
    best_edges = set()

    # Multiple restarts
    num_restarts = 50
    per_restart_time = max_time / num_restarts

    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break

        # Initialize
        if r == 0:
            # Start with a structured graph: K_{8,8} + 1 edge
            edges = set()
            # K_{8,8} between 0-7 and 8-15
            for i in range(8):
                for j in range(8, 16):
                    edges.add(edge_index[(i, j)])
            # Add one more edge to get 65 edges
            # Add an edge within one partition, e.g., (0, 1)
            edges.add(edge_index[(0, 1)])
        else:
            edges = random_graph()

        current_score = get_score(edges)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)

        if best_score <= 23:
            break

        restart_start = time.time()
        no_improve_count = 0

        while time.time() - restart_start < per_restart_time:
            if time.time() - start_time > max_time:
                break

            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if best_score <= 23:
                break

            worst_tri, _ = get_worst_triangle(edges)
            if not worst_tri:
                break

            u, v, w = worst_tri

            # Collect edges to remove (incident to u, v, or w)
            remove_candidates = set()
            for vert in (u, v, w):
                for neighbor in range(n):
                    if neighbor != vert and (min(vert, neighbor), max(vert, neighbor)) in edge_index:
                        ei = edge_index[(min(vert, neighbor), max(vert, neighbor))]
                        if ei in edges:
                            remove_candidates.add(ei)

            # Collect non-edges to add
            non_edges = []
            for ei in range(num_edges_total):
                if ei not in edges:
                    non_edges.append(ei)

            random.shuffle(non_edges)
            non_edges_to_check = non_edges[:30]

            improved = False
            for rem_ei in remove_candidates:
                for add_ei in non_edges_to_check:
                    if rem_ei == add_ei:
                        continue

                    # Apply swap
                    edges.remove(rem_ei)
                    edges.add(add_ei)

                    new_score = get_score(edges)

                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Undo swap
                        edges.remove(add_ei)
                        edges.add(rem_ei)
                if improved:
                    break

            if not improved:
                # Try a random swap to escape local minimum
                if non_edges:
                    rem_ei = random.choice(list(edges))
                    add_ei = random.choice(non_edges)
                    if rem_ei != add_ei:
                        edges.remove(rem_ei)
                        edges.add(add_ei)
                        new_score = get_score(edges)
                        if new_score <= best_score + 2:
                            current_score = new_score
                        else:
                            # Undo
                            edges.remove(add_ei)
                            edges.add(rem_ei)

    return {'n': n, 'edges': [list(all_possible_edges[ei]) for ei in best_edges]}

import time
import sys
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s evaluator limit

    # Start with K_{7,8}: A = 0..6 (7 vertices), B = 7..14 (8 vertices)
    base_edges = set()
    for u in range(7):
        for v in range(7, 15):
            base_edges.add((u, v))

    # All possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    missing_edges = sorted(all_edges - base_edges)

    # Efficient validity checking using adjacency sets and degree array
    def build_state(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_edge_add(adj, deg, u, v):
        """Check if adding edge (u,v) keeps the graph valid.
        Only paths using the new edge (u,v) can be violated.
        Three cases:
        A. (u,v) is middle edge: s-u-v-t, s in N(u)\{v}, t in N(v)\{u}, s!=t
           Need deg[s] != deg[t]
        B. (u,v) is first edge: u-v-x-y, x in N(v)\{u}, y in N(x), y!=u
           Need (deg[u]+1) != deg[y]
        C. (u,v) is last edge: x-y-u-v, x in N(u)\{v}, y in N(x), y!=v
           Need deg[x] != (deg[v]+1)
        """
        nu = adj[u] - {v}
        nv = adj[v] - {u}

        # Check A: middle edge
        for s in nu:
            for t in nv:
                if s != t and deg[s] == deg[t]:
                    return False

        # Check B: first edge, endpoints u and y
        new_deg_u = deg[u] + 1
        for x in nv:
            for y in adj[x]:
                if y != u and new_deg_u == deg[y]:
                    return False

        # Check C: last edge, endpoints x and v
        new_deg_v = deg[v] + 1
        for x in nu:
            for y in adj[x]:
                if y != v and deg[x] == new_deg_v:
                    return False

        return True

    def add_edge(state, u, v):
        adj, deg = state
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def local_improve(edge_set, max_time=None):
        """Greedily add any missing edge that maintains validity."""
        adj, deg = build_state(edge_set)
        current_set = set(edge_set)
        improved = True
        while improved:
            improved = False
            if max_time and time.time() > max_time:
                break
            for e in missing_edges:
                if e in current_set:
                    continue
                if max_time and time.time() > max_time:
                    break
                u, v = e
                if check_edge_add(adj, deg, u, v):
                    add_edge((adj, deg), u, v)
                    current_set.add(e)
                    improved = True
        return current_set

    def greedy_with_ordering(order, max_time=None):
        """Greedily add edges in given order."""
        edge_set = set(base_edges)
        adj, deg = build_state(edge_set)
        for e in order:
            if e in edge_set:
                continue
            if max_time and time.time() > max_time:
                break
            u, v = e
            if check_edge_add(adj, deg, u, v):
                add_edge((adj, deg), u, v)
                edge_set.add(e)
        return edge_set

    # Edge categorization for priority
    def edge_category(e):
        u, v = e
        if u < 7 and v < 7:
            return 0  # Intra-A
        elif u >= 7 and v >= 7:
            return 1  # Intra-B
        else:
            return 2  # Cross (shouldn't be missing)

    best_edges = set(base_edges)
    best_count = len(base_edges)

    rng = random.Random(42)
    num_restarts = 30

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        strategy = restart % 5
        if strategy == 0:
            # Intra-A first, then intra-B, then cross
            order = sorted(missing_edges, key=edge_category)
        elif strategy == 1:
            # Intra-B first, then intra-A, then cross
            order = sorted(missing_edges, key=lambda e: 2 - edge_category(e))
        elif strategy == 2:
            # Random shuffle
            order = missing_edges[:]
            rng.shuffle(order)
        elif strategy == 3:
            # Alternate intra and random cross
            intra = [e for e in missing_edges if edge_category(e) < 2]
            cross = [e for e in missing_edges if edge_category(e) == 2]
            rng.shuffle(intra)
            rng.shuffle(cross)
            # Interleave
            order = []
            i, j = 0, 0
            while i < len(intra) or j < len(cross):
                if i < len(intra):
                    order.append(intra[i])
                    i += 1
                if j < len(cross):
                    order.append(cross[j])
                    j += 1
        else:
            # Reverse of intra-A priority
            order = sorted(missing_edges, key=edge_category, reverse=True)

        # Greedy pass with time limit
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        greedy_limit = time.time() + min(remaining_time * 0.5, 5)
        edge_set = greedy_with_ordering(order, max_time=greedy_limit)

        # Local improvement
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        improve_limit = time.time() + min(remaining_time * 0.5, 5)
        edge_set = local_improve(edge_set, max_time=improve_limit)

        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = edge_set

        # Stop restarting if we've used 70% of time
        elapsed = time.time() - start_time
        if elapsed > 0.7 * (deadline - start_time):
            break

    # Final local improvement on best result with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        final_limit = time.time() + remaining_time * 0.8
        best_edges = local_improve(best_edges, max_time=final_limit)
        best_count = len(best_edges)

    # Convert to required format
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return final_edges

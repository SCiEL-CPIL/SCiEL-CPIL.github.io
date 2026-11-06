import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def pc(x):
        return x.bit_count()

    def check_regular_mask(adj, mask, n):
        """Check if induced subgraph on mask is regular."""
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = pc(adj[v] & mask)
            degs.append(deg)
            if len(degs) > 1 and degs[0] != deg:
                return False
        return len(set(degs)) == 1

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def full_check(adj, n):
        """Check all subsets of size >= 6."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_regular_mask(adj, mask, n):
                    return False
        return True

    def incremental_check(adj, n, new_vertex):
        """Check if adding new_vertex creates any regular induced subgraph of size >= 6.
        Only check subsets that include new_vertex."""
        # For each subset S of existing vertices (size 5 to n-1), check if S ∪ {new_vertex} is regular
        existing = [i for i in range(n) if i != new_vertex]
        for k in range(5, n):  # k = size of existing subset, total = k+1 >= 6
            for combo in itertools.combinations(existing, k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                mask |= (1 << new_vertex)
                if check_regular_mask(adj, mask, n):
                    return False
        return True

    # Try n=21 first with incremental construction
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break

        best_adj = None
        best_n_found = 0

        # Multiple random restarts with incremental construction
        for restart in range(30):
            if time.time() > deadline - 5:
                break

            adj = [0] * n
            current_n = 0

            # Build vertex by vertex
            for v in range(n):
                if time.time() > deadline - 5:
                    break

                # Try random edge patterns for this vertex
                found_valid = False
                for attempt in range(50):
                    # Randomly decide edges from v to existing vertices
                    candidate_adj = adj[:]
                    for u in range(v):
                        if random.random() < 0.5:
                            candidate_adj[v] |= (1 << u)
                            candidate_adj[u] |= (1 << v)

                    # Check if adding vertex v with these edges creates any regular subgraph of size >= 6
                    # Only need to check subsets that include v
                    valid = True
                    existing = list(range(v))
                    for k in range(5, min(v, n)):  # k existing vertices, total k+1
                        if k + 1 < 6:
                            continue
                        for combo in itertools.combinations(existing, k):
                            mask = 0
                            for w in combo:
                                mask |= (1 << w)
                            mask |= (1 << v)
                            if check_regular_mask(candidate_adj, mask, n):
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        adj = candidate_adj
                        current_n = v + 1
                        found_valid = True
                        break

                if not found_valid:
                    # Could not add vertex v, try different random seed or break
                    break

            if current_n > best_n_found:
                best_n_found = current_n
                best_adj = adj[:]

            # If we found a valid graph of size n, verify and return
            if current_n == n:
                if full_check(adj, n):
                    return {'n': n, 'edges': get_edges(adj, n)}

        # If we have a best_adj with best_n_found >= 6, try to use it
        if best_adj and best_n_found >= 6:
            if full_check(best_adj, best_n_found):
                return {'n': best_n_found, 'edges': get_edges(best_adj, best_n_found)}

    # Fallback: try smaller n with random + full check
    for n in [15, 12, 10]:
        if time.time() > deadline - 5:
            break
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if full_check(adj, n):
                return {'n': n, 'edges': get_edges(adj, n)}

    return {'n': 5, 'edges': []}

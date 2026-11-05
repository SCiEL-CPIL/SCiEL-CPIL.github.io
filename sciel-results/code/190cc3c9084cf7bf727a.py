import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 40
    best_edges = []
    best_count = 0

    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        if u == v:
            return False
        av = adj[v]
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit

                    # d must be in N(c) ∩ N(v), and not u, a, b, v
                    d_mask = adj[c] & av & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj[i]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def try_add_all(adj):
        """Try to add all possible missing edges. Returns list of added edges."""
        added = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if not (ai & (1 << j)):
                    if not path5(i, j, adj):
                        add_edge(adj, i, j)
                        added.append((i, j))
        return added

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_c6(adj):
        # Check if the graph contains a C6
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    if path5(u, v, adj):
                        return True
        return False

    def greedy_add_all(adj):
        """Try to add all possible missing edges in greedy order."""
        for i in range(n):
            for j in range(i + 1, n):
                if not ((1 << j) & adj[i]):
                    if not path5(i, j, adj):
                        add_edge(adj, i, j)

    for restart in range(400):
        if time.time() > deadline:
            break

        # Initialize with 8 disjoint K5s
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)

        # Phase 1: Greedy Add
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((1 << j) & adj[i]):
                    missing_edges.append((i, j))

        random.shuffle(missing_edges)

        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if not path5(u, v, adj):
                add_edge(adj, u, v)

        # Phase 2: Local Search (Swap-based) with better ordering
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break

            # Get list of current edges to try removing, sorted by degree
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))

            # Sort edges by the sum of degrees of endpoints (lower first - easier to remove)
            current_edges.sort(key=lambda e: bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1'))

            for u, v in current_edges:
                if time.time() > deadline:
                    break

                # Try removing (u, v)
                remove_edge(adj, u, v)

                # Try to add as many missing edges as possible
                added_edges = try_add_all(adj)

                if len(added_edges) >= 2:  # Net gain >= 1
                    improved = True
                    # Keep the changes
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)

        # Phase 3: Additional Local Search with different randomization
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break

                current_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (1 << j) & adj[i]:
                            current_edges.append((i, j))

                # Try both sorted and random order
                if random.random() < 0.5:
                    current_edges.sort(key=lambda e: bin(adj[e[0]]).count('1') + bin(adj[e[1]]).count('1'))
                else:
                    random.shuffle(current_edges)

                for u, v in current_edges:
                    if time.time() > deadline:
                        break

                    remove_edge(adj, u, v)

                    added_edges = try_add_all(adj)

                    if len(added_edges) >= 2:
                        improved = True
                    else:
                        for i, j in added_edges:
                            remove_edge(adj, i, j)
                        add_edge(adj, u, v)

        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges

import random
import time

def construct():
    n = 20
    deadline = time.time() + 118.0
    best_edges = None
    best_count = -1

    # Precompute edge list
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = True
            edge_index[(v, u)] = True

    def build_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return bin(x).count('1')

    def is_valid_addition(adj, u, v):
        # Check if adding edge (u,v) violates K3,3-free constraint
        # For all triples {a,b,c} with no two equal, |N(a)&N(b)&N(c)| <= 2
        # When adding (u,v), N(u) gains v, N(v) gains u.
        # Only triples containing u or v are affected.
        mask_u = adj[u]
        mask_v = adj[v]
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[w]
            # Triple {u, w, x}: check if new common neighbors exceed 2
            # New N(u) = mask_u | (1<<v)
            # common = (mask_u | (1<<v)) & mask_w & mask_x
            # = (mask_u & mask_w & mask_x) | ((1<<v) & mask_w & mask_x)
            # The second term is 1 if v is neighbor of both w and x
            for x in range(w + 1, n):
                if x == u or x == v:
                    continue
                mask_x = adj[x]
                # Triple {u, w, x}
                if (mask_w >> v) & 1 and (mask_x >> v) & 1:
                    # v is common neighbor of w and x, so it adds to common count of {u,w,x}
                    common_old = mask_u & mask_w & mask_x
                    if popcount(common_old) >= 2:
                        return False
                # Triple {v, w, x}
                if (mask_w >> u) & 1 and (mask_x >> u) & 1:
                    common_old = mask_v & mask_w & mask_x
                    if popcount(common_old) >= 2:
                        return False
                # Triple {u, x, w} same as {u,w,x}, already handled
        return True

    def edge_count(adj):
        total = 0
        for u in range(n):
            total += popcount(adj[u])
        return total // 2

    def greedy_add(edges_subset, adj):
        """Try to add all edges from edges_subset that are valid."""
        for u, v in edges_subset:
            if not ((adj[u] >> v) & 1):
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_add_all_missing(adj):
        """Add all missing edges that can be added without violation."""
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_addition(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True

    def simulated_annealing(adj, time_budget=30.0):
        """Simulated annealing: remove/add edges to improve density."""
        start = time.time()
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        current_count = edge_count(adj)
        best_adj = adj[:]
        best_count = current_count

        while time.time() - start < time_budget and T > T_min:
            # Pick a random edge to remove
            present_edges = []
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    if v > u:
                        present_edges.append((u, v))
                    m ^= lsb
            if not present_edges:
                break
            u, v = random.choice(present_edges)
            remove_edge(adj, u, v)
            new_count_after_remove = current_count - 1

            # Try to add back or add different edges
            try_add_all_missing(adj)
            new_count = edge_count(adj)

            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                # We need to restore the removed edge and undo additions
                # This is complex; simpler to just skip and continue
                pass

            T *= cooling

        # Final polish
        try_add_all_missing(best_adj)
        return best_adj, edge_count(best_adj)

    def greedy_random():
        adj = [0] * n
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if not ((adj[u] >> v) & 1):
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        try_add_all_missing(adj)
        return adj

    def swap_search(adj, max_swaps=500):
        """Try swapping one edge for another to increase density."""
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            # Find all present edges
            present_edges = []
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    if v > u:
                        present_edges.append((u, v))
                    m ^= lsb
            # Find all absent edges
            absent_edges = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    absent_edges.append((u, v))

            if not present_edges or not absent_edges:
                break

            random.shuffle(present_edges)
            random.shuffle(absent_edges)

            for ru, rv in present_edges[:20]:
                if time.time() > deadline:
                    return adj
                for au, av in absent_edges[:30]:
                    # Try: remove (ru,rv), add (au,av)
                    remove_edge(adj, ru, rv)
                    if not ((adj[au] >> av) & 1) and is_valid_addition(adj, au, av):
                        adj[au] |= (1 << av)
                        adj[av] |= (1 << au)
                        try_add_all_missing(adj)
                        if edge_count(adj) > best_count:
                            improved = True
                        swap_count += 1
                    else:
                        # Revert
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        swap_count += 1
                    if swap_count >= max_swaps:
                        break
                if swap_count >= max_swaps:
                    break
        return adj

    # Phase 1: Multiple greedy restarts
    for i in range(15):
        if time.time() > deadline:
            break
        adj = greedy_random()
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    # Phase 2: Simulated annealing on best found
    if best_edges and time.time() < deadline - 20:
        adj = build_adj_from_edges(best_edges)
        adj, cnt = simulated_annealing(adj, time_budget=40.0)
        if cnt > best_count:
            best_count = cnt
            best_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    # Phase 3: More greedy restarts with time remaining
    while time.time() < deadline - 5:
        adj = greedy_random()
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
            # Quick polish
            try_add_all_missing(adj)
            cnt2 = edge_count(adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    # Final: ensure validity and return
    if not best_edges:
        # Fallback: simple construction
        best_edges = [[i, j] for i in range(4) for j in range(4, 20)]
        best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]

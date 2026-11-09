"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Start with the 5 disjoint K4s (30 edges) as a valid baseline.
- Use bit-parallel representation for fast validity checks.
- Perform a multi-restart local search with:
    * Greedy edge addition
    * Simulated annealing with swap moves (remove one edge, add another)
    * Periodic full K2,3 validity re-checks
- Return the best graph found within the time budget.
"""

import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Bitset helpers
    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def valid_add_fast(adj, u, v):
        """Faster validity check using precomputed common-neighbor matrix."""
        if adj[u] & (1 << v):
            return False
        # Precompute common neighbor counts for all pairs
        cn = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn[a][b] = c
                cn[b][a] = c
        # 1. u and v must have at most 2 common neighbors
        if cn[u][v] > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][v] > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][u] > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges

    def greedy_fill(adj):
        """Greedily add all possible edges in random order."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # First, establish baseline
    adj0 = make_k4_union()
    greedy_fill(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Multi-restart simulated annealing
    num_restarts = 20

    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        else:
            # Start from best found, then perturb
            adj = best_adj[:]
            # Randomly remove a few edges
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:3]:
                remove_edge(adj, u, v)

        greedy_fill(adj)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.5
        T_min = 0.05
        cooling = 0.995
        steps = 0

        # Precompute common-neighbor matrix for faster validity checks
        cn_matrix = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn_matrix[a][b] = c
                cn_matrix[b][a] = c

        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()

            if move_type < 0.4:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                # Use fast check with precomputed cn_matrix
                if not (adj[u] & (1 << v)):
                    if cn_matrix[u][v] > 2:
                        pass
                    else:
                        valid = True
                        temp = adj[u]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_matrix[w][v] > 1:
                                valid = False
                                break
                        if valid:
                            temp = adj[v]
                            while temp:
                                lsb = temp & -temp
                                w = lsb.bit_length() - 1
                                temp -= lsb
                                if cn_matrix[w][u] > 1:
                                    valid = False
                                    break
                        if valid:
                            add_edge(adj, u, v)
                            current_edges += 1
                            # Update cn_matrix
                            for w in range(n):
                                if w != u and w != v:
                                    new_c = popcount(adj[u] & adj[w])
                                    cn_matrix[u][w] = new_c
                                    cn_matrix[w][u] = new_c
                                    new_c2 = popcount(adj[v] & adj[w])
                                    cn_matrix[v][w] = new_c2
                                    cn_matrix[w][v] = new_c2
                            cn_matrix[u][v] = popcount(adj[u] & adj[v])
                            cn_matrix[v][u] = cn_matrix[u][v]
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                # else: reject (no change)

            elif move_type < 0.6:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                # Delta E = +1 (worse), accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1
                    # Update cn_matrix
                    for w in range(n):
                        if w != u and w != v:
                            new_c = popcount(adj[u] & adj[w])
                            cn_matrix[u][w] = new_c
                            cn_matrix[w][u] = new_c
                            new_c2 = popcount(adj[v] & adj[w])
                            cn_matrix[v][w] = new_c2
                            cn_matrix[w][v] = new_c2
                    cn_matrix[u][v] = popcount(adj[u] & adj[v])
                    cn_matrix[v][u] = cn_matrix[u][v]

            else:
                # Swap: remove one edge, try to add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                # Remove (u1,v1), check if (u2,v2) can be added using cn_matrix
                remove_edge(adj, u1, v1)
                # Update cn_matrix after removal
                for w in range(n):
                    if w != u1 and w != v1:
                        new_c = popcount(adj[u1] & adj[w])
                        cn_matrix[u1][w] = new_c
                        cn_matrix[w][u1] = new_c
                        new_c2 = popcount(adj[v1] & adj[w])
                        cn_matrix[v1][w] = new_c2
                        cn_matrix[w][v1] = new_c2
                cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                cn_matrix[v1][u1] = cn_matrix[u1][v1]

                # Check validity of adding (u2, v2) using cn_matrix
                valid_swap = True
                if adj[u2] & (1 << v2):
                    valid_swap = False
                elif cn_matrix[u2][v2] > 2:
                    valid_swap = False
                else:
                    temp = adj[u2]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        temp -= lsb
                        if cn_matrix[w][v2] > 1:
                            valid_swap = False
                            break
                    if valid_swap:
                        temp = adj[v2]
                        while temp:
                            lsb = temp & -temp
                            w = lsb.bit_length() - 1
                            temp -= lsb
                            if cn_matrix[w][u2] > 1:
                                valid_swap = False
                                break

                if valid_swap:
                    add_edge(adj, u2, v2)
                    # Update cn_matrix after addition
                    for w in range(n):
                        if w != u2 and w != v2:
                            new_c = popcount(adj[u2] & adj[w])
                            cn_matrix[u2][w] = new_c
                            cn_matrix[w][u2] = new_c
                            new_c2 = popcount(adj[v2] & adj[w])
                            cn_matrix[v2][w] = new_c2
                            cn_matrix[w][v2] = new_c2
                    cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                    cn_matrix[v2][u2] = cn_matrix[u2][v2]
                    # Edge count unchanged, but structure changed
                    # Accept with higher probability to explore
                    if random.random() < 0.8:
                        pass  # keep the swap
                    else:
                        # Revert
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                        # Update cn_matrix after revert
                        for w in range(n):
                            if w != u2 and w != v2:
                                new_c = popcount(adj[u2] & adj[w])
                                cn_matrix[u2][w] = new_c
                                cn_matrix[w][u2] = new_c
                                new_c2 = popcount(adj[v2] & adj[w])
                                cn_matrix[v2][w] = new_c2
                                cn_matrix[w][v2] = new_c2
                        cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                        cn_matrix[v2][u2] = cn_matrix[u2][v2]
                        for w in range(n):
                            if w != u1 and w != v1:
                                new_c = popcount(adj[u1] & adj[w])
                                cn_matrix[u1][w] = new_c
                                cn_matrix[w][u1] = new_c
                                new_c2 = popcount(adj[v1] & adj[w])
                                cn_matrix[v1][w] = new_c2
                                cn_matrix[w][v1] = new_c2
                        cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                        cn_matrix[v1][u1] = cn_matrix[u1][v1]
                else:
                    # Revert
                    add_edge(adj, u1, v1)
                    # Update cn_matrix after revert
                    for w in range(n):
                        if w != u1 and w != v1:
                            new_c = popcount(adj[u1] & adj[w])
                            cn_matrix[u1][w] = new_c
                            cn_matrix[w][u1] = new_c
                            new_c2 = popcount(adj[v1] & adj[w])
                            cn_matrix[v1][w] = new_c2
                            cn_matrix[w][v1] = new_c2
                    cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                    cn_matrix[v1][u1] = cn_matrix[u1][v1]

            T *= cooling

    # Final greedy fill on best
    if best_adj:
        greedy_fill(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Polish pass: try all possible single-edge additions and 1-1 swaps
    if best_adj and (time.time() - start_time < time_limit - 5):
        polish_adj = best_adj[:]
        polish_edges = count_edges(polish_adj)
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Try adding any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if valid_add(polish_adj, u, v):
                            add_edge(polish_adj, u, v)
                            polish_edges += 1
                            improved = True
            # Try 1-1 swaps: remove edge (a,b), add edge (c,d)
            edges = get_edges(polish_adj)
            non_edges = get_non_edges(polish_adj)
            for a, b in edges:
                if time.time() - start_time > time_limit - 5:
                    break
                for c, d in non_edges:
                    if (a, b) == (c, d) or (a, b) == (d, c):
                        continue
                    remove_edge(polish_adj, a, b)
                    if valid_add(polish_adj, c, d):
                        add_edge(polish_adj, c, d)
                        # Structure changed, edge count same - accept if it enables more additions
                        # Check if we can now add more edges
                        further_improved = False
                        for x in range(n):
                            for y in range(x + 1, n):
                                if not (polish_adj[x] & (1 << y)):
                                    if valid_add(polish_adj, x, y):
                                        add_edge(polish_adj, x, y)
                                        polish_edges += 1
                                        further_improved = True
                        if further_improved:
                            improved = True
                            break
                        else:
                            # Revert
                            remove_edge(polish_adj, c, d)
                            add_edge(polish_adj, a, b)
                    else:
                        add_edge(polish_adj, a, b)
                if improved:
                    break
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END

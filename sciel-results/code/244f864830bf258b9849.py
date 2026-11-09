"""Rewritten program for cochromatic partition search."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80  # safety margin before 90s kill

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute: for each vertex, the set of vertices it can be adjacent to
    # and non-adjacent to (excluding itself)
    # adj_mask[v] has bits for neighbors of v
    # non_adj_mask[v] has bits for non-neighbors of v (excluding v itself)
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = (all_mask ^ (1 << v)) & ~adj[v]

    # State: parts is a list of (type, vertex_set_mask, vertex_list)
    # type: 1 = clique, 0 = independent set
    # We'll track best solution found

    best_parts = None  # list of (type, mask, list)
    best_count = n + 1

    # Order vertices by degree (descending) for better pruning
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    # Map: position in order -> vertex
    # vertex -> position in order
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i

    # We process vertices in order[0], order[1], ..., order[n-1]
    # At step k, we've assigned order[0..k-1]

    def can_add_clique(part_mask, v):
        # v must be adjacent to all vertices in part_mask
        # adj[v] has bits for neighbors of v
        return (part_mask & ~adj[v]) == 0

    def can_add_independent(part_mask, v):
        # v must be non-adjacent to all vertices in part_mask
        # non_adj[v] has bits for non-neighbors of v
        return (part_mask & ~non_adj[v]) == 0

    # Backtracking search
    # parts: list of (type, mask, list)
    # next_pos: index into order for next vertex to assign
    # current_count: number of parts so far
    def backtrack(next_pos, parts, current_count):
        nonlocal best_parts, best_count

        if time.time() > deadline:
            return

        # Prune: if current_count >= best_count, no improvement possible
        if current_count >= best_count:
            return

        if next_pos == n:
            # All vertices assigned
            if current_count < best_count:
                best_count = current_count
                best_parts = [(t, m, list(l)) for t, m, l in parts]
            return

        v = order[next_pos]

        # Try adding v to each existing part
        for i in range(len(parts)):
            t, m, l = parts[i]
            if t == 1:  # clique
                if can_add_clique(m, v):
                    parts[i] = (1, m | (1 << v), l + [v])
                    backtrack(next_pos + 1, parts, current_count)
                    parts[i] = (1, m, l)
            else:  # independent set
                if can_add_independent(m, v):
                    parts[i] = (0, m | (1 << v), l + [v])
                    backtrack(next_pos + 1, parts, current_count)
                    parts[i] = (0, m, l)

        # Try creating a new part (clique)
        if current_count + 1 < best_count:
            parts.append((1, 1 << v, [v]))
            backtrack(next_pos + 1, parts, current_count + 1)
            parts.pop()

        # Try creating a new part (independent set)
        if current_count + 1 < best_count:
            parts.append((0, 1 << v, [v]))
            backtrack(next_pos + 1, parts, current_count + 1)
            parts.pop()

    # Initial greedy solution for upper bound
    def greedy_solution():
        parts = []  # list of (type, mask, list)
        for v in range(n):
            placed = False
            for i in range(len(parts)):
                t, m, l = parts[i]
                if t == 1 and can_add_clique(m, v):
                    parts[i] = (1, m | (1 << v), l + [v])
                    placed = True
                    break
                elif t == 0 and can_add_independent(m, v):
                    parts[i] = (0, m | (1 << v), l + [v])
                    placed = True
                    break
            if not placed:
                parts.append((1, 1 << v, [v]))
        return parts

    greedy = greedy_solution()
    best_count = len(greedy)
    best_parts = [(t, m, list(l)) for t, m, l in greedy]

    # Run backtracking
    backtrack(0, [], 0)

    # Convert best_parts to list of lists of ints
    result = []
    for t, m, l in best_parts:
        result.append(list(l))

    # Ensure types are correct: lists of Python ints
    return result

# EVOLVE-BLOCK-END

"""Strong edge colouring with bitmask-accelerated local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, deadline):
        if not edges:
            return []
        n = max(max(u, v) for u, v in edges) + 1
        m = len(edges)

        # Build adjacency as sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets using bitmasks for speed
        # conflicts[i] is a bitmask where bit j is set if edge i conflicts with edge j
        conflicts = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i] |= (1 << j)
                    conflicts[j] |= (1 << i)
                    continue
                # Joined by an edge?
                if v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1] or u2 in adj[u1]:
                    conflicts[i] |= (1 << j)
                    conflicts[j] |= (1 << i)

        # Initial greedy colouring: order edges by conflict count descending
        conflict_counts = [bin(conflicts[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -conflict_counts[i])
        colour = [-1] * m
        used_colours = 0

        # For each colour, maintain a bitmask of edges with that colour
        colour_masks = []  # list of bitmasks

        for i in order:
            # Find first colour not used by any conflicting edge
            taken = 0
            for j in range(m):
                if (conflicts[i] >> j) & 1:
                    if colour[j] >= 0:
                        taken |= (1 << colour[j])
            c = 0
            while (taken >> c) & 1:
                c += 1
            colour[i] = c
            if c >= len(colour_masks):
                colour_masks.append(0)
            colour_masks[c] |= (1 << i)
            if c + 1 > used_colours:
                used_colours = c + 1

        # Local search: iteratively try to eliminate colours
        # Strategy: for colour c from high to low, try to reassign all its edges to lower colours
        # Use bitmask operations for speed

        max_iterations = 200
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            progress = False
            for c in range(used_colours - 1, 0, -1):
                if time.time() > deadline:
                    break
                # Collect edges with colour c
                mask_c = colour_masks[c]
                if mask_c == 0:
                    continue

                # Try to move each edge in colour c to a lower colour
                edges_of_c = [i for i in range(m) if (mask_c >> i) & 1]
                all_moved = True
                temp_colour = colour[:]
                temp_masks = colour_masks[:]

                for i in edges_of_c:
                    # Find a colour c2 < c such that no conflict of i has colour c2
                    # (considering the temporary state)
                    found = False
                    for c2 in range(c):
                        # Check if any conflict of i has colour c2 in temp state
                        conflict_mask = conflicts[i]
                        if (conflict_mask & temp_masks[c2]) == 0:
                            # Move edge i to colour c2
                            temp_colour[i] = c2
                            temp_masks[c] &= ~(1 << i)
                            temp_masks[c2] |= (1 << i)
                            found = True
                            break
                    if not found:
                        all_moved = False
                        break

                if all_moved:
                    colour = temp_colour
                    colour_masks = temp_masks
                    used_colours -= 1
                    progress = True
                    break  # Restart the outer loop since colour indices changed

            if not progress:
                # Try swap-based improvement: for each pair of colours, try to rearrange
                # to free up a colour
                swap_done = False
                for c in range(used_colours - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    mask_c = colour_masks[c]
                    if mask_c == 0:
                        continue
                    edges_of_c = [i for i in range(m) if (mask_c >> i) & 1]
                    if not edges_of_c:
                        continue

                    # For each edge in colour c, try to swap with an edge in a lower colour
                    for i in edges_of_c:
                        if time.time() > deadline:
                            break
                        for c2 in range(c):
                            # Find edges in colour c2 that conflict with i
                            conflict_mask = conflicts[i]
                            blockers = [j for j in range(m)
                                        if (colour_masks[c2] >> j) & 1 and (conflict_mask >> j) & 1]
                            if len(blockers) == 1:
                                k = blockers[0]
                                # Check if k can move to colour c (after i moves to c2)
                                # k's conflicts should not have colour c (excluding i)
                                conflict_mask_k = conflicts[k]
                                # Remove i from consideration since i will move to c2
                                conflict_mask_k_no_i = conflict_mask_k & ~(1 << i)
                                if (conflict_mask_k_no_i & colour_masks[c]) == 0:
                                    # Perform the swap
                                    colour[i] = c2
                                    colour[k] = c
                                    colour_masks[c] = (colour_masks[c] & ~(1 << i)) | (1 << k)
                                    colour_masks[c2] = (colour_masks[c2] & ~(1 << k)) | (1 << i)
                                    swap_done = True
                                    break
                        if swap_done:
                            break
                    if swap_done:
                        break

                if not swap_done:
                    # Try randomized restarts: shuffle edge order and re-greedy
                    if iteration % 10 == 0:
                        random_order = list(range(m))
                        random.shuffle(random_order)
                        # Sort by conflict count within random tie-breaking
                        random_order.sort(key=lambda i: -conflict_counts[i])
                        new_colour = [-1] * m
                        new_masks = []
                        new_used = 0
                        for i in random_order:
                            taken = 0
                            for j in range(m):
                                if (conflicts[i] >> j) & 1:
                                    if new_colour[j] >= 0:
                                        taken |= (1 << new_colour[j])
                            c = 0
                            while (taken >> c) & 1:
                                c += 1
                            new_colour[i] = c
                            if c >= len(new_masks):
                                new_masks.append(0)
                            new_masks[c] |= (1 << i)
                            if c + 1 > new_used:
                                new_used = c + 1
                        # Accept if better
                        if new_used < used_colours:
                            colour = new_colour
                            colour_masks = new_masks
                            used_colours = new_used

        # Recompress colours
        colour_map = {}
        new_colour = [0] * m
        next_c = 0
        for i in range(m):
            if colour[i] not in colour_map:
                colour_map[colour[i]] = next_c
                next_c += 1
            new_colour[i] = colour_map[colour[i]]
        return new_colour

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        cols = solve_instance(edges, DEADLINE)
        results.append(cols)

    return results
# EVOLVE-BLOCK-END

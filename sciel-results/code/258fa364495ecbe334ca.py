"""Cochromatic partition solver with bitmask DP and reconstruction."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute for every subset whether it is a clique and/or independent set.
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_indep[m] = True

    for popcount in range(2, n + 1):
        for mask in range(1 << n):
            if mask.bit_count() != popcount:
                continue
            if is_clique[mask] or is_indep[mask]:
                continue

            temp = mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                rest = mask ^ lsb

                if is_clique[rest]:
                    if (rest & ~adj[v]) == 0:
                        is_clique[mask] = True
                        break

                if not is_clique[mask] and is_indep[rest]:
                    if (rest & adj[v]) == 0:
                        is_indep[mask] = True
                        break

                temp ^= lsb

    # Precompute valid_part for speed
    valid_part = [False] * (1 << n)
    for mask in range(1, 1 << n):
        valid_part[mask] = is_clique[mask] or is_indep[mask]

    # DP to find minimum number of parts
    # min_parts[mask] = minimum number of valid parts to cover mask
    # choice[mask] = one valid sub-part chosen in the optimal solution for mask
    min_parts = [n + 1] * (1 << n)
    choice = [0] * (1 << n)
    min_parts[0] = 0

    for mask in range(1, 1 << n):
        # Iterate through submasks of mask
        # Optimization: Only consider submasks that contain the least significant bit of mask
        # to avoid duplicate permutations of the same partition
        lsb = mask & (-mask)
        sub = mask
        while sub:
            if sub & lsb:
                if valid_part[sub]:
                    rest = mask ^ sub
                    if min_parts[rest] + 1 < min_parts[mask]:
                        min_parts[mask] = min_parts[rest] + 1
                        choice[mask] = sub
            sub = (sub - 1) & mask

    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        verts = []
        temp = sub
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        parts.append(verts)
        mask ^= sub

    # Ensure types are correct: Python int, list
    final_result = []
    for part in parts:
        final_result.append([int(v) for v in part])

    return final_result

# EVOLVE-BLOCK-END

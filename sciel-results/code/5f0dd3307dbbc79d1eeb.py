"""Direct brute-force over all induced subsets of size >= 8 to maximize distinct degrees."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s wall-clock
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency as integer bitmasks.
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute popcount for all 17-bit masks
    pc = [0] * (1 << n)
    for i in range(1, 1 << n):
        pc[i] = pc[i >> 1] + (i & 1)

    best_count = -1
    best_vertices = list(range(minv))  # fallback seed

    vertices = list(range(n))

    # Enumerate all subsets of size minv..n.
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            # Compute degrees using precomputed popcount table
            degs = set()
            for v in combo:
                degs.add(pc[adj[v] & mask])
            c = len(degs)
            if c > best_count:
                best_count = c
                best_vertices = list(combo)
                # If we found a perfect grading (all distinct degrees), 
                # we can't do better for this size. But larger sizes could still beat it.
                # However, if size == 17, that's the max possible.
                if c == size:
                    # Perfect for this size; continue to check larger sizes if any remain
                    pass

    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END

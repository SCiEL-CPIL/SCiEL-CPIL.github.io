import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin

    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330

    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)

    # Build compatibility graph: edge if intersection size <= 2 (compatible)
    # Use numpy boolean matrix for vectorized operations
    compat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            inter = (masks[i] & masks[j]).bit_count()
            if inter <= 2:
                compat[i, j] = True
                compat[j, i] = True

    # Tomita's branch-and-bound for maximum clique
    # Colors are computed greedily in the order of the candidate list
    def compute_colors(candidates):
        """Greedy coloring of the subgraph induced by candidates.
        Returns (color_array, color_classes) where color_array[i] is the color
        of candidates[i] (1-indexed), and color_classes[c] is the list of
        indices in candidates with color c."""
        m = len(candidates)
        if m == 0:
            return [], []

        # Work with local indices 0..m-1
        # color[i] = color assigned to candidates[i], 0 = uncolored
        color = [0] * m
        color_classes = []  # color_classes[c-1] = list of local indices with color c

        for i in range(m):
            # Find smallest color not used by already-colored neighbors of i
            used = set()
            for j in range(i):
                if color[j] > 0 and compat[candidates[i], candidates[j]]:
                    used.add(color[j])
            # Assign smallest available color
            c = 1
            while c in used:
                c += 1
            color[i] = c
            if c > len(color_classes):
                color_classes.append([])
            color_classes[c - 1].append(i)

        return color, color_classes

    def branch_and_bound(candidates, clique, best_size, best_clique, deadline):
        """Tomita's algorithm. candidates is a list of node indices.
        clique is the current partial clique (list of node indices)."""
        if time.time() > deadline:
            return False

        m = len(candidates)
        if m == 0:
            if len(clique) > best_size[0]:
                best_size[0] = len(clique)
                best_clique[0] = list(clique)
            return False

        # Compute coloring
        color, color_classes = compute_colors(candidates)

        # Process in reverse order of color (highest color first)
        # This gives a valid upper bound at each step
        # We maintain a "remaining" list and process nodes from the end
        remaining = list(candidates)
        # We'll process groups by color in reverse
        # Actually, the standard approach: process nodes in the order they appear,
        # but use the color as an upper bound.
        # Let's use the standard MCQ approach:
        # Order candidates by color, process from highest color to lowest.

        # Reorder: sort by color descending
        order = sorted(range(m), key=lambda i: -color[i])
        # Now process in this order
        # We maintain a list of "available" nodes
        avail = [candidates[order[i]] for i in range(m)]
        # colors in this order
        avail_colors = [color[order[i]] for i in range(m)]

        # Process from end to start (highest color first)
        # At position k in avail (0-indexed from start), the upper bound is avail_colors[k]
        # We process from the last element (highest color) backwards

        for k in range(m - 1, -1, -1):
            if time.time() > deadline:
                return True  # timeout
            # Upper bound: current clique size + avail_colors[k]
            if len(clique) + avail_colors[k] <= best_size[0]:
                break  # No need to explore further

            v = avail[k]
            # Add v to clique
            clique.append(v)
            # New candidates: those in avail[0..k-1] that are adjacent to v
            new_candidates = [avail[i] for i in range(k) if compat[v, avail[i]]]
            branch_and_bound(new_candidates, clique, best_size, best_clique, deadline)
            clique.pop()
            # Remove v from avail
            avail.pop(k)

        return False

    # Run maximum clique
    best_size = [0]
    best_clique = [[]]

    # Initial: all nodes as candidates
    all_nodes = list(range(n))
    branch_and_bound(all_nodes, [], best_size, best_clique, deadline)

    best_is = set(best_clique[0])
    best_size_val = len(best_is)

    # If we didn't find a good clique (timeout or small), fall back to greedy
    if best_size_val < 20 and time.time() < deadline - 5:
        # Greedy fallback
        adj_conflict = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and not compat[i, j]:
                    adj_conflict[i].add(j)

        is_set = set()
        available = set(range(n))
        while available and time.time() < deadline - 2:
            # Pick node with fewest conflicts among available
            best_node = None
            best_conf = float('inf')
            for v in available:
                conf = len(adj_conflict[v] & is_set)
                if conf == 0 and len(adj_conflict[v] & available) < best_conf:
                    best_conf = len(adj_conflict[v] & available)
                    best_node = v
            if best_node is None:
                break
            is_set.add(best_node)
            available.discard(best_node)
            for nb in adj_conflict[best_node]:
                available.discard(nb)
        if len(is_set) > best_size_val:
            best_is = is_set
            best_size_val = len(is_set)

    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        # For each selected subset, add all 16 sign variations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)

    return np.array(vectors, dtype=np.int64)

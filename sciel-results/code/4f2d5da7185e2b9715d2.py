import time
import math
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1

    # Build conflict graph: a and b conflict if a+b is a perfect square
    # Also a conflicts with itself if 2a is a perfect square (diagonal constraint)
    # We want maximum independent set in this conflict graph
    # Equivalently: maximum clique in the complement graph

    # valid: a is valid if 2a is NOT a perfect square (diagonal constraint)
    valid = [False] * (N + 1)
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid[a] = True

    # Complement adjacency: comp_adj[i] is a bitset where bit j is set
    # if i and j are NOT in conflict (i.e., i+j is not a perfect square)
    # and both are valid, and i != j
    comp_adj = [0] * (N + 1)
    for a in range(1, N + 1):
        if not valid[a]:
            continue
        for b in range(1, N + 1):
            if a == b or not valid[b]:
                continue
            if (a + b) not in squares:
                comp_adj[a] |= (1 << b)

    # Map valid nodes to 0..M-1 for compact bitset representation
    valid_nodes = [a for a in range(1, N + 1) if valid[a]]
    M = len(valid_nodes)
    node_to_idx = {a: i for i, a in enumerate(valid_nodes)}
    idx_to_node = [0] * M
    for i, a in enumerate(valid_nodes):
        idx_to_node[i] = a

    # Compact adjacency: adj[i] is a Python int bitset over 0..M-1
    adj = [0] * M
    for i, a in enumerate(valid_nodes):
        mask = 0
        for j, b in enumerate(valid_nodes):
            if i != j and (a + b) not in squares:
                mask |= (1 << j)
        adj[i] = mask

    # Maximum clique via branch-and-bound with greedy coloring bound
    # Uses bitset operations for speed

    best_clique = []
    best_size = 0

    def bit_count(x):
        return bin(x).count('1')

    # Greedy coloring of a candidate set (represented as bitset)
    # Returns list of color classes and the maximum color used
    # We use a sequential greedy coloring: assign each vertex the smallest color
    # not used by its already-colored neighbors in the set

    def greedy_color(candidate_bits):
        """
        Greedily color the vertices in candidate_bits.
        Returns (color_of_vertex_dict, max_color)
        Vertices are processed in order of decreasing degree within candidate.
        """
        # Get list of vertices in candidate
        verts = []
        x = candidate_bits
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            verts.append(i)
            x ^= lsb

        if not verts:
            return {}, 0

        # Order by degree in candidate (descending)
        deg = [bit_count(adj[v] & candidate_bits) for v in verts]
        order = sorted(range(len(verts)), key=lambda k: -deg[k])
        verts_ordered = [verts[k] for k in order]

        color = {}
        color_used = [set() for _ in range(len(verts_ordered) + 1)]
        # color_used[c] = set of vertices with color c

        for v in verts_ordered:
            # Find smallest color not used by neighbors of v that are already colored
            neighbor_colors = set()
            for c in range(len(verts_ordered)):
                for u in color_used[c]:
                    if (adj[v] >> u) & 1:
                        neighbor_colors.add(c)
                        break
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            color_used[c].add(v)

        max_color = max(color.values()) if color else 0
        return color, max_color + 1  # 1-indexed color count

    def max_clique_bb(candidates, current_clique, depth=0):
        """
        Branch and bound for maximum clique.
        candidates: bitset of vertices to consider
        current_clique: list of vertices in current clique
        """
        nonlocal best_clique, best_size

        if time.time() > deadline - 2.0:
            return

        # Bound: if current size + max possible <= best, prune
        if len(current_clique) + bit_count(candidates) <= best_size:
            return

        if candidates == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return

        # Greedy coloring to get upper bound
        color_map, num_colors = greedy_color(candidates)

        # If even with all colors we can't beat best
        if len(current_clique) + num_colors <= best_size:
            return

        # Order vertices by color (process highest color first)
        # Group by color
        color_groups = {}
        for v, c in color_map.items():
            if c not in color_groups:
                color_groups[c] = []
            color_groups[c].append(v)

        # Process from highest color to lowest
        max_c = num_colors - 1
        for c in range(max_c, -1, -1):
            if time.time() > deadline - 2.0:
                return
            if len(current_clique) + c + 1 <= best_size:
                break  # All remaining colors are <= c, so can't improve

            group = color_groups.get(c, [])
            for v in group:
                if time.time() > deadline - 2.0:
                    return
                # Check if v is still in candidates
                if not (candidates & (1 << v)):
                    continue

                # Recurse: add v to clique, restrict candidates to neighbors of v
                new_candidates = candidates & adj[v]
                # Remove v and all vertices with color <= c (they can't be in a larger clique with v)
                # Actually, standard approach: new_candidates = candidates & adj[v] & (vertices with color > c)
                # But simpler: just use candidates & adj[v], and the bound will handle it
                new_candidates = candidates & adj[v]
                # Remove v itself
                new_candidates &= ~(1 << v)

                current_clique.append(v)
                max_clique_bb(new_candidates, current_clique, depth + 1)
                current_clique.pop()

                # Remove v from candidates for subsequent iterations
                candidates &= ~(1 << v)

                # Update bound check
                if len(current_clique) + bit_count(candidates) <= best_size:
                    return

    # Initial greedy clique to set a lower bound
    def greedy_clique_init():
        """Greedy construction of a large clique."""
        remaining = (1 << M) - 1
        clique = []
        while remaining:
            # Pick vertex with max degree in remaining
            best_v = -1
            best_deg = -1
            x = remaining
            while x:
                lsb = x & (-x)
                v = lsb.bit_length() - 1
                d = bit_count(adj[v] & remaining)
                if d > best_deg:
                    best_deg = d
                    best_v = v
                x ^= lsb
            if best_v < 0:
                break
            clique.append(best_v)
            remaining &= adj[best_v]
            remaining &= ~(1 << best_v)
        return clique

    # Get initial lower bound from greedy
    init_clique = greedy_clique_init()
    best_size = len(init_clique)
    best_clique = list(init_clique)

    # Run branch and bound starting with all vertices
    all_vertices = (1 << M) - 1
    max_clique_bb(all_vertices, [])

    # Convert back to original node values
    result = [idx_to_node[i] for i in best_clique]

    # Verify the result
    def is_valid(subset):
        sset = set(subset)
        for a in sset:
            if 2 * a in squares:
                return False
        for a in sset:
            for b in sset:
                if a <= b and (a + b) in squares:
                    return False
        return True

    if not is_valid(result):
        # Fallback: use greedy result
        result = [idx_to_node[i] for i in init_clique]

    if not result:
        # Last resort: find any valid node
        for a in range(1, N + 1):
            if valid[a]:
                result = [a]
                break

    return result

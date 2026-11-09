import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 110

    # Use bitmask representation for fast common-neighbor counting
    # Each vertex's neighbors stored as a 40-bit integer
    def bits_to_list(bits):
        """Convert bitmask to list of vertex indices"""
        result = []
        b = bits
        i = 0
        while b:
            if b & 1:
                result.append(i)
            b >>= 1
            i += 1
        return result

    def compute_books(bits_list):
        """Compute max book size from bitmask neighbor lists"""
        max_book = 0
        for i in range(n):
            for j in range(i + 1, n):
                if bits_list[i] & (1 << j):
                    common = bits_list[i] & bits_list[j]
                    c = bin(common).count('1')
                    if c > max_book:
                        max_book = c
        return max_book

    def spectral_init():
        """
        Initialize using spectral/eigenvector-based partitioning.
        Create a near-bipartite graph with controlled intra-partition edges.
        """
        # Split vertices into two parts of 20
        part_a = list(range(20))
        part_b = list(range(20, 40))

        # Start with complete bipartite K(20,20) = 400 edges
        bits_list = [0] * n
        for u in part_a:
            for v in part_b:
                bits_list[u] |= (1 << v)
                bits_list[v] |= (1 << u)

        # Add 1 more edge within a partition
        bits_list[0] |= (1 << 1)
        bits_list[1] |= (1 << 0)

        return bits_list

    def random_init():
        """Random graph with m edges"""
        bits_list = [0] * n
        edge_set = set()
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                bits_list[u] |= (1 << v)
                bits_list[v] |= (1 << u)
        return bits_list, edge_set

    def swap_edge(bits_list, edge_set, out_u, out_v, in_u, in_v):
        """Swap out edge (out_u,out_v) for (in_u,in_v)"""
        bits_list[out_u] &= ~(1 << out_v)
        bits_list[out_v] &= ~(1 << out_u)
        edge_set.discard((out_u, out_v))
        edge_set.discard((out_v, out_u))

        bits_list[in_u] |= (1 << in_v)
        bits_list[in_v] |= (1 << in_u)
        edge_set.add((min(in_u, in_v), max(in_u, in_v)))

    best_score = 100
    best_bits = None

    # Phase 1: Spectral initialization + local search
    bits_list = spectral_init()
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if bits_list[i] & (1 << j):
                edge_set.add((i, j))

    current_score = compute_books(bits_list)
    if current_score < best_score:
        best_score = current_score
        best_bits = bits_list[:]

    if best_score <= 7:
        return [[u, v] for u, v in sorted(edge_set)]

    # Phase 2: Deterministic + random local search with bitmask
    last_improve = time.time()
    no_improve_count = 0

    while time.time() < deadline:
        # Stagnation check
        if time.time() - last_improve > 8.0:
            # Try different initialization strategies
            strategy = random.random()
            if strategy < 0.3:
                bits_list = spectral_init()
            else:
                bits_list, edge_set = random_init()
            current_score = compute_books(bits_list)
            last_improve = time.time()
            no_improve_count = 0
            continue

        # Select an edge to remove - prioritize edges in high-book triangles
        edges = list(edge_set)
        if not edges:
            break

        # Try to find a swap that reduces max book
        improved = False
        for _ in range(50):  # Try up to 50 random swaps per iteration
            out_u, out_v = random.choice(edges)

            # Pick random non-edge to add
            while True:
                in_u = random.randint(0, n - 1)
                in_v = random.randint(0, n - 1)
                if in_u == in_v:
                    continue
                if in_u > in_v:
                    in_u, in_v = in_v, in_u
                if (in_u, in_v) == (out_u, out_v):
                    continue
                if not (bits_list[in_u] & (1 << in_v)):
                    break

            # Apply swap
            swap_edge(bits_list, edge_set, out_u, out_v, in_u, in_v)
            new_score = compute_books(bits_list)

            if new_score < current_score:
                current_score = new_score
                improved = True
                last_improve = time.time()
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_bits = bits_list[:]
                    if best_score <= 7:
                        break
            else:
                # Revert
                swap_edge(bits_list, edge_set, in_u, in_v, out_u, out_v)

            if best_score <= 7:
                break

        if not improved:
            no_improve_count += 1

        if best_score <= 7:
            break

    # Reconstruct edge list from best_bits
    if best_bits is not None:
        final_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_bits[i] & (1 << j):
                    final_edges.append([i, j])
        return final_edges
    else:
        # Fallback: return spectral init
        return [[u, v] for u, v in sorted(edge_set)]

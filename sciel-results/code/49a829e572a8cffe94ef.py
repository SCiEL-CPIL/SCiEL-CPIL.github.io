import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    def build_adj_bits(n, edges):
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    def is_valid_fast(n, adj_bits):
        """Fast validation using precomputed adjacency bitmasks."""
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj_bits[i] >> j) & 1):
                    continue
                common_ij = adj_bits[i] & adj_bits[j]
                k_mask = common_ij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_bit = k_mask & -k_mask
                    k = k_bit.bit_length() - 1
                    k_mask ^= k_bit
                    common_ijk = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    l_mask = common_ijk & ~((1 << (k + 1)) - 1)
                    if l_mask:
                        return False

        # Check Independent Set of 6 using complement
        all_mask = (1 << n) - 1
        comp_bits = []
        for i in range(n):
            comp_bits.append(all_mask ^ adj_bits[i] ^ (1 << i))

        for i in range(n):
            for j in range(i + 1, n):
                if not ((comp_bits[i] >> j) & 1):
                    continue
                common_ij = comp_bits[i] & comp_bits[j]
                k_mask = common_ij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_bit = k_mask & -k_mask
                    k = k_bit.bit_length() - 1
                    k_mask ^= k_bit
                    common_ijk = comp_bits[i] & comp_bits[j] & comp_bits[k]
                    l_mask = common_ijk & ~((1 << (k + 1)) - 1)
                    while l_mask:
                        l_bit = l_mask & -l_mask
                        l = l_bit.bit_length() - 1
                        l_mask ^= l_bit
                        common_ijkl = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l]
                        m_mask = common_ijkl & ~((1 << (l + 1)) - 1)
                        while m_mask:
                            m_bit = m_mask & -m_mask
                            m = m_bit.bit_length() - 1
                            m_mask ^= m_bit
                            common_ijklm = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l] & comp_bits[m]
                            p_mask = common_ijklm & ~((1 << (m + 1)) - 1)
                            if p_mask:
                                return False

        return True

    # Known 35-vertex construction: vertices are 3-subsets of {0,...,6}
    # Two vertices adjacent iff they share exactly 1 element
    vertices = list(combinations(range(7), 3))
    n35 = len(vertices)  # C(7,3) = 35
    edges35 = []
    for i in range(n35):
        for j in range(i + 1, n35):
            inter_size = len(set(vertices[i]).intersection(vertices[j]))
            if inter_size == 1:
                edges35.append([i, j])

    adj_bits35 = build_adj_bits(n35, edges35)
    if is_valid_fast(n35, adj_bits35):
        return {'n': n35, 'edges': edges35}

    # Try random local search for larger graphs
    # Start with n=36 and try to find a valid graph
    for n in range(36, 41):
        if time.time() > deadline - 10:
            break

        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break

            # Random initial graph
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)

            # Local search: flip edges to reduce violations
            for step in range(200):
                if time.time() > deadline - 5:
                    break

                # Check if valid
                if is_valid_fast(n, adj_bits):
                    # Build edges from adj_bits
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_bits[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

                # Find a violation and try to fix it
                # Try flipping a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i

                # Flip edge (i, j)
                if (adj_bits[i] >> j) & 1:
                    adj_bits[i] ^= (1 << j)
                    adj_bits[j] ^= (1 << i)
                else:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)

    # Fallback: return the 35-vertex graph if valid, or a smaller known good one
    if is_valid_fast(n35, adj_bits35):
        return {'n': n35, 'edges': edges35}

    # Try Paley graph of order 17
    q = 17
    squares = {x * x % q for x in range(1, q)}
    adj_bits_17 = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                adj_bits_17[i] |= (1 << j)
                adj_bits_17[j] |= (1 << i)

    if is_valid_fast(q, adj_bits_17):
        edges_17 = []
        for i in range(q):
            for j in range(i + 1, q):
                if (adj_bits_17[i] >> j) & 1:
                    edges_17.append([i, j])
        return {'n': q, 'edges': edges_17}

    return {'n': 1, 'edges': []}

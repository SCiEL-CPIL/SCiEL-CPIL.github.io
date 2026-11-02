import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate candidate vectors: weight-2 and weight-3 integer vectors
    # Condition: for all pairs i,j: ||v_i - v_j||^2 >= max(||v_k||^2)
    # With mixed norms, let M = max norm^2. Need ||v_i - v_j||^2 >= M for all i != j.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # So we need: ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= M
    # i.e., v_i.v_j <= (||v_i||^2 + ||v_j||^2 - M) / 2

    # Let's use only weight-2 vectors (norm^2 = 2) and weight-3 vectors (norm^2 = 3)
    # M = 3 (max norm^2)
    # For two weight-2: 2+2-2*d >= 3 => d <= 0.5 => d <= 0 (integer dot)
    # For w2 and w3: 2+3-2*d >= 3 => d <= 1
    # For two w3: 3+3-2*d >= 3 => d <= 1.5 => d <= 1

    # So conflict conditions:
    # w2-w2: dot > 0 (i.e., dot >= 1)
    # w2-w3: dot > 1 (i.e., dot >= 2)
    # w3-w3: dot > 1 (i.e., dot >= 2)

    # Generate weight-2 vectors
    w2_vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                w2_vecs.append(v)

    # Generate weight-3 vectors
    w3_vecs = []
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            w3_vecs.append(v)

    all_vecs = w2_vecs + w3_vecs
    n_w2 = len(w2_vecs)
    n_total = len(all_vecs)

    # Build conflict graph using numpy for speed
    V = np.array(all_vecs, dtype=np.int8)
    dots = V @ V.T  # (n_total, n_total)

    # Determine norms squared for each vector
    norms_sq = np.array([sum(x*x for x in v) for v in all_vecs])

    # Conflict matrix: conflict[i][j] = True if vectors i and j cannot coexist
    # Condition: dot[i][j] > (norms_sq[i] + norms_sq[j] - 3) / 2
    # i.e., 2*dot[i][j] > norms_sq[i] + norms_sq[j] - 3
    M = 3  # max norm squared
    rhs = (norms_sq[:, None] + norms_sq[None, :] - M) / 2.0
    conflict = (2.0 * dots > rhs).astype(np.int8)
    np.fill_diagonal(conflict, 0)

    # Build adjacency as bitsets for efficient MIS
    # Use Python integers as bitsets for the adjacency
    adj_bits = []
    for i in range(n_total):
        bits = 0
        row = conflict[i]
        # Set bit j if row[j] == 1
        # Use numpy to get indices where conflict exists
        j_indices = np.where(row)[0]
        for j in j_indices:
            if j != i:
                bits |= (1 << j)
        adj_bits.append(bits)

    # Tomita branch-and-bound MIS algorithm
    # State: candidate set C (as bitset), current solution S (as bitset)
    # Upper bound: greedy coloring of C gives an upper bound on MIS size

    best_size = 0
    best_solution = 0  # bitset of selected vertices

    def greedy_coloring(candidates):
        """Color the subgraph induced by candidates. Returns list of color classes."""
        color_classes = []
        uncolored = candidates
        while uncolored:
            # Pick a vertex from uncolored
            # Use lowest set bit
            v = (uncolored & (-uncolored)).bit_length() - 1
            color_class = 1 << v
            uncolored &= ~(1 << v)
            # Greedily add vertices to this color class that are not adjacent to any in it
            while uncolored:
                # Check if any uncolored vertex is non-adjacent to all in color_class
                # For efficiency, compute common non-neighbors
                # A vertex u can be added if adj[u] & color_class == 0
                # Iterate through uncolored
                u = (uncolored & (-uncolored)).bit_length() - 1
                if adj_bits[u] & color_class == 0:
                    color_class |= (1 << u)
                    uncolored &= ~(1 << u)
                else:
                    uncolored &= ~(1 << u)
            color_classes.append(color_class)
        return color_classes

    def tomita_mis(candidates, solution, depth):
        """Branch and bound MIS using Tomita's algorithm."""
        nonlocal best_size, best_solution

        if time.time() > deadline - 5:
            return

        if candidates == 0:
            if solution.bit_count() > best_size:
                best_size = solution.bit_count()
                best_solution = solution
            return

        # Upper bound: number of color classes
        color_classes = greedy_coloring(candidates)
        upper_bound = len(color_classes)

        if solution.bit_count() + upper_bound <= best_size:
            return

        # Process color classes in reverse order (largest first often works better)
        for cc in reversed(color_classes):
            if time.time() > deadline - 5:
                return

            # Branch: include vertices from this color class
            # Since cc is an independent set, we can pick any subset
            # Tomita: for each vertex v in cc, branch on including v
            for v in range(n_total):
                if not (cc & (1 << v)):
                    continue

                # Include v
                new_solution = solution | (1 << v)
                # New candidates: candidates that are not v and not adjacent to v
                new_candidates = candidates & ~(1 << v) & ~adj_bits[v]

                tomita_mis(new_candidates, new_solution, depth + 1)

                # Update best
                if new_solution.bit_count() > best_size:
                    best_size = new_solution.bit_count()
                    best_solution = new_solution

                # Remove v from candidates for subsequent branches
                candidates &= ~(1 << v)

                # Prune: if remaining candidates + current solution size <= best, break
                if candidates.bit_count() + new_solution.bit_count() - 1 <= best_size:
                    break

    # Initialize with all vertices
    all_bits = (1 << n_total) - 1
    tomita_mis(all_bits, 0, 0)

    # Extract best solution
    if best_size > 0:
        result_indices = []
        for i in range(n_total):
            if best_solution & (1 << i):
                result_indices.append(i)
        result_vectors = [all_vecs[i] for i in result_indices]
        return np.array(result_vectors, dtype=np.int64)
    else:
        # Fallback: return weight-2 vectors only (they form a valid set with dot <= 0)
        return np.array(w2_vecs, dtype=np.int64)

# EVOLVE-BLOCK-END

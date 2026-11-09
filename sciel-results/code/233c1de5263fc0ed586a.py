"""Construct a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Try to find a valid graph with n=21 first, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline - 10:
            break
        result = _find_valid_graph(target_n, deadline)
        if result is not None:
            return result

    # Fallback: return a small valid graph (n=5 is always valid)
    return {'n': 5, 'edges': []}


def _find_valid_graph(n, deadline):
    """Find a valid graph on n vertices using random trials with fast verification."""
    # Try multiple edge probabilities and seeds
    p_values = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.52, 0.48]

    # Use more seeds for smaller n where success is more likely
    max_trials = 200 if n <= 20 else 100

    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break

        p = p_values[trial % len(p_values)]
        seed = trial * 2654435761 % (2**32)
        rng = random.Random(seed)

        # Build adjacency as list of integers (bitmasks) for fast subset checks
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)

        if _is_valid_fast(adj_mask, n, deadline):
            edges = _masks_to_edges(adj_mask, n)
            return {'n': n, 'edges': edges}

    return None


def _is_valid_fast(adj_mask, n, deadline):
    """
    Check if no subset of 6+ vertices induces a regular subgraph.
    Uses bitmask operations for fast degree computation within subsets.
    """
    if n < 6:
        return True

    # Check all subset sizes from 6 to n
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            # Time check
            if time.time() > deadline:
                return False

            # Build mask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)

            # Compute degrees within the induced subgraph
            first_deg = (adj_mask[subset[0]] & mask).bit_count()
            for v in subset[1:]:
                if (adj_mask[v] & mask).bit_count() != first_deg:
                    break
            else:
                # All degrees equal -> regular induced subgraph found
                return False

    return True


def _masks_to_edges(adj_mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                edges.append([i, j])
    return edges

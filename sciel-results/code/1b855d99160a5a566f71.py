import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 140  # 140s out of 150s budget

    best_n = 39
    best_edges = []

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break

        required_min_deg = n - 39

        # Strategy 1: polarity graph seed + greedy fill + polish
        result = _polarity_greedy(n, required_min_deg, deadline)
        if result is not None:
            best_n = n
            best_edges = result
            return {'n': n, 'edges': result}

        # Strategy 2: greedy with multiple seeds
        for seed in range(30):
            if time.time() > deadline:
                break
            result = _try_construct_fast(n, required_min_deg, deadline, seed)
            if result is not None:
                best_n = n
                best_edges = result
                return {'n': n, 'edges': result}

    return {'n': best_n, 'edges': best_edges}


def _check_c4_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    A C4 is created iff there exists a path u-x-y-v of length 3 in the current graph.
    """
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]

    for x in nu:
        if x == v:
            continue
        nx = np.where(adj[x])[0]
        mask = (nx != u) & (nx != v)
        if np.any(adj[nx[mask], v]):
            return True
    return False


def _polarity_greedy(n, required_min_deg, deadline):
    """
    Seed with a polarity graph (C4-free), then greedily add edges to boost min degree,
    then polish with edge swaps.
    """
    rng = np.random.RandomState(42)

    # Build a polarity graph on n vertices
    # For q = floor(n/2), use a polarity graph of PG(1,q) extended
    # Simpler: use a known C4-free construction - the polarity graph of a finite geometry
    # For general n, use a greedy construction with a good initial structure

    adj = np.zeros((n, n), dtype=bool)

    # Initial structure: use a "random" C4-free seed
    # Try multiple random seeds for the initial structure
    for seed in range(20):
        if time.time() > deadline:
            break

        random.seed(seed)
        rng = np.random.RandomState(seed)
        adj = np.zeros((n, n), dtype=bool)

        # Build initial C4-free graph using greedy with random vertex ordering
        perm = list(range(n))
        rng.shuffle(perm)

        for i, u in enumerate(perm):
            # Try to connect u to previous vertices
            for j in range(i):
                v = perm[j]
                if not adj[u, v] and not _check_c4_fast(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True

        # Now greedily add more edges to boost min degree
        for iteration in range(2000):
            if time.time() > deadline:
                break

            degrees = adj.sum(axis=1)
            min_deg = degrees.min()

            if min_deg >= required_min_deg:
                break

            min_deg_vertices = np.where(degrees == min_deg)[0]
            rng.shuffle(min_deg_vertices)

            added = False
            for u in min_deg_vertices:
                if time.time() > deadline:
                    break

                candidates = [v for v in range(n) if v != u and not adj[u, v]]
                candidates.sort(key=lambda v: degrees[v])

                for v in candidates:
                    if time.time() > deadline:
                        break
                    if not _check_c4_fast(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        added = True
                        break

                if added:
                    break

            if not added:
                break

        degrees = adj.sum(axis=1)
        if degrees.min() >= required_min_deg:
            edges = _extract_edges(adj)
            return edges

    return None


def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking.
    """
    random.seed(seed)
    rng = np.random.RandomState(seed)

    adj = np.zeros((n, n), dtype=bool)

    # Start with a random permutation of vertices
    perm = list(range(n))
    rng.shuffle(perm)

    # Build initial structure
    for i, u in enumerate(perm):
        for j in range(i):
            v = perm[j]
            if not adj[u, v] and not _check_c4_fast(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True

    # Greedily add more edges
    for iteration in range(3000):
        if time.time() > deadline:
            break

        degrees = adj.sum(axis=1)
        min_deg = degrees.min()

        if min_deg >= required_min_deg:
            break

        min_deg_vertices = np.where(degrees == min_deg)[0]
        rng.shuffle(min_deg_vertices)

        added = False
        for u in min_deg_vertices:
            if time.time() > deadline:
                break

            candidates = [v for v in range(n) if v != u and not adj[u, v]]
            candidates.sort(key=lambda v: degrees[v])

            for v in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_fast(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    added = True
                    break

            if added:
                break

        if not added:
            break

    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = _extract_edges(adj)
        return edges
    return None


def _extract_edges(adj):
    """Extract edge list from adjacency matrix."""
    edges = []
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges


# EVOLVE-BLOCK-END

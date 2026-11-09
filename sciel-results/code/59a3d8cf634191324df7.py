"""Construct a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Strategy: try algebraic constructions (Paley, circulant) then random search
    # Paley graphs on q vertices where q is prime power ≡ 1 (mod 4)
    # These have strong structural properties that may avoid regular induced subgraphs

    best_result = None

    # Try Paley graphs for various q values
    for q in [21, 20, 19, 17, 13]:
        if time.time() > deadline - 30:
            break
        if q <= 5:
            continue

        # For q=21: not a prime power, but we can try circulant or other constructions
        # For prime powers q ≡ 1 mod 4: Paley graph
        # q=17, 13 are prime and ≡ 1 mod 4
        # q=21 is not prime power, try other approaches

        if q in [17, 13]:
            result = _paley_graph(q, deadline)
            if result is not None:
                return result

        # Try circulant graphs
        if q in [21, 20, 19]:
            result = _try_circulant(q, deadline)
            if result is not None:
                return result

    # Try Paley for q=17 (prime, 17≡1 mod 4)
    if time.time() < deadline - 25:
        result = _paley_graph(17, deadline)
        if result is not None:
            return result

    # Try random search with fast verification for n=20, 21
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
        result = _random_search_fast(n, deadline)
        if result is not None:
            return result

    # Try Paley 13 as fallback
    if time.time() < deadline - 5:
        result = _paley_graph(13, deadline)
        if result is not None:
            return result

    # Final fallback: small valid graph
    return {'n': 5, 'edges': []}


def _paley_graph(q, deadline):
    """
    Construct Paley graph on q vertices (q must be prime power ≡ 1 mod 4).
    Vertex i is connected to j iff (j-i) mod q is a quadratic residue.
    """
    if time.time() > deadline:
        return None

    # Check if q is a prime power ≡ 1 mod 4
    # For simplicity, handle small primes: 5, 13, 17, 29, 37, ...
    # q=5: 5≡1 mod 4 ✓
    # q=13: 13≡1 mod 4 ✓
    # q=17: 17≡1 mod 4 ✓
    # q=29: 29≡1 mod 4 ✓
    # q=37: 37≡1 mod 4 ✓

    if q % 4 != 1:
        return None

    # Compute quadratic residues mod q
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)

    # Build adjacency
    adj = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                adj[i, j] = 1
                adj[j, i] = 1

    # Verify with fast method
    if _verify_fast(adj, q, deadline):
        edges = _adj_to_edges(adj, q)
        return {'n': q, 'edges': edges}

    return None


def _try_circulant(n, deadline):
    """
    Try circulant graphs: vertex i connected to i±d for d in some set.
    Search over different connection sets.
    """
    if time.time() > deadline:
        return None

    # For circulant graphs, we choose a set S of distances
    # Vertex i connects to i±d for each d in S
    # Try various sets

    candidates = []

    # Generate candidate distance sets
    if n == 21:
        # Try various sets of distances for n=21
        candidate_sets = [
            [1, 2, 4, 5, 8, 10],
            [1, 2, 3, 5, 7, 9],
            [1, 3, 4, 6, 8, 10],
            [2, 3, 5, 7, 9, 10],
            [1, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 10],
            [1, 2, 3, 4, 7, 10],
            [1, 2, 4, 5, 8, 9],
            [2, 4, 5, 7, 8, 10],
            [1, 3, 4, 5, 8, 10],
        ]
    elif n == 20:
        candidate_sets = [
            [1, 2, 4, 6, 7],
            [1, 3, 4, 6, 8],
            [1, 2, 5, 6, 8],
            [2, 3, 5, 7, 8],
            [1, 3, 5, 6, 8],
            [1, 2, 4, 5, 7],
            [1, 3, 4, 5, 7],
            [2, 3, 4, 6, 8],
            [1, 2, 3, 5, 7],
            [1, 4, 5, 6, 7],
        ]
    elif n == 19:
        candidate_sets = [
            [1, 2, 4, 5, 8, 9],
            [1, 3, 4, 6, 8, 9],
            [2, 3, 5, 7, 8, 9],
            [1, 2, 3, 5, 7, 9],
            [1, 3, 5, 6, 8, 9],
        ]
    else:
        return None

    for dset in candidate_sets:
        if time.time() > deadline - 5:
            break

        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for d in dset:
                j1 = (i + d) % n
                j2 = (i - d) % n
                if j1 != i:
                    adj[i, j1] = 1
                if j2 != i:
                    adj[i, j2] = 1

        # Ensure symmetry (should already be)
        adj = np.maximum(adj, adj.T)

        if _verify_fast(adj, n, deadline):
            edges = _adj_to_edges(adj, n)
            return {'n': n, 'edges': edges}

    return None


def _verify_fast(adj, n, deadline):
    """
    Fast verification: check all subsets of size 6 first (most numerous, most likely to violate).
    If any size-6 subset is regular, return False.
    Then check larger subsets.
    """
    if n < 6:
        return True

    # Precompute: for each subset of size 6, check if regular
    # Use numpy for speed
    vertices = np.arange(n)

    # Check size-6 subsets
    for subset in combinations(range(n), 6):
        if time.time() > deadline:
            return False

        # Extract sub-adjacency
        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        # Check if all degrees equal
        if len(set(degrees.tolist())) == 1:
            return False

    # Check size-7 subsets
    for subset in combinations(range(n), 7):
        if time.time() > deadline:
            return False

        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        if len(set(degrees.tolist())) == 1:
            return False

    # Check size-8 subsets
    for subset in combinations(range(n), 8):
        if time.time() > deadline:
            return False

        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        if len(set(degrees.tolist())) == 1:
            return False

    # Check size-9 subsets
    for subset in combinations(range(n), 9):
        if time.time() > deadline:
            return False

        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        if len(set(degrees.tolist())) == 1:
            return False

    # Check size-10 subsets
    for subset in combinations(range(n), 10):
        if time.time() > deadline:
            return False

        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        if len(set(degrees.tolist())) == 1:
            return False

    # For larger subsets, the probability of regularity is lower
    # but we still need to check
    for size in range(11, n + 1):
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                return False

            sub_adj = adj[np.ix_(subset, subset)]
            degrees = sub_adj.sum(axis=1)

            if len(set(degrees.tolist())) == 1:
                return False

    return True


def _adj_to_edges(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges


def _random_search_fast(n, deadline):
    """
    Random search with fast verification.
    Try many random graphs, checking only size-6 subsets first for speed.
    """
    if time.time() > deadline - 5:
        return None

    max_trials = 200
    rng = random.Random(42)

    for trial in range(max_trials):
        if time.time() > deadline - 3:
            break

        # Vary edge probability
        p = 0.3 + 0.4 * (trial % 10) / 10.0  # p from 0.3 to 0.69

        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # Quick check: size-6 subsets only
        if _verify_size6_fast(adj, n, deadline):
            # Full verification
            if _verify_fast(adj, n, deadline):
                edges = _adj_to_edges(adj, n)
                return {'n': n, 'edges': edges}

    return None


def _verify_size6_fast(adj, n, deadline):
    """Quick check: only size-6 subsets."""
    if n < 6:
        return True

    for subset in combinations(range(n), 6):
        if time.time() > deadline:
            return False

        sub_adj = adj[np.ix_(subset, subset)]
        degrees = sub_adj.sum(axis=1)

        if len(set(degrees.tolist())) == 1:
            return False

    return True

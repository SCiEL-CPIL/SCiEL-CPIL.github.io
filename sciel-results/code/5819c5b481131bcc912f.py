"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 110  # 10s safety margin before 120s kill

def _has_regular_induced(G, n):
    """Check if G (as adjacency matrix) has any regular induced subgraph on >= 6 vertices."""
    # For each subset of size >= 6, check if induced subgraph is regular
    # Optimization: check sizes 6, 7, ..., n
    # A regular subgraph on k vertices means every vertex in the subset has the same degree within the subset
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            # Compute degrees within subset
            idx = list(subset)
            idx_set = set(idx)
            degs = []
            for v in idx:
                d = 0
                for u in idx:
                    if u != v and G[v][u]:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                return True
    return False

def _has_regular_induced_fast(G, n):
    """Faster check using bitmasks for n <= 22."""
    # G is a list of ints where bit j is set if edge (i,j) exists
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            idx = list(subset)
            degs = []
            for v in idx:
                d = 0
                for u in idx:
                    if u != v and (G[v] >> u) & 1:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                return True
    return False

def _try_local_search(n, seed, time_limit):
    """Randomized local search: start with random graph, flip edges to avoid regular induced subgraphs."""
    random.seed(seed)
    deadline = time.time() + time_limit
    
    # Start with random graph
    G = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
    
    # Check initial
    if not _has_regular_induced_fast(G, n):
        return G
    
    # Local search: find a bad subset, flip an edge in it
    for _ in range(100000):
        if time.time() > deadline:
            break
        # Find a bad subset
        bad_found = False
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                idx = list(subset)
                degs = []
                for v in idx:
                    d = 0
                    for u in idx:
                        if u != v and (G[v] >> u) & 1:
                            d += 1
                    degs.append(d)
                if len(set(degs)) == 1:
                    # Found bad subset - flip a random edge within it
                    pair = random.choice(list(combinations(idx, 2)))
                    u, v = pair
                    if (G[u] >> v) & 1:
                        G[u] &= ~(1 << v)
                        G[v] &= ~(1 << u)
                    else:
                        G[u] |= (1 << v)
                        G[v] |= (1 << u)
                    bad_found = True
                    break
            if bad_found:
                break
        if not bad_found:
            return G
    return None

def _to_edges(G, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (G[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    best_n = 5
    best_G = [0] * 5  # empty graph on 5 vertices
    
    # Try to find larger graphs
    for n in range(20, 4, -1):  # Try from 20 down to 5
        if time.time() - start > 100:
            break
        # Try multiple seeds
        for seed in range(20):
            if time.time() - start > 105:
                break
            G = _try_local_search(n, seed, 4.0)
            if G is not None:
                best_n = n
                best_G = G
                break
        if best_n >= n:
            break
    
    return {'n': best_n, 'edges': _to_edges(best_G, best_n)}
# EVOLVE-BLOCK-END

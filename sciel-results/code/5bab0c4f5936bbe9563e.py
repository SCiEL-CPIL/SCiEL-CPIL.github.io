"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    def generate_weight_k(k):
        """Generate all integer vectors with exactly k nonzero entries, each ±1."""
        vectors = []
        for coords in combinations(range(D), k):
            for signs in _signs_iter(k):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(coords, signs):
                    v[idx] = s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    def _signs_iter(k):
        """Generate all 2^k sign combinations."""
        for mask in range(1 << k):
            yield tuple(1 if (mask >> i) & 1 else -1 for i in range(k))

    def min_degree_greedy(n, adj):
        """
        Minimum Degree Greedy for Maximum Independent Set.
        adj: list of sets, adj[i] = set of neighbors of i.
        Returns list of selected vertex indices.
        """
        selected = []
        remaining = set(range(n))
        # Track degrees
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining:
            # Find vertex with minimum degree among remaining
            min_deg = min(deg[i] for i in remaining)
            candidates = [i for i in remaining if deg[i] == min_deg]
            # Pick the first one (or random for diversity)
            v = candidates[0]
            selected.append(v)
            remaining.discard(v)
            # Remove neighbors from remaining and update degrees
            for u in list(adj[v]):
                if u in remaining:
                    remaining.discard(u)
                    for w in adj[u]:
                        if w in remaining:
                            deg[w] -= 1
        return selected

    # Baseline: all weight-2 vectors (220)
    w2 = generate_weight_k(2)
    best_result = w2
    best_count = len(w2)

    if time.time() > deadline:
        return best_result

    # Try weight-3 vectors with Minimum Degree Greedy
    w3 = generate_weight_k(3)
    n3 = len(w3)
    
    # For weight-3 vectors, norm² = 3. Need distance² >= 3.
    # distance² = 3+3-2*dot = 6-2*dot >= 3 => dot <= 1.5 => dot <= 1
    # Conflict if dot > 1, i.e., dot >= 2
    
    # Compute dot products efficiently
    # w3 @ w3.T gives all pairwise dot products
    dots = w3 @ w3.T
    # Conflict if dots[i,j] >= 2 (and i != j)
    conflict_mask = (dots >= 2)
    np.fill_diagonal(conflict_mask, False)
    
    # Build adjacency list
    adj = [set() for _ in range(n3)]
    conflict_indices = np.argwhere(conflict_mask)
    for i, j in conflict_indices:
        adj[i].add(j)
        adj[j].add(i)
    
    # Use Minimum Degree Greedy
    selected = min_degree_greedy(n3, adj)
    result_w3 = w3[selected]
    
    if len(selected) > best_count:
        best_result = result_w3
        best_count = len(selected)
    
    if time.time() > deadline:
        return best_result

    # Try weight-4 vectors with Minimum Degree Greedy
    # norm² = 4, need distance² >= 4
    # distance² = 4+4-2*dot = 8-2*dot >= 4 => dot <= 2
    # Conflict if dot > 2, i.e., dot >= 3
    
    w4 = generate_weight_k(4)
    n4 = len(w4)
    
    if n4 < 100000:  # safety check
        dots4 = w4 @ w4.T
        conflict_mask4 = (dots4 >= 3)
        np.fill_diagonal(conflict_mask4, False)
        
        adj4 = [set() for _ in range(n4)]
        conflict_indices4 = np.argwhere(conflict_mask4)
        for i, j in conflict_indices4:
            adj4[i].add(j)
            adj4[j].add(i)
        
        selected4 = min_degree_greedy(n4, adj4)
        result_w4 = w4[selected4]
        
        if len(selected4) > best_count:
            best_result = result_w4
            best_count = len(selected4)
    
    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END

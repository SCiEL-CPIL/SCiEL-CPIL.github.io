import time
import random
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
    
    # For each node, store which other nodes form bad triples with it (as pairs)
    # pairs_by_node[i] = set of (j, k) such that (i, j, k) is a bad triple
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
    
    # Build adjacency bitmask: adj[i] has bit j set if i and j are in some bad triple
    adj = [0] * n
    for a, b, c in bad_triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Degree of each node in the conflict graph
    degree = [bin(adj[i]).count('1') for i in range(n)]
    
    def can_add_to_mask(selected_mask, node):
        """Check if adding node to selected_mask doesn't complete any bad triple."""
        for other1, other2 in pairs_by_node[node]:
            if (selected_mask & (1 << other1)) and (selected_mask & (1 << other2)):
                return False
        return True
    
    def greedy_construction(order):
        """Greedy construction following a given order."""
        selected_mask = 0
        for node in order:
            if can_add_to_mask(selected_mask, node):
                selected_mask |= (1 << node)
        return selected_mask
    
    def local_search(mask):
        """Iterative local search: add-one and 1-swap moves."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Add-one moves
            for node in range(n):
                if time.time() > deadline - 10:
                    return mask
                if not (mask & (1 << node)):
                    if can_add_to_mask(mask, node):
                        mask |= (1 << node)
                        improved = True
            # 1-swap moves: remove one, add one (if net gain)
            if time.time() > deadline - 10:
                return mask
            for remove_node in range(n):
                if time.time() > deadline - 10:
                    return mask
                if (mask & (1 << remove_node)):
                    new_mask = mask & ~(1 << remove_node)
                    for add_node in range(n):
                        if time.time() > deadline - 10:
                            return mask
                        if not (new_mask & (1 << add_node)):
                            if can_add_to_mask(new_mask, add_node):
                                new_mask |= (1 << add_node)
                                sz_old = bin(mask).count('1')
                                sz_new = bin(new_mask).count('1')
                                if sz_new > sz_old:
                                    mask = new_mask
                                    improved = True
                                    break
        return mask
    
    # Initial greedy solutions with multiple orderings
    orderings = [
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: degree[i], reverse=True),
        list(range(n)),
        list(range(n - 1, -1, -1)),
    ]
    
    best_mask = 0
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        mask = greedy_construction(order)
        mask = local_search(mask)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Multi-restart randomized greedy with local search
    rng = random.Random(42)
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Create a random ordering with some structure
        # Mix of random and degree-based
        if restarts % 3 == 0:
            # Pure random
            order = list(range(n))
            rng.shuffle(order)
        elif restarts % 3 == 1:
            # Degree-sorted with random perturbation
            order = sorted(range(n), key=lambda i: degree[i] + rng.uniform(-5, 5))
        else:
            # Reverse degree-sorted with random perturbation
            order = sorted(range(n), key=lambda i: -degree[i] + rng.uniform(-5, 5))
        
        mask = greedy_construction(order)
        mask = local_search(mask)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final local search on best solution
    if time.time() < deadline - 10:
        best_mask = local_search(best_mask)
        best_size = bin(best_mask).count('1')
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]

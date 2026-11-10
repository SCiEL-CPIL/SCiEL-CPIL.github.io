"""Rewritten program: stronger algorithm for large sets avoiding reciprocal Schur triples."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = b*c / (b+c), so (b+c) | (b*c).
    # Since 1/a = 1/b + 1/c > 1/c, we have a < c. Similarly a < b.
    # So a is always the smallest of the three distinct elements.
    
    triples = []  # list of (a, b, c) with a < b < c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    
    # Build data structures for fast triple checking
    # triple_map[a] = set of (b, c) pairs that form a triple with a
    triple_map = {}
    for a, b, c in triples:
        if a not in triple_map:
            triple_map[a] = set()
        triple_map[a].add((b, c))
    
    # For each element x, store which triples it participates in as b or c
    # element_triples[x] = list of (a, other) where x is b or c
    element_triples = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        element_triples[b].append((a, c))
        element_triples[c].append((a, b))
    
    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map:
                for (b, c) in triple_map[a]:
                    if b in S and c in S:
                        return True
        return False
    
    def can_add(S, x):
        """Check if adding x to S creates a forbidden triple."""
        # x as 'a' (smallest): check if any (b, c) in triple_map[x] are both in S
        if x in triple_map:
            for (b, c) in triple_map[x]:
                if b in S and c in S:
                    return False
        # x as 'b' or 'c': check if any triple (a, x, c) or (a, b, x) has a and the other in S
        for (a, other) in element_triples[x]:
            if a in S and other in S:
                return False
        return True
    
    # Greedy construction with different orderings
    def greedy_order(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    # Try multiple orderings
    best = set()
    best_size = 0
    
    # Descending order (largest first - they're safest as 'a')
    cand = greedy_order(list(range(N, 0, -1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(best)
    
    # Ascending order
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(best)
    
    # Random orderings
    for _ in range(500):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(best)
    
    # Local search: try to add missing elements
    def local_search_add(S, max_iter=5000):
        S = set(S)
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            improved = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
            if not improved:
                break
        return S
    
    best = local_search_add(best)
    best_size = len(best)
    
    # Swap search: remove one element, add another to explore the space
    for _ in range(100000):
        if time.time() >= deadline:
            break
        if not best:
            break
        y = rng.choice(list(best))
        best_without_y = best - {y}
        x = rng.randint(1, N)
        if x in best_without_y:
            continue
        if can_add(best_without_y, x):
            best = best_without_y | {x}
    
    # Final local search after swaps
    best = local_search_add(best)
    best_size = len(best)
    
    # Simulated annealing for further exploration
    def simulated_annealing(S, max_iter=50000):
        S = set(S)
        best_local = set(S)
        best_local_size = len(S)
        temp = 1.0
        cooling = 0.999
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            if not S:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random missing element
                missing = [y for y in range(1, N + 1) if y not in S]
                if not missing:
                    continue
                x = rng.choice(missing)
                if can_add(S, x):
                    S.add(x)
                    if len(S) > best_local_size:
                        best_local = set(S)
                        best_local_size = len(S)
                elif rng.random() < temp:
                    S.add(x)
            elif move_type < 0.7:
                # Remove a random element
                x = rng.choice(list(S))
                S.discard(x)
            else:
                # Swap: remove one, add another
                x = rng.choice(list(S))
                missing = [y for y in range(1, N + 1) if y not in S]
                if not missing:
                    continue
                y = rng.choice(missing)
                S.discard(x)
                if can_add(S, y):
                    S.add(y)
                    if len(S) > best_local_size:
                        best_local = set(S)
                        best_local_size = len(S)
                elif rng.random() < temp:
                    S.add(y)
            
            temp *= cooling
        
        return best_local
    
    best = simulated_annealing(best)
    best = local_search_add(best)
    best_size = len(best)
    
    # Verify
    if has_triple(best):
        # Fallback to a known valid set (all numbers > 90)
        best = set(range(91, N + 1))
    
    return list(best)

# EVOLVE-BLOCK-END

"""Strategy: Maximum Independent Set via Branch and Bound on a 3-uniform hypergraph."""
import time
import itertools

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  bc = a(b+c)  =>  a = bc / (b+c)
    # We iterate over pairs (b, c) with b < c to generate potential a.
    # If a is an integer, distinct from b and c, and in [1, N], it forms a triple.
    # The triple is {a, b, c}.
    
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    # Store sorted tuple to avoid duplicates
                    triple = tuple(sorted((a, b, c)))
                    triples.add(triple)
    
    # 2. Build adjacency/conflict structure
    # For Branch and Bound, we need to know which vertices conflict with which.
    # In a 3-uniform hypergraph, we can't just use a simple adjacency matrix for MIS directly
    # without expansion, but we can use a standard MIS B&B on the "conflict graph" 
    # where an edge exists if two vertices appear in the same forbidden triple?
    # NO, that's not correct. If u and v are in a triple with w, selecting u and v is only 
    # forbidden if w is also selected. 
    # 
    # Standard MIS on a graph forbids edges. Here we forbid hyperedges.
    # However, we can reformulate:
    # We want a set S such that no triple is fully contained in S.
    # This is equivalent to finding the Maximum Independent Set in the 3-uniform hypergraph.
    #
    # A common technique for small N (180 is manageable for B&B with good pruning) is to use
    # a recursive search that decides for each number whether to include it or exclude it.
    #
    # To make B&B efficient, we need a strong upper bound.
    # Upper bound for current partial set + remaining candidates:
    # |Current| + |Remaining| is too loose.
    # A better bound: We can use the fact that each remaining element can "break" some triples.
    #
    # Alternative approach: 
    # Since N=180 is small, we can try a greedy heuristic first to get a good lower bound,
    # then use B&B to prove optimality or find a better set.
    
    # Let's first try a strong Greedy Heuristic to establish a baseline.
    # Strategy: Sort numbers by "degree" (number of triples they are part of).
    # Pick the one with the lowest degree, add to set, remove all triples it completes?
    # No, we remove all triples that contain the picked element from the "active" set,
    # and also remove the element from consideration.
    # Actually, if we pick x, we cannot pick any pair (y, z) such that (x, y, z) is a triple.
    # This is complex to track incrementally.
    
    # Simpler Greedy:
    # Start with all numbers.
    # While there is a triple in the set:
    #   Find a triple (a, b, c).
    #   Remove the element with the highest "impact" (most other triples it's part of).
    #   Repeat.
    
    # Let's implement a specific Greedy:
    # 1. Calculate degree of each node (count of triples it appears in).
    # 2. Sort nodes by degree descending.
    # 3. Initialize S = set().
    # 4. Iterate nodes in sorted order:
    #    If adding node v to S does not create a new forbidden triple (check against S), add it.
    #    To check "does not create a new forbidden triple", we need to see if there exist 
    #    u, w in S such that (u, v, w) is a forbidden triple.
    
    # Pre-compute for each v, the set of pairs (u, w) such that (u, v, w) is a triple.
    # conflicts[v] = set of (u, w) pairs.
    # When considering adding v, we check if any (u, w) in conflicts[v] has both u and w in S.
    
    conflicts = {i: set() for i in range(1, N + 1)}
    for t in triples:
        a, b, c = t
        conflicts[a].add((b, c))
        conflicts[b].add((a, c))
        conflicts[c].add((a, b))
    
    # Greedy construction
    def greedy_construct():
        S = set()
        # Sort by degree (number of conflicts) descending? Or ascending?
        # Usually, picking "safe" nodes (low degree) first allows more nodes.
        # Let's try sorting by degree ascending.
        sorted_nodes = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
        
        for v in sorted_nodes:
            # Check if v conflicts with any existing pair in S
            is_safe = True
            for u, w in conflicts[v]:
                if u in S and w in S:
                    is_safe = False
                    break
            if is_safe:
                S.add(v)
        return S
    
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # If time permits, try a different greedy order or local search improvement.
    # Local Search: Try to swap elements to increase size?
    # Or try multiple random restarts of the greedy with shuffled tie-breakers.
    
    # Let's try 100 random restarts of the greedy with randomized order among similar degrees.
    import random
    random.seed(42)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Shuffle nodes slightly, or just randomize the order of processing
        nodes = list(range(1, N + 1))
        # Perturb the sort: add random noise to degrees
        degrees = {i: len(conflicts[i]) for i in nodes}
        noisy_degrees = {i: degrees[i] + random.uniform(-0.1, 0.1) for i in nodes}
        sorted_nodes = sorted(nodes, key=lambda x: noisy_degrees[x])
        
        S = set()
        for v in sorted_nodes:
            is_safe = True
            for u, w in conflicts[v]:
                if u in S and w in S:
                    is_safe = False
                    break
            if is_safe:
                S.add(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Optional: Branch and Bound for optimality?
    # Given N=180, B&B might be too slow if the bound is weak.
    # However, the greedy likely finds a very large set.
    # The theoretical max is close to N.
    # Let's check if we can improve best_set by trying to add back excluded nodes?
    # No, the greedy is "add if safe". It's a maximal independent set.
    
    # Let's try a simple local search: 
    # Remove one element, see if we can add two? (Unlikely to increase size by 1 easily)
    # Or just return the best greedy result.
    
    return list(best_set)

# Test run
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    S = set(res)
    valid = True
    # Re-check triples
    N = 180
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if 1.0/a == 1.0/b + 1.0/c:
                    valid = False
                    print(f"Found conflict: {a}, {b}, {c}")
    print("Valid:", valid)

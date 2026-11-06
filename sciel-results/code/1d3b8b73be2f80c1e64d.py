import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin

    # 255 = 3 * 5 * 17
    # The set must contain 255.
    # Any other element x must have gcd(x, 255) > 1, so x is divisible by 3, 5, or 17.
    # Let C be the set of such numbers in [1, 255] excluding 255.
    # We want to find the maximum clique in the graph G where vertices are C
    # and edges connect numbers with gcd > 1.
    # The final answer is this clique plus {255}.

    candidates = []
    for i in range(1, 256):
        if i == 255:
            continue
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)

    n = len(candidates)

    # Build adjacency list (sets)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)

    # Bron-Kerbosch with pivot
    # We use a non-recursive implementation or careful recursion to avoid stack limits,
    # though n=127 is small enough for recursion.
    # However, Python recursion overhead can be significant.
    # Let's try a standard recursive BK with pivot, optimized with bitmasks if possible,
    # or sets if simpler. Sets are fine for n=127.

    best_clique = []
    max_size = 0

    # Initial lower bound: all multiples of 3
    # This is a valid clique.
    mult3_indices = [i for i in range(n) if candidates[i] % 3 == 0]
    max_size = len(mult3_indices)
    best_clique = mult3_indices[:]

    # We can also check multiples of 5 and 17, but 3 is the largest group.
    # Multiples of 3: 85
    # Multiples of 5: 51
    # Multiples of 17: 15

    def bron_kerbosch(R, P, X):
        nonlocal max_size, best_clique

        if time.time() > deadline:
            return

        if not P and not X:
            if len(R) > max_size:
                max_size = len(R)
                best_clique = R[:]
            return

        if not P:
            # Maximal clique found
            if len(R) > max_size:
                max_size = len(R)
                best_clique = R[:]
            return

        # Choose pivot u from P ∪ X to minimize branching
        # We want u such that |P ∩ N(u)| is maximized.
        # Then we only need to branch on P \ N(u).

        # To find the best pivot efficiently:
        # Iterate through P ∪ X.
        # Since X is usually small or P is shrinking, we can just iterate P first,
        # then X if needed. Or just P is often good enough.
        # Let's try to find u in P with max degree in P.
        
        u = -1
        max_deg = -1
        # Optimization: If P is large, iterating all P to find max degree is O(|P|^2) due to set intersection.
        # We can approximate or just pick the first one? No, pivot is crucial for performance.
        # For n=127, O(n^2) per node is fine.
        
        for v in P:
            deg = len(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                u = v
        
        # If u is still -1 (shouldn't happen if P is not empty), pick first
        if u == -1:
            u = next(iter(P))

        # Branch on P \ N(u)
        # We must iterate over a copy of the list because we modify P
        candidates_to_visit = list(P - adj[u])
        
        # Sort candidates? Maybe by degree to find large cliques earlier?
        # Not strictly necessary for correctness, but might find better clique sooner.
        # Let's sort by degree in P descending.
        # This might help prune earlier if we find a large clique.
        candidates_to_visit.sort(key=lambda x: len(P & adj[x]), reverse=True)

        for v in candidates_to_visit:
            if time.time() > deadline:
                return
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(R | {v}, new_P, new_X)
            
            # Remove v from P and add to X
            P.remove(v)
            X.add(v)

    # Start the search
    # R = empty
    # P = all nodes
    # X = empty
    bron_kerbosch(set(), set(range(n)), set())

    # Map back to values
    result = [candidates[i] for i in best_clique]
    result.append(255)

    return result

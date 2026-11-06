import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)

    # Build adjacency: edge if gcd > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)

    # Greedy maximum clique heuristic
    def greedy_clique(order):
        clique = []
        for v in order:
            if all(u in adj[v] for u in clique):
                clique.append(v)
        return clique

    # Known good: multiples of 3
    best_clique = [i for i, x in enumerate(candidates) if x % 3 == 0]
    best_size = len(best_clique)

    # Try greedy by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order_desc = sorted(range(n), key=lambda i: -degrees[i])
    clique = greedy_clique(order_desc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)

    # Try greedy by degree ascending
    order_asc = sorted(range(n), key=lambda i: degrees[i])
    clique = greedy_clique(order_asc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)

    # Systematic local search: try adding each non-member, and try all single swaps
    if time.time() < deadline - 5:
        current = set(best_clique)

        # Phase 1: Try to add every possible vertex
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if v not in current and all(u in adj[v] for u in current):
                    current.add(v)
                    changed = True
            best_clique = list(current)
            best_size = len(current)

        # Phase 2: Try all single swaps (remove one, add one)
        random.seed(42)
        for iteration in range(20000):
            if time.time() > deadline:
                break

            improved = False
            # Try to add a random vertex
            not_in = [i for i in range(n) if i not in current]
            if not_in:
                v = random.choice(not_in)
                if all(u in adj[v] for u in current):
                    current.add(v)
                    best_clique = list(current)
                    best_size = len(current)
                    improved = True
                    continue

            # Try swap: remove one, add another
            if len(current) > 1:
                to_remove = random.choice(list(current))
                candidate_new = current - {to_remove}
                not_in = [i for i in range(n) if i not in candidate_new]
                if not_in:
                    v = random.choice(not_in)
                    if all(u in adj[v] for u in candidate_new):
                        current = candidate_new | {v}
                        if len(current) > best_size:
                            best_clique = list(current)
                            best_size = len(current)
                        improved = True
                        continue

            if not improved:
                break

    result = [candidates[i] for i in best_clique]

    # Ensure 255 is in the result
    if 255 not in result:
        idx_255 = candidates.index(255)
        # Check if 255 is compatible with all current elements
        if all(idx_255 in adj[candidates.index(x)] for x in result):
            result.append(255)
        else:
            # Replace an incompatible element with 255
            for i, x in enumerate(result):
                if x != 255 and idx_255 not in adj[candidates.index(x)]:
                    result[i] = 255
                    break

    return result

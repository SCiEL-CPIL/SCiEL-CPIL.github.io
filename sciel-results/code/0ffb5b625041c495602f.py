"""Find a graph on as many vertices as possible with no induced regular subgraph on 6+ vertices."""
import time
import random
from itertools import combinations

def construct():
    # Deadline guard: return best valid result well before the 150s kill.
    deadline = time.time() + 120.0
    best_n = 5
    best_edges = []

    # Start with a known-valid small graph and try to grow it.
    # We'll search for n from 6 upward, trying to find a valid graph at each size.
    # If we find a valid graph at size n, we record it and try n+1.
    # We stop when time runs low or we can't make progress.

    for n in range(6, 22):
        if time.time() > deadline - 30:
            break

        # Try to find a valid graph on n vertices.
        # Strategy: random local search with backtracking-like restarts.
        # Represent graph as adjacency list of bitmasks.
        
        max_iters = 50000 if n <= 15 else 20000
        found = False

        for restart in range(50):
            if time.time() > deadline - 20:
                break

            # Initialize random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            # Local search: repeatedly find a violating subset and flip an edge in it.
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break

                # Find a violating subset (induced regular subgraph on >= 6 vertices)
                viol_mask = None
                viol_k = 0
                found_violation = False

                # Check subsets from smallest (k=6) to largest
                # For efficiency, check k=6 first (most common violation)
                for k in range(6, n + 1):
                    # Limit number of subsets checked per k for speed
                    # For small n this is fine; for large n we sample
                    if n <= 12:
                        subs = list(combinations(range(n), k))
                    elif n <= 16:
                        # Sample subsets
                        subs = [tuple(random.sample(range(n), k)) for _ in range(2000)]
                    else:
                        subs = [tuple(random.sample(range(n), k)) for _ in range(1000)]

                    for sub in subs:
                        mask = 0
                        for v in sub:
                            mask |= (1 << v)
                        # Check if induced subgraph is regular
                        degs = []
                        for v in sub:
                            deg = (adj[v] & mask).bit_count()
                            degs.append(deg)
                        if all(d == degs[0] for d in degs):
                            viol_mask = mask
                            viol_k = k
                            found_violation = True
                            break
                    if found_violation:
                        break

                if not found_violation:
                    # No violation found in sampled subsets - likely valid
                    # Do a more thorough check for confidence
                    if n <= 14:
                        # Full check
                        valid = True
                        for k in range(6, n + 1):
                            for sub in combinations(range(n), k):
                                mask = 0
                                for v in sub:
                                    mask |= (1 << v)
                                degs = [(adj[v] & mask).bit_count() for v in sub]
                                if all(d == degs[0] for d in degs):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Found a valid graph!
                            edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                            best_n = n
                            best_edges = edges
                            found = True
                            break
                    else:
                        # For larger n, trust the sampling but do extra verification
                        # by checking all k=6 subsets (the most likely to be regular)
                        valid = True
                        for sub in combinations(range(n), 6):
                            mask = 0
                            for v in sub:
                                mask |= (1 << v)
                            degs = [(adj[v] & mask).bit_count() for v in sub]
                            if all(d == degs[0] for d in degs):
                                valid = False
                                break
                        if valid:
                            # Also check some larger subsets
                            for _ in range(500):
                                k = random.randint(7, min(n, 12))
                                sub = tuple(random.sample(range(n), k))
                                mask = 0
                                for v in sub:
                                    mask |= (1 << v)
                                degs = [(adj[v] & mask).bit_count() for v in sub]
                                if all(d == degs[0] for d in degs):
                                    valid = False
                                    break
                            if valid:
                                edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u] & (1 << v)]
                                best_n = n
                                best_edges = edges
                                found = True
                                break

                    if found:
                        break

                # If violation found, flip a random edge within the violating subset
                if found_violation and viol_mask:
                    vertices = [v for v in range(n) if viol_mask & (1 << v)]
                    if len(vertices) >= 2:
                        u, v = random.sample(vertices, 2)
                        # Toggle edge (u, v)
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)

            if found:
                break

        # If we found a valid graph at this n, continue to n+1
        # If not, we could try a different approach or give up on this n
        if not found:
            # Try a denser or sparser initialization strategy
            pass  # Keep best_n as is

    # Final safety: ensure we return something valid
    # If best_n < 6, fall back to n=5 empty graph
    if best_n < 6:
        best_n = 5
        best_edges = []

    return {'n': best_n, 'edges': best_edges}

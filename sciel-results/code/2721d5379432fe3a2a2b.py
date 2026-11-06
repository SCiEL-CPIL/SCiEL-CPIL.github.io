"""
Strategy: Randomized greedy + local search for Maximum Independent Set
in a 3-uniform hypergraph. Multiple random orderings with conflict-driven
removal and iterative improvement via re-insertion attempts.
"""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Generate all forbidden triples
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    # Parameterization: b = kx(x+y), c = ky(x+y), a = kxy
    # with gcd(x,y)=1, x!=y, and k >= 1.
    triples = []
    max_xy = 20
    for x in range(1, max_xy + 1):
        for y in range(1, max_xy + 1):
            if x == y:
                continue
            if math.gcd(x, y) != 1:
                continue
            M = (x + y) * max(x, y)
            if M > N:
                continue
            max_k = N // M
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= N and b <= N and c <= N:
                    triples.append((a - 1, b - 1, c - 1))
    
    if not triples:
        return numbers
    
    # Build adjacency: for each vertex, list of triples it participates in
    vertex_triples = [[] for _ in range(N)]
    for idx, (u, v, w) in enumerate(triples):
        vertex_triples[u].append(idx)
        vertex_triples[v].append(idx)
        vertex_triples[w].append(idx)
    
    # Local search: start from a greedy solution, try to improve
    def _greedy_with_order(order):
        """Greedy: add vertices in given order if no conflict."""
        in_set = [False] * N
        triple_count = [0] * len(triples)  # how many vertices of each triple are in set
        
        for v in order:
            # Check if adding v creates any forbidden triple
            conflict = False
            for t_idx in vertex_triples[v]:
                if triple_count[t_idx] >= 2:
                    conflict = True
                    break
            if not conflict:
                in_set[v] = True
                for t_idx in vertex_triples[v]:
                    triple_count[t_idx] += 1
        
        return in_set, triple_count
    
    def _try_improve(in_set, triple_count, max_iters=200):
        """Try to re-insert removed vertices or swap to improve."""
        removed = [v for v in range(N) if not in_set[v]]
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            random.shuffle(removed)
            for v in removed:
                # Check if v can be added
                conflict = False
                for t_idx in vertex_triples[v]:
                    if triple_count[t_idx] >= 2:
                        conflict = True
                        break
                if not conflict:
                    in_set[v] = True
                    for t_idx in vertex_triples[v]:
                        triple_count[t_idx] += 1
                    removed.remove(v)
                    improved = True
                    break
            # Also try swaps: remove one, add another
            if not improved and len(removed) > 0 and max_iters > 0:
                max_iters -= 1
                for v_out in [v for v in range(N) if in_set[v]]:
                    # Temporarily remove v_out
                    for t_idx in vertex_triples[v_out]:
                        triple_count[t_idx] -= 1
                    in_set[v_out] = False
                    # Try to add a removed vertex
                    added = False
                    for v_in in removed:
                        conflict = False
                        for t_idx in vertex_triples[v_in]:
                            if triple_count[t_idx] >= 2:
                                conflict = True
                                break
                        if not conflict:
                            in_set[v_in] = True
                            for t_idx in vertex_triples[v_in]:
                                triple_count[t_idx] += 1
                            removed.remove(v_in)
                            added = True
                            break
                    if added:
                        improved = True
                        break
                    else:
                        # Restore
                        in_set[v_out] = True
                        for t_idx in vertex_triples[v_out]:
                            triple_count[t_idx] += 1
        
        return in_set, triple_count
    
    best_set = None
    best_size = 0
    
    # Try multiple random orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N))
        random.shuffle(order)
        
        in_set, triple_count = _greedy_with_order(order)
        in_set, triple_count = _try_improve(in_set, triple_count, max_iters=50)
        
        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_set = in_set[:]
        
        # If we've found a very large set, we can stop
        if best_size >= 150:
            break
    
    if best_set is None:
        # Fallback
        return list(range(91, 181))
    
    result = [i + 1 for i in range(N) if best_set[i]]
    if result:
        return result
    return list(range(91, 181))

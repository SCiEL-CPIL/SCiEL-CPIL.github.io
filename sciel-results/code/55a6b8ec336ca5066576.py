"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    N = 256
    # Perfect squares up to 512
    squares = set()
    for k in range(1, 23):
        sq = k * k
        if sq <= 512:
            squares.add(sq)
    
    # Build conflict sets
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if a + b in squares:
                conflicts[a].add(b)
                if a != b:
                    conflicts[b].add(a)
    
    # Greedy: sort by degree, add if no conflict with current set
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: degrees[x])
    
    in_set = [False] * (N + 1)
    for v in order:
        if time.time() > deadline:
            break
        if not any(in_set[u] for u in conflicts[v]):
            in_set[v] = True
    
    # Local search with conflict tracking
    def can_add(v, in_set, conflicts):
        for u in conflicts[v]:
            if in_set[u]:
                return False
        return True
    
    # Phase 1: try adding any excluded vertex
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            if in_set[v]:
                continue
            if can_add(v, in_set, conflicts):
                in_set[v] = True
                improved = True
    
    # Phase 2: 1-swap (remove one, add one)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            if not in_set[v]:
                continue
            for w in range(1, N + 1):
                if in_set[w]:
                    continue
                # Check if w can be added after removing v
                ok = True
                for u in conflicts[w]:
                    if in_set[u] and u != v:
                        ok = False
                        break
                if ok:
                    in_set[v] = False
                    in_set[w] = True
                    improved = True
                    break
            if improved:
                break
    
    # Phase 3: 2-swap (remove two, add one)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v1 in range(1, N + 1):
            if not in_set[v1]:
                continue
            for v2 in range(v1 + 1, N + 1):
                if not in_set[v2]:
                    continue
                for w in range(1, N + 1):
                    if in_set[w]:
                        continue
                    ok = True
                    for u in conflicts[w]:
                        if in_set[u] and u != v1 and u != v2:
                            ok = False
                            break
                    if ok:
                        in_set[v1] = False
                        in_set[v2] = False
                        in_set[w] = True
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    result = [i for i in range(1, N + 1) if in_set[i]]
    return result
# EVOLVE-BLOCK-END

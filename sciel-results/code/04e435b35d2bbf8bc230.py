"""Rewritten program: maximum independent set in a 3-uniform hypergraph
with branch-and-bound using degree-based upper bounds."""
import time

def construct():
    N = 256
    vals = list(range(2, N + 1))
    n = len(vals)

    # Build hyperedges: triple {a,b,c} is "bad" if some member divides the
    # product of the other two.
    bad = [set() for _ in range(n)]
    for i in range(n):
        ai = vals[i]
        for j in range(i + 1, n):
            aj = vals[j]
            for k in range(j + 1, n):
                ak = vals[k]
                if ai % (aj * ak) == 0 or aj % (ai * ak) == 0 or ak % (ai * aj) == 0:
                    bad[i].add((j, k))
                    bad[j].add((i, k))
                    bad[k].add((i, j))

    best_set = [False] * n
    best_count = 0
    start_time = time.time()
    TIMEOUT = 55.0

    def branch(remaining, chosen, count):
        nonlocal best_set, best_count
        if time.time() - start_time > TIMEOUT:
            return
        # Upper bound
        if count + len(remaining) <= best_count:
            return
        if not remaining:
            if count > best_count:
                best_count = count
                best_set[:] = chosen[:]
            return
        rem_list = list(remaining)
        # Pick pivot: vertex with maximum degree within remaining
        max_deg = -1
        pivot = rem_list[0]
        for v in rem_list:
            d = 0
            for (u, w) in bad[v]:
                if u in remaining and w in remaining:
                    d += 1
            if d > max_deg:
                max_deg = d
                pivot = v
        # Branch 1: exclude pivot
        branch(remaining - {pivot}, chosen, count)
        # Branch 2: include pivot
        new_remaining = set(remaining)
        new_remaining.discard(pivot)
        to_remove = set()
        for (u, w) in bad[pivot]:
            if w in new_remaining:
                to_remove.add(u)
            if u in new_remaining:
                to_remove.add(w)
        new_remaining -= to_remove
        chosen.append(pivot)
        branch(new_remaining, chosen, count + 1)
        chosen.pop()

    all_vertices = set(range(n))
    branch(all_vertices, [], 0)

    result = [vals[i] for i in range(n) if best_set[i]]
    return list(result)

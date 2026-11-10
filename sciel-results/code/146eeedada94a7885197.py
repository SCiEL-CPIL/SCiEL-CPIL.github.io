import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute edge cycles
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i in range(N):
        for j in range(i + 1, N):
            eij = edge_index(i, j)
            for k in range(N):
                if k == i or k == j:
                    continue
                for l in range(N):
                    if l == i or l == j or l == k:
                        continue
                    ejk = edge_index(j, k)
                    ekl = edge_index(k, l)
                    eli = edge_index(l, i)
                    edge_to_cycles[eij].append((ejk, ekl, eli))
                    ejl = edge_index(j, l)
                    elk = edge_index(l, k)
                    eki = edge_index(k, i)
                    edge_to_cycles[eij].append((ejl, elk, eki))
    
    # Deduplicate
    edge_to_cycles = [list(set(cycles)) for cycles in edge_to_cycles]
    
    # Initial: color by first vertex
    coloring = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            coloring[edge_index(i, j)] = i
    
    def check_edge_valid(e, coloring):
        for cycle in edge_to_cycles[e]:
            if len({coloring[cycle[0]], coloring[cycle[1]], coloring[cycle[2]], coloring[cycle[3]]}) == 4:
                return False
        return True
    
    # Verify initial is valid
    for e in range(NUM_EDGES):
        if not check_edge_valid(e, coloring):
            break
    
    best = coloring[:]
    best_count = len(set(best))
    
    random.seed(42)
    current = coloring[:]
    
    for it in range(500000):
        if time.time() > deadline:
            break
        e = random.randint(0, NUM_EDGES - 1)
        new_c = random.randint(0, NUM_COLORS - 1)
        old_c = current[e]
        if new_c == old_c:
            continue
        current[e] = new_c
        ok = True
        for cyc in edge_to_cycles[e]:
            if len({current[cyc[0]], current[cyc[1]], current[cyc[2]], current[cyc[3]]}) == 4:
                ok = False
                break
        if ok:
            cnt = len(set(current))
            if cnt > best_count:
                best = current[:]
                best_count = cnt
        else:
            current[e] = old_c
    
    return best

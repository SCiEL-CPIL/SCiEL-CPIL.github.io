import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Bipartite graph K10,10
    # L: 0..9, R: 10..19
    # Edge (i, 10+j) has color C[i][j]
    
    C = [[random.randint(0, 19) for _ in range(10)] for _ in range(10)]
    
    def is_rainbow(i, j, k, l):
        c1 = C[i][j]
        c2 = C[k][j]
        c3 = C[k][l]
        c4 = C[i][l]
        return len({c1, c2, c3, c4}) == 4

    def count_bad_for_edge(i, j):
        bad = 0
        for k in range(10):
            if k == i: continue
            for l in range(10):
                if l == j: continue
                if not is_rainbow(i, j, k, l):
                    bad += 1
        return bad

    def count_total_bad():
        bad = 0
        for i in range(10):
            for k in range(10):
                if i == k: continue
                for j in range(10):
                    for l in range(10):
                        if j == l: continue
                        if not is_rainbow(i, j, k, l):
                            bad += 1
        return bad

    current_bad = count_total_bad()
    best_bad = current_bad
    best_C = [row[:] for row in C]
    
    # Local Search
    stuck_counter = 0
    while current_bad > 0 and time.time() < deadline:
        # Find a bad cycle
        # If current_bad is small, scan all. If large, random sample.
        cycle = None
        if current_bad < 100:
            # Scan all
            for i in range(10):
                for k in range(10):
                    if i == k: continue
                    for j in range(10):
                        for l in range(10):
                            if j == l: continue
                            if not is_rainbow(i, j, k, l):
                                cycle = (i, j, k, l)
                                break
                        if cycle: break
                    if cycle: break
                if cycle: break
        else:
            # Random sample
            for _ in range(100):
                i = random.randint(0, 9)
                k = random.randint(0, 9)
                if i == k: continue
                j = random.randint(0, 9)
                l = random.randint(0, 9)
                if j == l: continue
                if not is_rainbow(i, j, k, l):
                    cycle = (i, j, k, l)
                    break
        
        if cycle is None:
            # Should not happen if current_bad > 0, but safe
            break
            
        i, j, k, l = cycle
        
        # Pick random edge in cycle
        edge_idx = random.randint(0, 3)
        if edge_idx == 0:
            ei, ej = i, j
        elif edge_idx == 1:
            ei, ej = k, j
        elif edge_idx == 2:
            ei, ej = k, l
        else:
            ei, ej = i, l
            
        old_color = C[ei][ej]
        old_bad_for_edge = count_bad_for_edge(ei, ej)
        
        best_color = old_color
        min_bad = old_bad_for_edge
        
        for c in range(20):
            if c == old_color:
                continue
            C[ei][ej] = c
            new_bad = count_bad_for_edge(ei, ej)
            if new_bad < min_bad:
                min_bad = new_bad
                best_color = c
                
        C[ei][ej] = best_color
        current_bad += min_bad - old_bad_for_edge
        
        if current_bad < best_bad:
            best_bad = current_bad
            best_C = [row[:] for row in C]
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        # If stuck for a long time, random restart or kick
        if stuck_counter > 5000:
            # Randomly perturb a few edges
            for _ in range(10):
                ri = random.randint(0, 9)
                rj = random.randint(0, 9)
                C[ri][rj] = random.randint(0, 19)
            current_bad = count_total_bad()
            stuck_counter = 0
            if current_bad < best_bad:
                best_bad = current_bad
                best_C = [row[:] for row in C]

    # Construct output
    edges = []
    for i in range(10):
        for j in range(10):
            edges.append([i, 10+j, best_C[i][j]])
            
    return edges

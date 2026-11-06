import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Seed for n=9
    def seed_colors(n):
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        
    current_n = 9
    current_colors = seed_colors(9)
    
    # Helper to get edge index
    def get_edge_idx(n, i, j):
        if i > j: i, j = j, i
        # Lexicographic order
        # Pairs: (0,1)..(0,n-1), (1,2)..(1,n-1), ...
        # Count pairs before i:
        # i=0: 0
        # i=1: n-1
        # i=2: (n-1) + (n-2)
        # Sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_c6(coloring, n):
        graphs = [ [set() for _ in range(n)] for _ in range(4) ]
        for i in range(n):
            for j in range(i+1, n):
                c = coloring[get_edge_idx(n, i, j)]
                graphs[c][i].add(j)
                graphs[c][j].add(i)
                
        count = 0
        for c in range(4):
            for verts in itertools.combinations(range(n), 6):
                deg = [0] * 6
                v_set = set(verts)
                for i_idx, u in enumerate(verts):
                    for v in graphs[c][u]:
                        if v in v_set:
                            deg[i_idx] += 1
                if all(d == 2 for d in deg):
                    count += 1
        return count

    def find_one_c6(coloring, n):
        graphs = [ [set() for _ in range(n)] for _ in range(4) ]
        for i in range(n):
            for j in range(i+1, n):
                c = coloring[get_edge_idx(n, i, j)]
                graphs[c][i].add(j)
                graphs[c][j].add(i)
                
        for c in range(4):
            for verts in itertools.combinations(range(n), 6):
                deg = [0] * 6
                v_set = set(verts)
                valid = True
                for i_idx, u in enumerate(verts):
                    for v in graphs[c][u]:
                        if v in v_set:
                            deg[i_idx] += 1
                    if deg[i_idx] != 2:
                        valid = False
                        break
                if valid:
                    edges = []
                    for i_idx, u in enumerate(verts):
                        for v in graphs[c][u]:
                            if v in v_set and u < v:
                                edges.append(get_edge_idx(n, u, v))
                    return edges
        return None

    def local_search(coloring, n, max_iters, deadline):
        score = count_c6(coloring, n)
        if score == 0:
            return coloring
            
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            violation = find_one_c6(coloring, n)
            if violation is None:
                return coloring
                
            edge_idx = random.choice(violation)
            old_c = coloring[edge_idx]
            best_new_c = old_c
            best_new_score = score
            
            for new_c in range(4):
                if new_c == old_c: continue
                coloring[edge_idx] = new_c
                new_score = count_c6(coloring, n)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_c = new_c
                if new_score == 0:
                    return coloring
            coloring[edge_idx] = best_new_c
            score = best_new_score
            
        return None

    def try_solve_n(target_n, prev_colors, prev_n, deadline):
        # Extend
        if prev_n == target_n - 1:
            new_colors = [0] * (target_n * (target_n - 1) // 2)
            # Copy old
            for i in range(prev_n):
                for j in range(i+1, prev_n):
                    new_colors[get_edge_idx(target_n, i, j)] = prev_colors[get_edge_idx(prev_n, i, j)]
            # Randomize new edges
            for i in range(prev_n):
                j = target_n - 1
                new_colors[get_edge_idx(target_n, i, j)] = random.randint(0, 3)
            
            # Local search
            res = local_search(new_colors, target_n, 2000, deadline)
            if res:
                return res
                
        # Random restarts
        for _ in range(10):
            if time.time() > deadline:
                return None
            random_colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            res = local_search(random_colors, target_n, 1000, deadline)
            if res:
                return res
                
        return None

    for target_n in range(10, 14):
        if time.time() > deadline:
            break
            
        res = try_solve_n(target_n, current_colors, current_n, deadline)
        if res:
            current_n = target_n
            current_colors = res
        else:
            break
            
    return {'n': current_n, 'colors': current_colors}

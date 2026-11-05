import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try 17, 16, ...
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 2.0:
            break
            
        time_left = deadline - time.time()
        # Allocate time
        alloc_time = min(10.0, time_left * 0.4)
        if alloc_time < 1.0:
            continue
            
        result = solve_csp(n, alloc_time)
        if result:
            best_n = n
            best_colors = result
            if n == 17 and time.time() < deadline - 10.0:
                 res18 = solve_csp(18, min(10.0, deadline - time.time() - 2.0))
                 if res18:
                     best_n = 18
                     best_colors = res18

    return {'n': best_n, 'colors': best_colors}

def solve_csp(n, time_limit):
    start_time = time.time()
    
    # Generate all 6-vertex subsets
    six_sets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        for p in range(m+1, n):
                            six_sets.append((i, j, k, l, m, p))
                            
    # Precompute patterns
    all_patterns = []
    edge_index = {}
    edges_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edges_list.append((i,j))
            idx += 1
    m = len(edges_list)
    
    edge_to_patterns = [[] for _ in range(m)]
    
    for S in six_sets:
        verts = list(S)
        seen = set()
        for perm in itertools.permutations([1,2,3,4,5]):
            edges = []
            for i in range(5):
                u = 0 if i == 0 else perm[i-1]
                v = perm[i]
                if u > v: u, v = v, u
                edges.append((u,v))
            u = perm[4]
            v = 0
            if u > v: u, v = v, u
            edges.append((u,v))
            fe = frozenset(edges)
            if fe not in seen:
                seen.add(fe)
                ge = []
                for eu, ev in edges:
                    gu, gv = verts[eu], verts[ev]
                    ge.append((min(gu,gv), max(gu,gv)))
                pat_idx = len(all_patterns)
                all_patterns.append(ge)
                for e in ge:
                    edge_to_patterns[edge_index[e]].append(pat_idx)
                    
    # Initialize colors
    colors = [random.randint(0,3) for _ in range(m)]
    
    # Initialize counts
    pat_counts = [[0,0,0,0] for _ in range(len(all_patterns))]
    for p, pat in enumerate(all_patterns):
        for e in pat:
            c = colors[edge_index[e]]
            pat_counts[p][c] += 1
            
    energy = sum(1 for p in range(len(pat_counts)) if max(pat_counts[p]) == 6)
    
    T = 1.0
    t_start = time.time()
    
    while time.time() - t_start < time_limit:
        e_idx = random.randint(0, m-1)
        c_old = colors[e_idx]
        c_new = random.randint(0,3)
        if c_new == c_old:
            continue
            
        delta = 0
        pats = edge_to_patterns[e_idx]
        for p in pats:
            counts = pat_counts[p]
            if counts[c_old] == 6:
                delta -= 1
            if counts[c_new] == 5:
                delta += 1
                
        if delta <= 0 or random.random() < pow(2.718, -delta / T):
            colors[e_idx] = c_new
            energy += delta
            for p in pats:
                pat_counts[p][c_old] -= 1
                pat_counts[p][c_new] += 1
                
        T *= 0.995
        
    if energy == 0:
        return colors
    return None

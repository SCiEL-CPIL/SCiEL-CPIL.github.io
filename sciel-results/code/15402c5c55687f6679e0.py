import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    m = len(edges)
    
    k4s = []
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        es = []
        for u, v in itertools.combinations(quad, 2):
            es.append(edge_index[(u,v)])
        k4s.append(quad)
        k4_edges.append(es)
        
    edge_to_k4s = [[] for _ in range(m)]
    for idx, es in enumerate(k4_edges):
        for e in es:
            edge_to_k4s[e].append(idx)
            
    def get_bad_k4s(coloring):
        bads = []
        for idx, es in enumerate(k4_edges):
            colors = [coloring[e] for e in es]
            if len(set(colors)) < 5:
                bads.append(idx)
        return bads

    def solve(K, time_limit):
        # Random init
        coloring = [random.randint(0, K-1) for _ in range(m)]
        bads = get_bad_k4s(coloring)
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() > time_limit:
                return None
            
            if not bads:
                return coloring
                
            # Pick a random bad K4
            bad_idx = random.choice(bads)
            es = k4_edges[bad_idx]
            
            # Pick a random edge in this K4 to recolor
            e_idx = random.choice(es)
            
            # Try all K colors
            best_new_c = None
            min_bad = float('inf')
            
            for c in range(K):
                if c == coloring[e_idx]:
                    continue
                    
                total_bad_after = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    es_k4 = k4_edges[k4_idx]
                    # Optimize distinct count
                    # List of 6 colors
                    c0 = coloring[es_k4[0]] if es_k4[0] != e_idx else c
                    c1 = coloring[es_k4[1]] if es_k4[1] != e_idx else c
                    c2 = coloring[es_k4[2]] if es_k4[2] != e_idx else c
                    c3 = coloring[es_k4[3]] if es_k4[3] != e_idx else c
                    c4 = coloring[es_k4[4]] if es_k4[4] != e_idx else c
                    c5 = coloring[es_k4[5]] if es_k4[5] != e_idx else c
                    
                    # Check distinct
                    # Since K is small, set is fast enough
                    if len({c0, c1, c2, c3, c4, c5}) < 5:
                        total_bad_after += 1
                        
                if total_bad_after < min_bad:
                    min_bad = total_bad_after
                    best_new_c = c
            
            if best_new_c is not None:
                coloring[e_idx] = best_new_c
                # Update bads
                # To save time, only update if we suspect improvement or every few steps
                # But we need to know if bads is empty.
                # Let's update bads every 10 steps.
                if it % 10 == 0:
                    bads = get_bad_k4s(coloring)
            else:
                # Stuck, perturb
                e_rand = random.randint(0, m-1)
                c_rand = random.randint(0, K-1)
                coloring[e_rand] = c_rand
                bads = get_bad_k4s(coloring)
                
        return None

    best_result = None
    best_colors = 100
    
    # Time allocation
    time_per_k = 30.0
    current_time = start_time
    
    for K in range(13, 18):
        if time.time() > deadline - 5: # Leave 5s for finalization
            break
            
        target_time = current_time + time_per_k
        if target_time > deadline - 5:
            target_time = deadline - 5
            
        found = False
        # Try a few restarts
        for restart in range(5):
            if time.time() > target_time:
                break
            res = solve(K, target_time)
            if res:
                used = set(res)
                mapping = {c: i for i, c in enumerate(sorted(used))}
                final = [mapping[c] for c in res]
                if len(used) < best_colors:
                    best_colors = len(used)
                    best_result = final
                found = True
                break
                
        if found and K == 13:
            return best_result
            
        current_time = time.time()
        
    if best_result:
        return best_result
        
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

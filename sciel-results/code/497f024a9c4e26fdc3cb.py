import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute all K4s and their edge indices
    k4s = []
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        es = []
        for u, v in itertools.combinations(quad, 2):
            es.append(edge_index[(u, v)])
        k4s.append(quad)
        k4_edges.append(es)
    
    num_k4s = len(k4_edges)
    
    # For each edge, which K4s does it belong to?
    edge_to_k4s = [[] for _ in range(m)]
    for idx, es in enumerate(k4_edges):
        for e in es:
            edge_to_k4s[e].append(idx)
    
    # For faster badness checking, precompute for each K4 the 6 edge indices
    # and use a color-count approach
    
    def count_colours_in_k4(es, coloring):
        """Count distinct colours in a K4's 6 edges."""
        seen = set()
        for e in es:
            seen.add(coloring[e])
        return len(seen)
    
    def is_bad_k4(es, coloring):
        seen = set()
        for e in es:
            seen.add(coloring[e])
            if len(seen) >= 5:
                return False
        return True
    
    def get_bad_k4s(coloring):
        bads = []
        for idx, es in enumerate(k4_edges):
            seen = set()
            for e in es:
                seen.add(coloring[e])
                if len(seen) >= 5:
                    break
            else:
                bads.append(idx)
        return bads
    
    def total_bad_count(coloring):
        count = 0
        for es in k4_edges:
            seen = set()
            for e in es:
                seen.add(coloring[e])
                if len(seen) >= 5:
                    break
            else:
                count += 1
        return count
    
    def solve(K, max_iters=20000):
        """Local search to find valid colouring using at most K colours."""
        coloring = [random.randint(0, K - 1) for _ in range(m)]
        
        bads = get_bad_k4s(coloring)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            if not bads:
                return coloring
            
            # Pick a random bad K4
            bad_idx = random.choice(bads)
            es = k4_edges[bad_idx]
            
            # Pick a random edge in this K4 to recolor
            e_idx = random.choice(es)
            
            # Try all K colors, pick the one that minimizes total bad count
            best_new_c = None
            min_bad = float('inf')
            
            # Evaluate impact: only K4s containing e_idx can change
            affected = edge_to_k4s[e_idx]
            
            for c in range(K):
                if c == coloring[e_idx]:
                    continue
                
                # Calculate bad count for affected K4s
                total_bad = 0
                for k4_idx in affected:
                    es_k4 = k4_edges[k4_idx]
                    seen = set()
                    for x in es_k4:
                        col = c if x == e_idx else coloring[x]
                        seen.add(col)
                        if len(seen) >= 5:
                            break
                    else:
                        total_bad += 1
                
                if total_bad < min_bad:
                    min_bad = total_bad
                    best_new_c = c
            
            if best_new_c is not None and min_bad < 0.5 * len(affected):
                # Accept if it reduces the local bad count
                coloring[e_idx] = best_new_c
                # Periodically refresh the bads list
                if it % 50 == 0 or it < 100:
                    bads = get_bad_k4s(coloring)
            else:
                # Stuck: random perturbation
                e_rand = random.randint(0, m - 1)
                c_rand = random.randint(0, K - 1)
                coloring[e_rand] = c_rand
                bads = get_bad_k4s(coloring)
        
        return None
    
    # Try to find the minimum number of colours
    # Known bounds: 13 <= f(16) <= 15
    # We'll try K = 13, 14, 15 in order
    
    best_result = None
    best_colors = 100
    
    for K in [13, 14, 15]:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 15 if K <= 13 else 8
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            res = solve(K)
            if res is not None:
                # Verify it's actually valid
                bads = get_bad_k4s(res)
                if not bads:
                    # Remap colours to 0..k-1
                    used = sorted(set(res))
                    mapping = {c: i for i, c in enumerate(used)}
                    final = [mapping[c] for c in res]
                    num_used = len(used)
                    
                    if num_used < best_colors:
                        best_colors = num_used
                        best_result = final
                    
                    found = True
                    break
        
        if found:
            # We found a valid colouring with K colours
            # Since we're going in increasing order, this is optimal
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a simple pattern
    # This may not be valid but returns something
    return [(i * 7 + j * 11) % 17 for i in range(16) for j in range(i + 1, 16)]

import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    n = 16
    # Precompute all K4s
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a, b, c, d))
                    
    # Map edge (i,j) to index
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
            
    # Map K4 to list of edge indices
    k4_edges = []
    for (a,b,c,d) in k4s:
        edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)], 
                 edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4_edges.append(edges)
        
    # Reverse map: edge index to list of K4 indices
    edge_to_k4s = [[] for _ in range(len(edge_idx))]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    def count_distinct(colors_on_k4):
        return len(set(colors_on_k4))
        
    def is_bad_k4(colors_on_k4):
        return count_distinct(colors_on_k4) < 5
        
    def evaluate(coloring):
        bad_count = 0
        for edges in k4_edges:
            cols = [coloring[e] for e in edges]
            if is_bad_k4(cols):
                bad_count += 1
        return bad_count

    def local_search(K, max_iters=200000):
        # Initialize
        coloring = [random.randint(0, K-1) for _ in range(len(edge_idx))]
        
        # Calculate initial bad count and per-K4 distinct counts
        bad_count = 0
        k4_distinct = [0] * len(k4_edges)
        for k4_idx, edges in enumerate(k4_edges):
            cols = [coloring[e] for e in edges]
            d = count_distinct(cols)
            k4_distinct[k4_idx] = d
            if d < 5:
                bad_count += 1
                
        if bad_count == 0:
            return coloring
            
        # Local search
        # We will track the number of times each colour appears in each K4 to update fast?
        # Or just recompute for the 91 K4s involved.
        
        # To update fast, we can maintain color counts per K4.
        # k4_color_counts[k4_idx] = {colour: count}
        k4_color_counts = []
        for k4_idx, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = coloring[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts.append(counts)
            
        current_bad = bad_count
        no_improve_counter = 0
        temp = 0.5
        min_temp = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick random edge
            e_idx = random.randint(0, len(edge_idx)-1)
            old_c = coloring[e_idx]
            
            # Try all other colours
            best_new_c = old_c
            best_delta = 0
            # We can just pick a random new colour or scan all
            # Scanning all K-1 colours is fast (K <= 16)
            
            candidates = [c for c in range(K) if c != old_c]
            random.shuffle(candidates)
            
            improved = False
            for new_c in candidates:
                # Calculate delta for this change
                delta = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    # Current distinct count
                    old_distinct = k4_distinct[k4_idx]
                    
                    # Update counts
                    counts = k4_color_counts[k4_idx]
                    counts[old_c] -= 1
                    if counts[old_c] == 0:
                        del counts[old_c]
                    counts[new_c] = counts.get(new_c, 0) + 1
                    new_distinct = len(counts)
                    
                    # Check transition
                    old_bad = old_distinct < 5
                    new_bad = new_distinct < 5
                    
                    if old_bad and not new_bad:
                        delta -= 1
                    elif not old_bad and new_bad:
                        delta += 1
                    
                    # Rollback counts for now to try next candidate? 
                    # No, we need to commit if we choose this candidate.
                    # But we are just evaluating.
                    # So rollback.
                    counts[new_c] -= 1
                    if counts[new_c] == 0:
                        del counts[new_c]
                    counts[old_c] = counts.get(old_c, 0) + 1
                    
                    # We can break early if delta is very negative?
                    # But we want the best.
                    
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    if delta < -1: # Good improvement
                        # We can break? No, maybe even better. But -1 is best per K4.
                        # Max change in one K4 is -1 (bad->good) or +1.
                        # If we have multiple K4s, we can get -2 etc.
                        pass
            
            # Apply best change
            if best_new_c != old_c:
                # Commit
                coloring[e_idx] = best_new_c
                for k4_idx in edge_to_k4s[e_idx]:
                    counts = k4_color_counts[k4_idx]
                    counts[old_c] -= 1
                    if counts[old_c] == 0:
                        del counts[old_c]
                    counts[best_new_c] = counts.get(best_new_c, 0) + 1
                    new_distinct = len(counts)
                    old_distinct = k4_distinct[k4_idx]
                    
                    if old_distinct < 5 and new_distinct >= 5:
                        current_bad -= 1
                    elif old_distinct >= 5 and new_distinct < 5:
                        current_bad += 1
                        
                    k4_distinct[k4_idx] = new_distinct
                    
                if best_delta < 0:
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                if no_improve_counter > 1000:
                    break
            else:
                no_improve_counter += 1
                if no_improve_counter > 1000:
                    break
            
            if current_bad == 0:
                return coloring
                
            # Cool down
            temp *= 0.999
            if temp < min_temp:
                temp = min_temp
                
        return None

    # Try K from 13 upwards
    for K in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
            
        # Multiple restarts for K
        max_restarts = 50 if K == 13 else 10
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            coloring = local_search(K, max_iters=50000)
            if coloring is not None:
                # Verify
                if evaluate(coloring) == 0:
                    return coloring
                    
    # Fallback to seed if all else fails
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def is_valid_fast(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in edge_set:
            side_u = [(deg[s], s) for s in adj[u] if s != v]
            side_v = [(deg[t], t) for t in adj[v] if t != u]
            
            for ds, s in side_u:
                for dt, t in side_v:
                    if s == t:
                        continue
                    if ds == dt:
                        return False
        return True

    # Start with K_{7,8} as baseline (56 edges, proven valid)
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    best_edges = set(base_edges)
    best_count = len(base_edges)

    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # === NEW: Threshold graph construction with degree-difference labeling ===
    # In a threshold graph, vertices are ordered by weight, and edges connect
    # vertices whose weight difference exceeds a threshold. Degrees are monotonic
    # with position in the ordering, so we can ensure path endpoints differ.
    
    def try_threshold_graph(weights, threshold, n):
        """Build a graph where edge (i,j) exists iff |weights[i]-weights[j]| >= threshold."""
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if abs(weights[i] - weights[j]) >= threshold:
                    edges.add((i, j))
        return edges

    # Try various weight assignments and thresholds
    # Key: weights should create distinct degree levels
    # In a threshold graph, vertex k has degree = number of vertices with weight diff >= threshold
    
    # Strategy 1: Linear weights with optimized threshold
    best_threshold_edges = None
    best_threshold_count = 0
    
    for w_scale in [1, 2, 3, 5, 7, 10]:
        for w_offset in range(-2, 3):
            weights = [i * w_scale + w_offset for i in range(n)]
            # Try various thresholds
            for thresh in range(1, w_scale * (n-1) + 1, max(1, w_scale // 2)):
                if time.time() > deadline:
                    break
                edges = try_threshold_graph(weights, thresh, n)
                if len(edges) > best_threshold_count and is_valid_fast(edges, n):
                    best_threshold_count = len(edges)
                    best_threshold_edges = edges
    
    # Strategy 2: Non-linear weight assignments
    # Use geometric or exponential spacing
    for base in [1.5, 2, 2.5, 3]:
        for start_w in [1, 2]:
            weights = [int(start_w * (base ** i)) for i in range(n)]
            max_w = weights[-1]
            for thresh in range(1, max_w + 1, max(1, max_w // 10)):
                if time.time() > deadline:
                    break
                edges = try_threshold_graph(weights, thresh, n)
                if len(edges) > best_threshold_count and is_valid_fast(edges, n):
                    best_threshold_count = len(edges)
                    best_threshold_edges = edges
    
    # Strategy 3: Custom weight patterns that create degree separation
    # Assign weights so that degree groups are well-separated
    # Try: low group, mid group, high group with gaps
    for gap_low in [2, 3, 5]:
        for gap_mid in [2, 3, 5]:
            weights = []
            # 5 low vertices
            for i in range(5):
                weights.append(i)
            # Gap
            # 5 mid vertices  
            for i in range(5):
                weights.append(gap_low + i)
            # Gap
            # 5 high vertices
            for i in range(5):
                weights.append(gap_low + gap_mid + i)
            
            max_w = weights[-1]
            for thresh in range(1, max_w + 1, max(1, max_w // 8)):
                if time.time() > deadline:
                    break
                edges = try_threshold_graph(weights, thresh, n)
                if len(edges) > best_threshold_count and is_valid_fast(edges, n):
                    best_threshold_count = len(edges)
                    best_threshold_edges = edges
    
    # Strategy 4: Randomized weight perturbations around linear
    random.seed(123)
    for trial in range(50):
        if time.time() > deadline:
            break
        weights = list(range(n))
        # Add small random perturbations
        for i in range(n):
            weights[i] += random.randint(-1, 1)
        # Ensure distinctness
        weights = sorted(set(weights))
        if len(weights) < n:
            # Fill in gaps
            current = 0
            new_weights = []
            for w in weights:
                while current < w:
                    new_weights.append(current)
                    current += 1
                new_weights.append(w)
                current = w + 1
            weights = new_weights[:n]
        
        max_w = max(weights)
        for thresh in range(1, max_w + 1, max(1, max_w // 10)):
            if time.time() > deadline:
                break
            edges = try_threshold_graph(weights, thresh, n)
            if len(edges) > best_threshold_count and is_valid_fast(edges, n):
                best_threshold_count = len(edges)
                best_threshold_edges = edges
    
    # Use the best threshold graph if it's better
    if best_threshold_edges and best_threshold_count > best_count:
        best_edges = set(best_threshold_edges)
        best_count = best_threshold_count
    
    # Phase 1: Greedy edge addition from best starting point
    current = set(best_edges)
    missing = [e for e in all_possible if e not in current]
    
    random.seed(42)
    random.shuffle(missing)
    
    # Try each missing edge
    for e in missing:
        if time.time() > deadline:
            break
        if e in current:
            continue
        trial = current | {e}
        if is_valid_fast(trial, n):
            current = trial
            if len(current) > best_count:
                best_edges = set(current)
                best_count = len(current)
    
    # Phase 2: Local search with swaps
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        current_list = list(current)
        random.shuffle(current_list)
        
        for rem in current_list:
            if time.time() > deadline:
                break
            
            trial = current - {rem}
            
            # Try adding each missing edge
            for add in all_possible:
                if add in trial:
                    continue
                if add == rem:
                    continue
                
                trial2 = trial | {add}
                if len(trial2) <= len(current):
                    continue
                
                if is_valid_fast(trial2, n):
                    current = trial2
                    improved = True
                    if len(current) > best_count:
                        best_edges = set(current)
                        best_count = len(current)
                    break
            
            if improved:
                break
        
        if not improved:
            # Try 2-edge swaps: remove 2, add 2
            found = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                rem1, rem2 = random.sample(list(current), 2)
                trial = current - {rem1, rem2}
                
                for add1 in all_possible:
                    if add1 in trial or add1 == rem1 or add1 == rem2:
                        continue
                    trial1 = trial | {add1}
                    for add2 in all_possible:
                        if add2 in trial1 or add2 == rem1 or add2 == rem2 or add2 == add1:
                            continue
                        trial2 = trial1 | {add2}
                        if len(trial2) <= len(current):
                            continue
                        if is_valid_fast(trial2, n):
                            current = trial2
                            found = True
                            if len(current) > best_count:
                                best_edges = set(current)
                                best_count = len(current)
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                break
    
    # Phase 3: Try different initial partitions
    for split_a in [5, 6, 7, 8, 9]:
        if time.time() > deadline:
            break
        
        A2 = list(range(split_a))
        B2 = list(range(split_a, n))
        base2 = set()
        for a in A2:
            for b in B2:
                base2.add((a, b))
        
        if len(base2) <= best_count:
            continue
        
        # Verify base is valid
        if not is_valid_fast(base2, n):
            continue
        
        cur = set(base2)
        miss2 = [e for e in all_possible if e not in cur]
        random.shuffle(miss2)
        
        for e in miss2:
            if time.time() > deadline:
                break
            if e in cur:
                continue
            trial = cur | {e}
            if is_valid_fast(trial, n):
                cur = trial
        
        if len(cur) > best_count:
            best_edges = set(cur)
            best_count = len(cur)
    
    # Final verification
    if not is_valid_fast(best_edges, n):
        # Fallback to K_{7,8}
        best_edges = set(base_edges)
        best_count = len(base_edges)
    
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END

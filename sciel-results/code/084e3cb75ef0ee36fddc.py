import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # Use hexagon of radius 4 for more flexibility (61 points)
    R = 4
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
    
    n_points = len(points)  # 61 points
    
    # Compute edges (distance squared == 1)
    # dist_sq(a1, b1, a2, b2) = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    adj = [set() for _ in range(n_points)]
    deg = [0] * n_points
    
    for i in range(n_points):
        a1, b1 = points[i]
        for j in range(i + 1, n_points):
            a2, b2 = points[j]
            da = a1 - a2
            db = b1 - b2
            d2 = da*da + db*db + da*db
            if d2 == 1:
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
                
    # Greedy initialization: start with all 61 points, remove 29 points
    # Remove points with lowest degree first (they contribute fewest edges)
    # But we need to be smart: removing a point also removes its internal edges
    
    # Better: use a greedy approach to select 32 points
    # Start with empty set, add points one by one that maximize marginal edge gain
    
    # Actually, let's use a different strategy:
    # 1. Start with the 37-point hexagon (radius 3) as initial solution
    # 2. Then try to improve by swapping points
    
    # Generate radius-3 hexagon points
    R3 = 3
    points3 = []
    for a in range(-R3, R3 + 1):
        for b in range(-R3, R3 + 1):
            if abs(a + b) <= R3:
                points3.append((a, b))
    
    # Map radius-3 points to indices in the radius-4 set
    point_to_idx = {p: i for i, p in enumerate(points)}
    idx3 = [point_to_idx[p] for p in points3]
    
    # Start with radius-3 hexagon (37 points), remove 5 to get 32
    # Find best 5 to remove from the 37
    is_edge = [[False] * n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in adj[i]:
            is_edge[i][j] = True
    
    # Compute degrees within the 37-point subset
    deg3 = [0] * len(idx3)
    for i_idx, i in enumerate(idx3):
        for j_idx, j in enumerate(idx3):
            if i_idx < j_idx and is_edge[i][j]:
                deg3[i_idx] += 1
                deg3[j_idx] += 1
    
    # Find best 5 to remove from the 37
    min_cost = float('inf')
    best_S = None
    for S in itertools.combinations(range(len(idx3)), 5):
        sum_deg = sum(deg3[idx] for idx in S)
        edges_int = 0
        for k in range(5):
            i = S[k]
            for l in range(k+1, 5):
                j = S[l]
                if is_edge[idx3[i]][idx3[j]]:
                    edges_int += 1
        cost = sum_deg - edges_int
        if cost < min_cost:
            min_cost = cost
            best_S = S
        if time.time() > deadline - 10:
            break
    
    if best_S is None:
        best_S = tuple(range(5))
    
    # Build initial 32-point set
    keep_set = set(range(len(idx3))) - set(best_S)
    result_idx = [idx3[i] for i in sorted(keep_set)]
    result = [list(points[i]) for i in result_idx]
    
    # Count edges
    def count_edges(result):
        ec = 0
        for i in range(len(result)):
            a1, b1 = result[i]
            for j in range(i + 1, len(result)):
                a2, b2 = result[j]
                da = a1 - a2
                db = b1 - b2
                d2 = da*da + db*db + da*db
                if d2 == 1:
                    ec += 1
        return ec
    
    edge_count = count_edges(result)
    
    # Local search: try swapping a kept point with a point from the full 61-point set
    # that is not in the current solution
    result_set = set(result_idx)
    candidate_pool = [i for i in range(n_points) if i not in result_set]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap_gain = 0
        best_swap = None
        for i in range(32):
            a1, b1 = points[result_idx[i]]
            for c in candidate_pool:
                a2, b2 = points[c]
                # Check distance constraint: a2,b2 must be at distance >= 1 from all other 31 points
                valid = True
                for k in range(32):
                    if k == i:
                        continue
                    a3, b3 = points[result_idx[k]]
                    da = a2 - a3
                    db = b2 - b3
                    d2 = da*da + db*db + da*db
                    if d2 < 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Compute edge change
                edges_removed = 0
                edges_added = 0
                for k in range(32):
                    if k == i:
                        continue
                    a3, b3 = points[result_idx[k]]
                    # Edges removed: between result_idx[i] and result_idx[k]
                    da_r = a1 - a3
                    db_r = b1 - b3
                    d2_r = da_r*da_r + db_r*db_r + da_r*db_r
                    if d2_r == 1:
                        edges_removed += 1
                    # Edges added: between c and result_idx[k]
                    da_a = a2 - a3
                    db_a = b2 - b3
                    d2_a = da_a*da_a + db_a*db_a + da_a*db_a
                    if d2_a == 1:
                        edges_added += 1
                gain = edges_added - edges_removed
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (i, c)
        if best_swap is not None and best_swap_gain > 0:
            i, c = best_swap
            old_idx = result_idx[i]
            result_idx[i] = c
            result[i] = list(points[c])
            result_set.discard(old_idx)
            result_set.add(c)
            candidate_pool.remove(c)
            candidate_pool.append(old_idx)
            improved = True
    
    return result

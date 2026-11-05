import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # Generate hexagon of radius 3
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
    
    n_points = len(points)
    # n_points should be 37
    
    # Compute edges (distance squared == 1)
    # dist_sq(a1, b1, a2, b2) = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    edges = []
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
                edges.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
                
    total_edges = len(edges)
    
    # We need to remove 5 points to get 32.
    # Minimize lost edges = sum(deg[i] for i in S) - edges_within(S)
    # S is a subset of 5 indices.
    
    # Precompute edge lookup for fast internal edge counting
    # Since n is small, we can use a set of frozensets or just a 2D boolean matrix
    is_edge = [[False] * n_points for _ in range(n_points)]
    for i, j in edges:
        is_edge[i][j] = True
        is_edge[j][i] = True
        
    min_cost = float('inf')
    best_S = None
    
    # Iterate over all combinations of 5 points
    # 37 choose 5 is ~4.3M. This is fast.
    for S in itertools.combinations(range(n_points), 5):
        sum_deg = 0
        for idx in S:
            sum_deg += deg[idx]
            
        edges_int = 0
        # Check pairs in S
        for k in range(5):
            i = S[k]
            for l in range(k+1, 5):
                j = S[l]
                if is_edge[i][j]:
                    edges_int += 1
                    
        cost = sum_deg - edges_int
        if cost < min_cost:
            min_cost = cost
            best_S = S
            # Optional: early stop if we reach the theoretical minimum?
            # Theoretical min cost for 5 points:
            # If they form a cluster, deg is high but edges_int is high.
            # We can just let it run.
            
        if time.time() > deadline:
            break
            
    # If best_S is None, something went wrong, return first 32
    if best_S is None:
        return [list(p) for p in points[:32]]
        
    # Construct result
    remove_set = set(best_S)
    result = []
    for i in range(n_points):
        if i not in remove_set:
            result.append(list(points[i]))
            
    # Verify the result has 32 points and all are within bounds
    assert len(result) == 32
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64
        
    # Count edges in the result to verify
    edge_count = 0
    for i in range(32):
        a1, b1 = result[i]
        for j in range(i + 1, 32):
            a2, b2 = result[j]
            da = a1 - a2
            db = b1 - b2
            d2 = da*da + db*db + da*db
            if d2 == 1:
                edge_count += 1
                
    # If we didn't achieve 76 edges, try a local search improvement
    if edge_count < 76:
        # Try swapping: for each point in result and each point not in result (from hexagon),
        # check if swapping improves edge count
        removed_points = [points[i] for i in best_S]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(32):
                for rp in removed_points:
                    a1, b1 = result[i]
                    a2, b2 = rp
                    da = a1 - a2
                    db = b1 - b2
                    d2 = da*da + db*db + da*db
                    if d2 < 1:
                        continue  # Would violate distance constraint
                    # Check if rp is already in result
                    if rp in result:
                        continue
                    # Compute edge change: remove edges of result[i], add edges of rp
                    edges_removed = 0
                    edges_added = 0
                    for k in range(32):
                        if k == i:
                            continue
                        a3, b3 = result[k]
                        da_r = a1 - a3
                        db_r = b1 - b3
                        d2_r = da_r*da_r + db_r*db_r + da_r*db_r
                        if d2_r == 1:
                            edges_removed += 1
                        da_a = a2 - a3
                        db_a = b2 - b3
                        d2_a = da_a*da_a + db_a*db_a + da_a*db_a
                        if d2_a == 1:
                            edges_added += 1
                    if edges_added > edges_removed:
                        # Perform swap
                        result[i] = list(rp)
                        removed_points[removed_points.index(rp)] = list(result[i])
                        improved = True
                        break
                if improved:
                    break
                    
    return result

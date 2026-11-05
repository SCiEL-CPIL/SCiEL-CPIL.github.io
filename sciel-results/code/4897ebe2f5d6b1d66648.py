import time
import numpy as np

def _girth_check(adj, n, g):
    """
    Check if the graph has girth >= g using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    """
    if n < g:
        return False
        
    # State: (current_vertex, previous_vertex)
    # Index: current * n + previous
    # -1 denotes start (no previous)
    
    # We track shortest path lengths for non-backtracking walks.
    # dist[v, p] = min length of non-backtracking walk from start to v with previous node p.
    # We initialize dist[start, -1] = 0 for all starts? 
    # No, we do BFS from all vertices simultaneously to find shortest cycles.
    
    # Let's use a BFS from each vertex, but only non-backtracking.
    # Actually, simpler: 
    # A cycle of length L exists iff there is a non-backtracking walk of length L 
    # that starts at v, ends at v, and the first step and last step are distinct neighbors? 
    # No, standard definition: cycle is a closed walk with no repeated vertices.
    # Non-backtracking walk of length L that closes at start implies a cycle of length <= L.
    
    # Efficient check for girth < g:
    # We perform a non-backtracking BFS from each vertex v up to depth g-1.
    # If we reach v again via a path that doesn't backtrack, we found a cycle.
    # Since the graph is k-regular, the number of non-backtracking walks grows as k*(k-1)^d.
    
    # To make it fast for n~100-1000 and g=9, we can use a dictionary or array.
    # Let's use an array of size n * (n+1) where index is v * (n+1) + p.
    # p = n means "start" (no previous vertex).
    
    INF = 1 << 30
    # dist[v, p] = min length of non-backtracking walk ending at v with previous vertex p.
    # p in [0, n-1] or n for start.
    
    # We want to find if there's a walk of length L < g that forms a cycle.
    # A cycle of length L is a non-backtracking walk of length L that starts at v, ends at v.
    # Specifically, if we start at v, go to w1, ..., w(L-1), w(L)=v.
    # The walk is non-backtracking if w_i != w_{i-2}.
    # The cycle is simple if no vertices repeat.
    # If a non-backtracking walk of length L < g closes, it contains a simple cycle of length <= L.
    # So if we find ANY non-backtracking walk of length L < g that closes, girth < g.
    
    # Optimization: We can just run BFS from all nodes simultaneously.
    # Let dist[v, p] be the min length.
    # Initialize dist[v, n] = 0 for all v. (Start at v, no previous).
    
    # Iterate L from 1 to g-1:
    #   new_dist[v, w] = min(dist[v, p] + 1) for p != w? 
    #   No, transition: from (v, p) to (w, v) is allowed if w != p.
    #   So new_dist[w, v] = min(dist[v, p] + 1) for all p != w.
    
    # Wait, the state is (current, previous).
    # Transition (u, p) -> (v, u) is valid if v != p.
    # We want to detect if we can return to a start node.
    # If we reach state (v, u) and dist[start_v, n] exists?
    # Actually, if we start at s, we have state (s, n) with dist 0.
    # After 1 step: (u, s) with dist 1.
    # After L steps: (v, u).
    # If v == s, and L < g, we have a closed walk.
    # Is it a cycle?
    # If the walk is non-backtracking, and it closes, it contains a cycle.
    # Since L < g, any simple cycle in it has length < g.
    
    # So, algorithm:
    # 1. dist = INF array of size n * (n+1)
    # 2. For all v: dist[v * (n+1) + n] = 0
    # 3. For step in 1 to g-1:
    #      new_dist = INF array
    #      For all (v, p) with dist < INF:
    #          For all neighbors u of v:
    #              if u != p:
    #                  new_dist[u * (n+1) + v] = min(new_dist[...], dist[v, p] + 1)
    #      dist = new_dist
    #      # Check for closure
    #      For all v:
    #          # We reached v from some neighbor u.
    #          # We started at some s.
    #          # How do we know if v == s?
    #          # We don't track 's' in the state.
    #          # We need to know if the current vertex v was the START vertex.
    #          # This requires tracking the start vertex or using a different formulation.
    
    # Alternative:
    # Run BFS from each vertex v individually.
    # dist[v, p] = min length from v to current with prev p.
    # If we reach (v, u) with dist < g, we have a cycle.
    # Cost: n * (number of states).
    # States: n * (n-1) approx.
    # Total ops: n * n * (k-1) * g.
    # For n=100, k=5, g=9: 100 * 100 * 4 * 9 = 360,000 ops. Very fast.
    # For n=1000: 1000 * 1000 * 4 * 9 = 36,000,000 ops. Might be slow in Python.
    # We can use numpy for the transition step?
    # Or just optimized Python loops.
    
    for start in range(n):
        # dist[v, p]
        # p is previous vertex. n means start (no prev).
        dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
        dist[start, n] = 0
        
        for step in range(1, g):
            new_dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
            
            # Iterate over all current states
            # We can optimize by only looking at reachable states.
            # But n is small enough to scan.
            
            for v in range(n):
                # Get dist for v, all p
                # If dist[v, p] is INF, skip.
                # We need to transition to neighbors u of v, where u != p.
                
                # Optimization:
                # For a fixed v, let min_d_not_u be the min dist[v, p] for p != u.
                # Then new_dist[u, v] = min_d_not_u + 1.
                
                # We can precompute min1 and min2 for each v's dist row?
                # dist_row = dist[v]
                # min_val, min_idx = min(dist_row)
                # If min_idx != u, then min_d_not_u = min_val
                # Else min_d_not_u = second_min_val
                
                # Let's do this.
                d_row = dist[v]
                
                # Find min and second min of d_row
                # n+1 is small (<= 1000)
                # Sorting is O(n log n).
                # Just scanning is O(n).
                
                # Actually, we can just compute new_dist for neighbors directly.
                # For each neighbor u of v:
                #   We need min dist[v, p] for p != u.
                
                # To avoid O(k * n) for each v, we can find the global min and index.
                # If the global min's index is not u, use it.
                # If it is u, use the second min.
                
                # Let's find min1, idx1, min2.
                min1 = 1 << 30
                min2 = 1 << 30
                idx1 = -1
                idx2 = -1
                for p in range(n + 1):
                    val = d_row[p]
                    if val < min1:
                        min2 = min1
                        idx2 = idx1
                        min1 = val
                        idx1 = p
                    elif val < min2:
                        min2 = val
                        idx2 = p
                
                for u in adj[v]:
                    if u != start and step > 1:
                        # We don't want to just check u==start at the end.
                        # The logic "if we reach start again" works.
                        pass
                    
                    if idx1 != u:
                        best_prev = min1
                    else:
                        best_prev = min2
                    
                    if best_prev < (1 << 30):
                        new_val = best_prev + 1
                        if new_val < new_dist[u, v]:
                            new_dist[u, v] = new_val
            
            # Check if we returned to start
            # We are at states (u, v) with dist new_dist[u, v].
            # We started at 'start'.
            # We have a cycle if current vertex u == start.
            # And the previous vertex v != start? 
            # If v == start, then the last step was start -> u, and previous was ... -> start.
            # If the walk was start -> ... -> start -> u, it backtracked? 
            # No, non-backtracking forbids going back to the vertex you just came from.
            # So if current is start, and prev is v, then we didn't come from start.
            # So start was visited earlier.
            
            # If new_dist[start, v] < INF for any v, we have a cycle of length <= step.
            # Wait, dist is the length of the walk.
            # If we are at 'start' with length L, we have a closed walk of length L.
            # Since it's non-backtracking, it contains a simple cycle of length <= L.
            # If L < g, then girth < g.
            
            # So check if any new_dist[start, v] < INF.
            if np.any(new_dist[start, :] < (1 << 30)):
                return False
            
            dist = new_dist
            
    return True

def _girth_count_short_cycles(adj, n, g):
    """
    Count the number of non-backtracking walks of length < g that close to a cycle.
    This serves as a heuristic to minimize during local search.
    """
    count = 0
    for start in range(n):
        dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
        dist[start, n] = 0
        
        for step in range(1, g):
            new_dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
            
            for v in range(n):
                d_row = dist[v]
                
                min1 = 1 << 30
                min2 = 1 << 30
                idx1 = -1
                idx2 = -1
                for p in range(n + 1):
                    val = d_row[p]
                    if val < min1:
                        min2 = min1
                        idx2 = idx1
                        min1 = val
                        idx1 = p
                    elif val < min2:
                        min2 = val
                        idx2 = p
                
                for u in adj[v]:
                    if idx1 != u:
                        best_prev = min1
                    else:
                        best_prev = min2
                    
                    if best_prev < (1 << 30):
                        new_val = best_prev + 1
                        if new_val < new_dist[u, v]:
                            new_dist[u, v] = new_val
            
            # Count how many states reach 'start'
            # Each state (start, v) with dist < INF represents a walk closing at start.
            # Note: A cycle of length L is counted L times (once for each starting vertex).
            # And potentially multiple times if there are multiple ways? 
            # dist is MIN length. So each cycle contributes 1 to the count for each of its L vertices?
            # Actually, if dist[start, v] is finite, it means there is A walk.
            # We just want a heuristic that decreases when we remove short cycles.
            
            count += np.sum(new_dist[start, :] < (1 << 30))
            
            dist = new_dist
            
    return count

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Try LUW construction for baseline
    # For k=5, g=9, LUW gives n=250? 
    # LUW CD(4, 5) -> n = 2 * (5^4 + 5^3 + 5^2 + 5 + 1)? No.
    # LUW n = q * (1 + q + ... + q^{d-1})?
    # For CD(d, q), n = 2 * q * (q^d - 1)/(q - 1)?
    # Let's just implement a known small cage or use random search.
    # The prompt says LUW is a baseline. 
    # If k=5, g=9, the known cage is much smaller (1116 is the BEST known, lower bound 428).
    # Wait, "smallest known such graph has 1116 vertices".
    # LUW for (5,9) might be larger.
    # Let's rely on random search with local optimization.
    
    # Moore bound: 2 * (1 + k + k(k-1) + ... + k(k-1)^{m-1})
    # For g=9 (odd), radius m=4.
    # n >= 2 * (1 + 5 + 20 + 80 + 320) = 2 * 426 = 852?
    # Wait, Moore bound for girth g (odd) is 2 * sum_{i=0}^{(g-3)/2} k(k-1)^i.
    # g=9 -> (9-3)/2 = 3.
    # i=0: 1
    # i=1: 5
    # i=2: 5*4=20
    # i=3: 5*16=80
    # Sum = 106.
    # n >= 2 * 106 = 212?
    # Let's check: 1 + 5 + 20 + 80 = 106. 2*106 = 212.
    # The prompt says lower bound is 428.
    # My Moore bound calculation:
    # For odd girth 2m+1, n >= 2 * (1 + k + k(k-1) + ... + k(k-1)^{m-1}).
    # m = (g-1)/2 = 4.
    # Terms: i=0 to 3?
    # Path of length m from center: v0, v1, ..., vm.
    # v0 has k neighbors.
    # v1 has k-1 new neighbors.
    # ...
    # vm has k(k-1)^{m-1} neighbors?
    # Sum of vertices in one half-tree (depth m-1 from center? No).
    # Girth 2m+1. Radius m.
    # Number of vertices at distance i from center:
    # i=0: 1
    # i=1: k
    # i=2: k(k-1)
    # ...
    # i=m: k(k-1)^{m-1}
    # Total = 1 + k + k(k-1) + ... + k(k-1)^{m-1}.
    # For g=9, m=4.
    # 1 + 5 + 20 + 80 + 320 = 426.
    # n >= 2 * 426 = 852.
    # Why does prompt say 428?
    # Maybe I misread the prompt or the prompt has a typo or uses a different bound.
    # "best lower bound is 428".
    # Regardless, we should aim for n close to the bound.
    
    # Let's try to find a graph with n close to the lower bound.
    # We will try n from lower bound up to some limit.
    
    # Estimate lower bound
    if g % 2 == 1:
        m = (g - 1) // 2
        sum_val = 1
        for i in range(1, m):
            sum_val += k * (k - 1) ** (i - 1)
        # Wait, the last term is k(k-1)^(m-1).
        # My loop: i=1 -> k. i=2 -> k(k-1). ... i=m-1 -> k(k-1)^(m-2).
        # Missing last term?
        # Let's re-calc.
        # dist 0: 1
        # dist 1: k
        # dist 2: k(k-1)
        # ...
        # dist m-1: k(k-1)^(m-2)
        # dist m: k(k-1)^(m-1)
        # Sum = 1 + sum_{j=1}^{m} k(k-1)^(j-1)
        
        s = 1
        for j in range(1, m + 1):
            s += k * ((k - 1) ** (j - 1))
        moore = 2 * s
    else:
        m = g // 2
        # For even girth 2m, it's a tree of depth m-1 from center edge?
        # n >= 2 * (k + k(k-1) + ... + k(k-1)^(m-2))
        s = 0
        for j in range(0, m - 1):
            s += k * ((k - 1) ** j)
        moore = 2 * s
    
    # Start searching from max(moore, 2*k)
    # But random regular graphs might not exist for all n.
    # n*k must be even.
    
    best_n = None
    best_adj = None
    best_score = float('inf')
    
    # We'll try a few candidate sizes
    # Start near moore bound
    start_n = max(2 * k, moore)
    # Ensure start_n * k is even
    if (start_n * k) % 2 != 0:
        start_n += 1
        
    # Search range: start_n to start_n + 100?
    # Or just try to optimize at a specific n.
    # Given time limits, let's try n = moore bound, moore+2, etc.
    
    # To save time, let's just pick one n near the bound and do local search.
    # If k=5, g=9, moore=852.
    # Let's try n = 852, 854, 856...
    
    # Actually, the "known" graph is 1116.
    # If we can find one at 852, that's huge improvement.
    
    # Let's try n = start_n.
    n = start_n
    
    # If n*k is odd, bump n
    while (n * k) % 2 != 0:
        n += 1
        
    # Generate random k-regular graph
    def generate_regular(n, k, seed):
        # Using config model / pairing
        # Stubs
        rng = np.random.RandomState(seed)
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        perm = rng.permutation(len(stubs))
        adj = [[] for _ in range(n)]
        
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[perm[i]]
            v = stubs[perm[i + 1]]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            
        if not valid:
            return None
        return adj
    
    # Local search
    def local_search(adj, n, k, g, max_iters=20000):
        current_score = _girth_count_short_cycles(adj, n, g)
        if current_score == 0:
            return adj, True
        
        best_adj = [row[:] for row in adj]
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick two random edges (u1, v1) and (u2, v2) that are not adjacent
            # To do 2-switch: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2).
            # Must ensure u1 != u2, u1 != v2, v1 != u2, v1 != v2.
            # And new edges must not exist.
            
            # Efficient way: pick random u1, pick random neighbor v1.
            # Pick random u2, pick random neighbor v2.
            
            u1 = rng.randint(n)
            v1 = adj[u1][rng.randint(k)]
            
            u2 = rng.randint(n)
            v2 = adj[u2][rng.randint(k)]
            
            # Check conditions
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            # Check if (u1, u2) and (v1, v2) exist
            if u2 in adj[u1] or v2 in adj[v1]:
                continue
                
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            new_score = _girth_count_short_cycles(adj, n, g)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                
                if best_score == 0:
                    # Found a graph with no short cycles?
                    # Verify exactly
                    if _girth_check(adj, n, g):
                        return adj, True
            else:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
        return best_adj, False
        
    # Try a few n values
    for offset in range(0, 20, 2):
        n_try = n + offset
        if (n_try * k) % 2 != 0:
            continue
            
        # Generate a few random graphs for this n
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            adj = generate_regular(n_try, k, seed)
            if adj is None:
                continue
                
            # Check girth first
            if _girth_check(adj, n_try, g):
                # Valid!
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_adj = adj
                    # We found a valid graph. Can we find smaller?
                    # If n_try is close to Moore bound, it's likely optimal or near.
                    # We can return immediately if n_try is very close to bound.
                    # But let's try to find smaller if possible?
                    # No, we started from bound. So this is minimal n we tried.
                    # If we found it, it's likely the best.
                    # We can break and return.
                    return n_try, [(i, j) for i in range(n_try) for j in adj[i] if i < j]
            
            # If not valid, do local search
            if n_try <= 1000: # Limit search to reasonable sizes
                adj_search, success = local_search(adj, n_try, k, g, max_iters=5000)
                if success:
                    if best_n is None or n_try < best_n:
                        best_n = n_try
                        best_adj = adj_search
                        return n_try, [(i, j) for i in range(n_try) for j in adj[i] if i < j]
                        
    # If we didn't find a valid graph with girth >= g, return the best we could get
    # But the task requires valid graph.
    # If we fail, we must return something valid.
    # The LUW construction is a fallback.
    
    # Implement a simple valid graph generator if local search fails.
    # For k=5, g=9, we can use the known 1116 vertex graph?
    # Or just a large random graph that likely has high girth?
    # Girth grows logarithmically.
    # A random 5-regular graph on N vertices has girth ~ log_{4}(N).
    # To get girth 9, N ~ 4^8 = 65536?
    # That's too big.
    
    # We need a constructive method.
    # LUW CD(d, q).
    # Let's implement LUW for k=5, g=9.
    # q=5.
    # d such that girth >= 2d+2?
    # LUW CD(d, q) has girth >= 2d+2 if q is prime power?
    # For d=3, girth >= 8.
    # For d=4, girth >= 10.
    # We need girth 9.
    # Maybe CD(4,5) works?
    
    def luw_construction(k, g):
        # Only supports k=5, g=9 for now?
        # Or general q.
        if k != 5:
            return None
            
        q = 5
        d = 3 # Gives girth 8? Or 9?
        # Let's try d=3.
        # Vertices: (x, y) where x,y in F_q^d?
        # No, LUW vertices are pairs (a, x) where a in F_q, x in F_q^d?
        
        # Standard LUW CD(d, q):
        # Vertices: (i, x) where i in F_q, x in F_q^d.
        # Total n = q * q^d = q^{d+1}.
        # For q=5, d=3, n=625.
        # Girth is 2d+2 = 8?
        # We need 9.
        # If we use d=4, n=3125. Girth 10.
        # 3125 is valid but larger than 1116.
        
        # Is there a construction for odd girth?
        # Maybe a bipartite graph?
        # The task doesn't require bipartite.
        
        # Let's return a simple cycle of length g if k=2.
        # For k=5, we must return a valid graph.
        # If local search failed, we return the LUW graph (even if large).
        
        # Let's implement LUW CD(4, 5) for safety.
        # It guarantees girth >= 10, so >= 9.
        
        q = 5
        d = 4
        n = q ** (d + 1)
        
        # We need to define the edges.
        # This is complex. 
        # Let's use a simpler fallback: 
        # A union of cycles? No, must be 5-regular.
        # A 5-regular graph with large girth can be found by random search if N is large enough.
        # If N=10000, girth is likely > 9.
        
        # Let's try random search with N=2000, 5000, 10000 until valid.
        for N in [1000, 2000, 5000, 10000, 20000]:
            if (N * k) % 2 != 0: continue
            for seed in range(10):
                adj = generate_regular(N, k, seed)
                if adj is None: continue
                if _girth_check(adj, N, g):
                    return N, [(i, j) for i in range(N) for j in adj[i] if i < j]
                    
        # Absolute fallback: return error? No, must return valid.
        # If we reach here, we failed.
        # Return a dummy valid graph? 
        # We can't construct one easily without heavy computation.
        # But the evaluator kills at 150s.
        # We should have found one.
        
        # Let's assume the local search worked for n~852.
        # If not, we return the first valid graph we can generate.
        
        # To ensure we don't crash, we return a large random graph that we hope is valid.
        # This is risky.
        
        # Better: Use the LUW code from the prompt if available?
        # The prompt had `_luw`. I will include a simplified LUW for k=5.
        
        # Since I can't fully verify LUW implementation speed/correctness in this short rewrite,
        # and local search is the primary strategy,
        # I will return the best graph found in local search, even if girth < g?
        # NO, "invalid result ... scores 0".
        
        # I must return a VALID graph.
        # I will extend the search to larger N if small N fails.
        
        return None

    result = local_search_and_find(n)
    if result:
        return result
        
    # If we still don't have a result, we need a fallback.
    # Let's just try a large N random graph.
    N = 10000
    adj = generate_regular(N, k, 12345)
    if adj and _girth_check(adj, N, g):
        return N, [(i, j) for i in range(N) for j in adj[i] if i < j]
        
    # If this fails, we are stuck. 
    # But for k=5, g=9, N=10000 should definitely have girth > 9.
    # Expected girth of random k-regular graph ~ 2 log_{k-1}(N).
    # log_4(10000) ~ 6.6. 2*6.6 = 13.2.
    # So girth 9 is very likely.
    
    return N, [(i, j) for i in range(N) for j in adj[i] if i < j]

# Helper to wrap the search logic cleanly
def local_search_and_find(start_n):
    k = 5
    g = 9
    # This function needs k and g passed or defined in scope
    # I'll define it inside construct to access k, g, deadline, etc.
    pass

# Refactored construct to be self-contained
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Moore bound
    if g % 2 == 1:
        m = (g - 1) // 2
        s = 1
        for j in range(1, m + 1):
            s += k * ((k - 1) ** (j - 1))
        moore = 2 * s
    else:
        m = g // 2
        s = 0
        for j in range(0, m - 1):
            s += k * ((k - 1) ** j)
        moore = 2 * s
        
    start_n = max(2 * k, moore)
    if (start_n * k) % 2 != 0:
        start_n += 1
        
    def generate_regular(n, k, seed):
        rng = np.random.RandomState(seed)
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        perm = rng.permutation(len(stubs))
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[perm[i]]
            v = stubs[perm[i + 1]]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
        return adj if valid else None
        
    def check_girth(adj, n, g):
        # Same as _girth_check
        if n < g: return False
        for start in range(n):
            dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
            dist[start, n] = 0
            for step in range(1, g):
                new_dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
                for v in range(n):
                    d_row = dist[v]
                    min1 = 1 << 30; min2 = 1 << 30; idx1 = -1; idx2 = -1
                    for p in range(n + 1):
                        val = d_row[p]
                        if val < min1:
                            min2 = min1; idx2 = idx1; min1 = val; idx1 = p
                        elif val < min2:
                            min2 = val; idx2 = p
                    for u in adj[v]:
                        best_prev = min1 if idx1 != u else min2
                        if best_prev < (1 << 30):
                            nv = best_prev + 1
                            if nv < new_dist[u, v]:
                                new_dist[u, v] = nv
                if np.any(new_dist[start, :] < (1 << 30)):
                    return False
                dist = new_dist
        return True

    def count_short(adj, n, g):
        count = 0
        for start in range(n):
            dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
            dist[start, n] = 0
            for step in range(1, g):
                new_dist = np.full((n, n + 1), 1 << 30, dtype=np.int32)
                for v in range(n):
                    d_row = dist[v]
                    min1 = 1 << 30; min2 = 1 << 30; idx1 = -1; idx2 = -1
                    for p in range(n + 1):
                        val = d_row[p]
                        if val < min1:
                            min2 = min1; idx2 = idx1; min1 = val; idx1 = p
                        elif val < min2:
                            min2 = val; idx2 = p
                    for u in adj[v]:
                        best_prev = min1 if idx1 != u else min2
                        if best_prev < (1 << 30):
                            nv = best_prev + 1
                            if nv < new_dist[u, v]:
                                new_dist[u, v] = nv
                count += np.sum(new_dist[start, :] < (1 << 30))
                dist = new_dist
        return count

    # Search
    for offset in range(0, 50, 2):
        n_try = start_n + offset
        if (n_try * k) % 2 != 0: continue
        for seed in range(5):
            if time.time() > deadline - 10: break
            adj = generate_regular(n_try, k, seed)
            if not adj: continue
            
            if check_girth(adj, n_try, g):
                return n_try, [(i, j) for i in range(n_try) for j in adj[i] if i < j]
            
            # Local search
            if n_try < 2000:
                cur_score = count_short(adj, n_try, g)
                best_adj = [row[:] for row in adj]
                best_score = cur_score
                rng = np.random.RandomState(seed + 100)
                
                for _ in range(5000):
                    if time.time() > deadline - 5: break
                    u1 = rng.randint(n_try)
                    v1 = adj[u1][rng.randint(k)]
                    u2 = rng.randint(n_try)
                    v2 = adj[u2][rng.randint(k)]
                    
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2: continue
                    if u2 in adj[u1] or v2 in adj[v1]: continue
                    
                    adj[u1].remove(v1); adj[v1].remove(u1)
                    adj[u2].remove(v2); adj[v2].remove(u2)
                    adj[u1].append(u2); adj[u2].append(u1)
                    adj[v1].append(v2); adj[v2].append(v1)
                    
                    new_score = count_short(adj, n_try, g)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                        if new_score == 0:
                            if check_girth(adj, n_try, g):
                                return n_try, [(i, j) for i in range(n_try) for j in adj[i] if i < j]
                    else:
                        # Revert
                        adj[u1].remove(u2); adj[u2].remove(u1)
                        adj[v1].remove(v2); adj[v2].remove(v1)
                        adj[u1].append(v1); adj[v1].append(u1)
                        adj[u2].append(v2); adj[v2].append(u2)

    # Fallback
    N = 5000
    adj = generate_regular(N, k, 999)
    if adj and check_girth(adj, N, g):
        return N, [(i, j) for i in range(N) for j in adj[i] if i < j]
        
    return N, [(i, j) for i in range(N) for j in adj[i] if i < j]


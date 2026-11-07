import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    start_time = time.time()
    deadline = start_time + 85 # 5s margin

    # Precompute valid bipartite edges
    # Path P24 is bipartite with partitions E (even) and O (odd)
    # Any edge added must connect E to O to preserve bipartiteness (and thus triangle-freeness)
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if u % 2 != v % 2:
                valid_edges.append((u, v))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_diameter_and_fars(adj):
        max_dist = 0
        far_pairs = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        if dist[v] > 4:
                            # Record far pair (sorted)
                            a, b = min(src, v), max(src, v)
                            far_pairs.append((a, b))
                        queue.append(v)
            if -1 in dist:
                return float('inf'), []
        return max_dist, far_pairs

    def is_valid(added_edges):
        adj = build_adj(added_edges)
        diam, _ = get_diameter_and_fars(adj)
        return diam <= 4

    def greedy_build(randomize=False):
        added_edges = set()
        adj = build_adj(added_edges)
        
        while True:
            if time.time() > deadline:
                break
            
            diam, far_pairs = get_diameter_and_fars(adj)
            if diam <= 4:
                break
            
            if not far_pairs:
                break
                
            # Candidate edges: all valid bipartite edges not already added
            candidates = []
            for u, v in valid_edges:
                if (u, v) not in added_edges:
                    candidates.append((u, v))
            
            if not candidates:
                break

            best_edge = None
            best_score = -1
            best_new_diam = float('inf')
            
            # Shuffle if randomizing
            if randomize:
                random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                
                new_set = added_edges | {(u, v)}
                new_adj = build_adj(new_set)
                new_diam, new_fars = get_diameter_and_fars(new_adj)
                
                # Score: Primary is reducing diameter, Secondary is reducing number of far pairs
                # If new_diam is inf, skip (disconnected, though unlikely in bipartite supergraph of connected path)
                if new_diam == float('inf'):
                    continue
                
                # We prefer edges that reduce the diameter.
                # If diameter doesn't reduce, prefer those that reduce the count of far pairs.
                score = 0
                if new_diam < diam:
                    score += 1000 - new_diam # Strong preference for diam reduction
                else:
                    # If diam is same, check far pairs reduction
                    old_far_count = len(far_pairs)
                    new_far_count = len(new_fars)
                    score += (old_far_count - new_far_count)
                
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
                    best_new_diam = new_diam
                
                # Early termination if we found an edge that makes diameter <= 4
                if new_diam <= 4:
                    best_edge = (u, v)
                    break
            
            if best_edge is None:
                # Fallback: pick any valid edge
                best_edge = candidates[0]
            
            added_edges.add(best_edge)
            adj = build_adj(added_edges)
            
        return added_edges

    def local_search_reduce(added_edges):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Iterate over a copy of the set to allow modification
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                new_set = added_edges - {edge}
                if is_valid(new_set):
                    added_edges = new_set
                    improved = True
                    break
        return added_edges

    best_result = None
    best_count = float('inf')
    
    random.seed(42)
    
    # Try multiple restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        res = greedy_build(randomize=(attempt > 0))
        
        if not is_valid(res):
            continue
            
        res = local_search_reduce(res)
        
        if len(res) < best_count:
            best_count = len(res)
            best_result = res
            
        if best_count <= 8: # Good enough?
            break

    if best_result is None:
        # Fallback: Construct a known valid bipartite solution
        # Add edges (i, i+5) for all i where parity matches (always does for odd offset)
        # This creates a dense bipartite graph, definitely valid.
        # Then reduce.
        fallback_edges = set()
        for i in range(n):
            j = i + 5
            if j < n:
                if i % 2 != j % 2:
                    fallback_edges.add((i, j))
        # Also add (i, i+7) etc if needed? 
        # Actually, let's just run one more greedy with a different seed if this fails
        res = greedy_build(randomize=False)
        if is_valid(res):
             res = local_search_reduce(res)
             return [[int(u), int(v)] for u, v in res]
        
        # Absolute fallback: Return a hardcoded small set if possible
        # Or return the fallback_edges reduced
        res = local_search_reduce(fallback_edges)
        return [[int(u), int(v)] for u, v in res]

    return [[int(u), int(v)] for u, v in best_result]

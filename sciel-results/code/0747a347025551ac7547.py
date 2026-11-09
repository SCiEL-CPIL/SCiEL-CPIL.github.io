import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for safety
    
    n = 15
    max_possible_edges = n * (n - 1) // 2  # 105
    
    # Initialize with a random dense graph
    # Start with a high probability of edges to ensure high initial edge count
    current_edges = []
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.7:
                current_edges.append([u, v])
                edge_set.add((u, v))
    
    # Helper to compute degrees
    def get_degrees(edges):
        degs = [0] * n
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs
    
    # Helper to check validity and return penalty
    # Penalty > 0 means invalid. Penalty 0 means valid.
    # We want to minimize penalty and maximize edge count.
    # Composite score for SA: score = -len(edges) * 100 + penalty
    # Actually, let's just track validity and edge count separately for the final result,
    # but for SA acceptance, we use an energy function.
    
    def calculate_penalty(degs, edge_set):
        # Check all paths s-u-v-t
        # We can optimize this by precomputing adjacency lists
        # But for n=15, direct check is fast enough.
        # adj[u] is set of neighbors
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        penalty = 0
        # Iterate over all edges (u, v)
        for u, v in edge_set:
            # s in adj[u], t in adj[v]
            # s != v, t != u, s != t
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if degs[s] == degs[t]:
                        penalty += 1
        return penalty

    current_degs = get_degrees(current_edges)
    current_penalty = calculate_penalty(current_degs, edge_set)
    current_score = -len(current_edges) * 1000 + current_penalty
    
    best_edges = None
    best_penalty = float('inf')
    best_edge_count = -1
    
    if current_penalty == 0:
        best_edges = [list(e) for e in current_edges]
        best_penalty = 0
        best_edge_count = len(current_edges)
        
    T = 1000.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Generate a neighbor state
        # Randomly add or remove an edge
        # To bias towards adding edges, we can try adding first
        # But we need to ensure we explore the space.
        
        # Pick a random pair
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Normalize
        if u > v:
            u, v = v, u
            
        if (u, v) in edge_set:
            # Remove edge
            new_edges = [e for e in current_edges if not (e[0] == u and e[1] == v)]
            new_edge_set = set(new_edges)
            new_degs = [d - 1 for d in current_degs]
            new_degs[u] -= 1
            new_degs[v] -= 1
        else:
            # Add edge
            new_edges = current_edges + [[u, v]]
            new_edge_set = edge_set | {(u, v)}
            new_degs = current_degs[:]
            new_degs[u] += 1
            new_degs[v] += 1
            
        new_penalty = calculate_penalty(new_degs, new_edge_set)
        new_score = -len(new_edges) * 1000 + new_penalty
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < pow(sys.float_info.epsilon, delta / T):
            current_edges = new_edges
            edge_set = new_edge_set
            current_degs = new_degs
            current_penalty = new_penalty
            current_score = new_score
            
            # Update best if valid
            if current_penalty == 0:
                if len(current_edges) > best_edge_count:
                    best_edge_count = len(current_edges)
                    best_edges = [list(e) for e in current_edges]
                    best_penalty = 0
                    # If we reach max possible, we can stop
                    if best_edge_count == max_possible_edges:
                        break
        else:
            # Reject, but we might still want to track if this rejected state was valid?
            # No, SA usually only updates best from accepted states or explicit checks.
            # However, if the rejected state was valid and had more edges, we should record it.
            if new_penalty == 0:
                if len(new_edges) > best_edge_count:
                    best_edge_count = len(new_edges)
                    best_edges = [list(e) for e in new_edges]
                    best_penalty = 0
                    if best_edge_count == max_possible_edges:
                        break

        T *= cooling_rate
        if T < 1e-4:
            # Reheat slightly to escape deep local minima if stuck
            if random.random() < 0.1:
                T = 500.0
            else:
                # If T is too low, we are effectively doing greedy hill climbing.
                # Let's keep it low but not zero to allow some escape.
                T = 1e-4

    if best_edges is None:
        # Fallback to a simple valid graph if SA failed completely
        # Star graph is always valid? 
        # Path s-u-v-t in star: center is u or v.
        # If u is center, s is leaf, v is leaf. t is leaf.
        # deg(s)=1, deg(t)=1. Equal! So star is INVALID.
        # We need a valid fallback.
        # A path graph P15?
        # Degrees: 1, 2, 2, ..., 2, 1.
        # Path s-u-v-t.
        # If u,v are internal (deg 2), s,t are neighbors.
        # If s is end (deg 1), t is internal (deg 2). 1 != 2. OK.
        # If s is internal (deg 2), t is end (deg 1). OK.
        # If s is internal (deg 2), t is internal (deg 2). 
        # Path s-u-v-t. u,v adjacent. s neighbor of u, t neighbor of v.
        # In a path graph, if u,v are internal, their other neighbors s,t are also internal (unless u or v is adjacent to an end).
        # Example: 0-1-2-3-4. Path 0-1-2-3. deg(0)=1, deg(3)=2. OK.
        # Path 1-2-3-4. deg(1)=2, deg(4)=1. OK.
        # Path 0-1-2-3-4... wait, 3-edge path.
        # Consider 1-2-3-4. s=1, u=2, v=3, t=4. deg(1)=2, deg(4)=2. INVALID.
        # So simple path is not always valid.
        
        # Let's just return an empty graph or a known small valid graph.
        # Actually, the task says "Maximize edge count".
        # If SA finds nothing, returning empty is bad.
        # Let's try a few random restarts if best is None.
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            # Random sparse graph
            edges = []
            eset = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.2:
                        edges.append([u, v])
                        eset.add((u, v))
            degs = get_degrees(edges)
            pen = calculate_penalty(degs, eset)
            if pen == 0:
                best_edges = edges
                best_edge_count = len(edges)
                break

    if best_edges is None:
        # Absolute fallback: Empty graph is valid (no paths).
        # But score will be 0.
        # Is there a simple valid graph?
        # A matching?
        # Path s-u-v-t requires 3 edges. Matching has no paths of length 3.
        # So a perfect matching is valid.
        # n=15, max matching size 7.
        # Edges: (0,1), (2,3), ..., (12,13). Vertex 14 isolated.
        # This is valid.
        edges = []
        for i in range(0, n - 1, 2):
            edges.append([i, i + 1])
        best_edges = edges

    # Ensure types are correct
    final_edges = []
    for u, v in best_edges:
        final_edges.append([int(u), int(v)])
        
    return final_edges

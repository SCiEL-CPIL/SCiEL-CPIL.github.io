import time
import random
from collections import defaultdict, deque

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 100.0  # 100 seconds to be safe against 120s limit
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(len(order) - 1):
            u, v = order[i], order[i+1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: trivial decomposition
            edges = instance_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance_graph(n, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        best_pieces = None
        best_count = len(edge_set) + 100 # Large initial value
        
        # Try multiple randomizations to find a better decomposition
        # The number of trials can be adjusted based on time
        max_trials = 200
        trial_seed = 42
        
        for _ in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            current_edges = set(edge_set)
            pieces = []
            
            # Build adjacency for the current free edges
            # We will modify this as we remove edges
            adj = defaultdict(list)
            for u, v in current_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Helper to remove an edge from adj and current_edges
            def remove_edge(u, v):
                if u > v: u, v = v, u
                current_edges.discard((u, v))
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)
            
            # Iteratively find cycles
            # To encourage finding cycles, we pick a random edge and try to complete a cycle
            # We repeat this until no more cycles can be found
            progress = True
            while progress and current_edges:
                if time.time() > deadline - 5.0:
                    break
                progress = False
                # Shuffle edges to randomize the order of selection
                # We pick a random edge from the current set
                # Note: iterating over a set while modifying it is bad, so we convert to list
                # But we only pick one, so it's okay to pick a random element from the set
                if not current_edges:
                    break
                
                # Pick a random edge to start the cycle search
                # Using a list copy for random choice is safer if we were to modify, 
                # but we just pick one.
                u, v = random.choice(list(current_edges))
                
                # Temporarily remove (u, v)
                remove_edge(u, v)
                
                # Try to find a path from u to v in the remaining graph
                # BFS/DFS
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue and not found:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    for neighbor in adj[curr]:
                        if neighbor not in parent:
                            parent[neighbor] = curr
                            queue.append(neighbor)
                
                if found:
                    # Reconstruct path
                    path = []
                    curr = v
                    while curr != u:
                        path.append(curr)
                        curr = parent[curr]
                    path.append(u)
                    path.reverse() # u -> ... -> v
                    
                    # The cycle is path + edge (u, v)
                    # Vertices in cycle: path
                    # Edges in cycle: consecutive pairs in path + (v, u) which is (u, v)
                    cycle_vertices = path
                    # Add cycle to pieces
                    # The piece is the list of vertices
                    pieces.append(cycle_vertices)
                    
                    # Remove all edges in the cycle from current_edges and adj
                    # Edges are (path[i], path[i+1]) for i in range(len(path)-1) and (path[-1], path[0])
                    # Note: path is u->...->v. The closing edge is v->u.
                    for i in range(len(path) - 1):
                        remove_edge(path[i], path[i+1])
                    # The edge (u, v) was already removed
                    # Wait, we removed (u, v) before the search.
                    # The path edges are (u, p1), (p1, p2)... (pk, v).
                    # We need to remove these path edges from current_edges and adj.
                    # We already removed (u, v).
                    # So we just remove the path edges.
                    # But wait, remove_edge modifies current_edges and adj.
                    # We must be careful not to double remove or remove non-existent.
                    # The path edges are definitely in current_edges because they were part of the BFS tree.
                    
                    progress = True
                else:
                    # No cycle found using (u, v) as the closing edge.
                    # Restore (u, v)
                    # We need to re-add it to current_edges and adj
                    current_edges.add((min(u, v), max(u, v)))
                    uu, vv = min(u, v), max(u, v)
                    adj[uu].append(vv)
                    adj[vv].append(uu)
                    # If we fail to find a cycle with a random edge, we might be stuck with bridges.
                    # To avoid infinite loops if no cycles exist, we can break if we've tried many edges.
                    # But the while loop condition `progress` ensures we only continue if we found a cycle.
                    # If we don't find a cycle, we break out of the while loop?
                    # No, the current logic breaks the inner while if no cycle is found?
                    # `progress` remains False, so the while loop terminates.
                    # This means we stop looking for cycles as soon as one random edge fails to close a cycle.
                    # This is suboptimal. We should try other edges.
                    # Let's restructure:
                    # Instead of `while progress`, we can have a loop that tries to find ANY cycle.
                    # If none found, break.
                    
            # After the cycle extraction loop, whatever is left in current_edges are single edges
            for u, v in current_edges:
                pieces.append([u, v])
            
            # Evaluate this decomposition
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces] # Deep copy
                
            # Randomize the next trial
            trial_seed += 1
            random.seed(trial_seed) # This affects the global random state, which is used in the loop
            
        if best_pieces is None:
            # Fallback if no cycles found or error
            best_pieces = [list(e) for e in edge_set]
            
        results.append(best_pieces)
        
    return results

# To avoid global random state issues interfering between trials or instances, 
# it's better to use a local Random instance.

def construct():
    start_time = time.time()
    deadline = 100.0
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(len(order) - 1):
            u, v = order[i], order[i+1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            edges = instance_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance_graph(n, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        best_pieces = None
        best_count = len(edge_set) + 100
        
        rng = random.Random(12345)
        
        max_trials = 100
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            current_edges = set(edge_set)
            pieces = []
            
            adj = defaultdict(list)
            for u, v in current_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            def remove_edge(u, v):
                if u > v: u, v = v, u
                current_edges.discard((u, v))
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)
            
            def add_edge(u, v):
                if u > v: u, v = v, u
                current_edges.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            # Try to find cycles
            # We iterate while we can find a cycle.
            # To make it more robust, we try to find a cycle by picking a random edge.
            # If we fail, we try another. If we fail many times in a row, we assume no more cycles.
            failures = 0
            max_failures = 10
            while current_edges and failures < max_failures:
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random edge
                u, v = rng.choice(list(current_edges))
                
                remove_edge(u, v)
                
                # Find path u to v
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue and not found:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    for neighbor in adj[curr]:
                        if neighbor not in parent:
                            parent[neighbor] = curr
                            queue.append(neighbor)
                
                if found:
                    failures = 0
                    path = []
                    curr = v
                    while curr != u:
                        path.append(curr)
                        curr = parent[curr]
                    path.append(u)
                    path.reverse()
                    
                    pieces.append(path)
                    
                    for i in range(len(path) - 1):
                        remove_edge(path[i], path[i+1])
                else:
                    failures += 1
                    add_edge(u, v)
                    
            for u, v in current_edges:
                pieces.append([u, v])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
            
        results.append(best_pieces)
        
    return results

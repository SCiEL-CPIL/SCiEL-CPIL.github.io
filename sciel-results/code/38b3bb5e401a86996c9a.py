import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        n, p, seed = spec
        edges = generate_graph(n, p, seed)
        
        best_decomp = None
        best_count = float('inf')
        
        # Determine restarts based on N
        # N=20 is fast, N=30 is slower
        if n <= 20:
            max_restarts = 500
        elif n <= 25:
            max_restarts = 200
        else:
            max_restarts = 100

        restarts_done = 0
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(r * 1000 + seed)
            decomp = greedy_decompose(edges, n, rng)
            count = len(decomp)
            
            if count < best_count:
                best_count = count
                best_decomp = decomp
                # Optional: Early stop if we reached a theoretical lower bound?
                # Lower bound is ceil(E / N) roughly, but cycles can be shorter.
                # A Hamiltonian cycle is 1 piece. If we find 1 piece, we can stop.
                if count == 1:
                    break
            restarts_done += 1

        if best_decomp is None:
            # Fallback: trivial decomposition
            best_decomp = [[u, v] for u, v in edges]
        
        results.append(best_decomp)

    return results

def generate_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    # Path edges
    for i in range(n - 1):
        u, v = order[i], order[i+1]
        edges.add((min(u,v), max(u,v)))
    
    # Random edges
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def greedy_decompose(edges, n, rng):
    remaining = set(edges)
    pieces = []
    
    while remaining:
        # Find a cycle
        cycle = find_cycle_heuristic(remaining, n, rng)
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.remove(e)
        else:
            # No cycle, take an edge
            # To make this deterministic but varied, we can pick a random edge
            # or the first one. Picking a random edge might help in restarts?
            # Actually, if no cycle exists, the graph is a forest.
            # Any edge removal is fine.
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.remove(e)
            
    return pieces

def find_cycle_heuristic(remaining, n, rng):
    if not remaining:
        return None
        
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    # Filter out nodes with no edges
    active_nodes = [i for i in range(n) if adj[i]]
    if not active_nodes:
        return None
        
    # Convert to lists and shuffle for randomization
    for node in active_nodes:
        adj[node] = list(adj[node])
        rng.shuffle(adj[node])
        
    best_cycle = None
    best_len = 0
    step_limit = 5000  # Limit DFS depth/steps to prevent hanging
    
    # Try to find longest cycle starting from each active node
    # We can stop early if we find a Hamiltonian cycle (len == n)
    
    for start in active_nodes:
        # DFS
        # We use a stack to avoid recursion limits and allow step counting
        # Path: list of nodes
        # Visited: set of nodes in current path
        # We need to track the path to extract the cycle
        
        # Since we want the LONGEST, and DFS explores depth-first,
        # we will update best_cycle whenever we find a valid cycle.
        
        # To make it faster, we can prune if len(path) + (n - len(visited)) <= best_len
        # But n - len(visited) is an upper bound on additional nodes.
        
        def dfs(u, path, visited):
            nonlocal best_cycle, best_len, step_limit
            
            if step_limit <= 0:
                return False # Signal to stop
            
            step_limit -= 1
            
            # Check if we can close the cycle
            if len(path) > 2 and start in adj[u]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            
            # Pruning
            # Max possible length = current length + (number of unvisited nodes)
            # If this is <= best_len, we can prune
            # Note: We don't need to visit ALL unvisited, just enough to beat best_len
            # But this is a safe upper bound.
            if len(path) + (n - len(visited)) <= best_len:
                return False
            
            for v in adj[u]:
                if v == start:
                    if len(path) > 2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    continue
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    if not dfs(v, path, visited):
                        # Step limit reached
                        path.pop()
                        visited.remove(v)
                        return False
                    path.pop()
                    visited.remove(v)
                    
            return True

        # Run DFS for this start
        # We reset step_limit for each start? Or global?
        # Global is better to ensure total time bound.
        # But if we reset, we might spend too much on one start.
        # Let's use a global step budget for the whole find_cycle_heuristic call.
        
        # Actually, the step_limit is local to this function call.
        # I will set a fresh step_limit for each start node to ensure fairness?
        # No, if one start is bad, we waste time.
        # Let's use a single step_limit for the whole function.
        
        # Wait, the code above defines step_limit outside.
        # I need to restructure to use a shared counter or pass it.
        # Let's use a class or closure variable that is reset per call to find_cycle_heuristic.
        
        # Re-implementing DFS with shared step counter
        steps_used = 0
        max_steps = 20000  # ~20k steps per cycle search is fast
        
        def dfs_shared(u, path, visited):
            nonlocal steps_used, best_cycle, best_len
            
            if steps_used >= max_steps:
                return False
            
            steps_used += 1
            
            if len(path) > 2 and start in adj[u]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
            
            if len(path) + (n - len(visited)) <= best_len:
                return False
            
            for v in adj[u]:
                if v == start:
                    if len(path) > 2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    continue
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    if not dfs_shared(v, path, visited):
                        path.pop()
                        visited.remove(v)
                        return False
                    path.pop()
                    visited.remove(v)
                    
            return True

        dfs_shared(start, [start], {start})
        
        if best_len == n:
            break
        if steps_used >= max_steps:
            # If we hit the limit, we might have a partial best_cycle
            # We should probably break and use what we have
            break

    return best_cycle

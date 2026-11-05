import time
import random
from collections import defaultdict, deque

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_cycle_heuristic(edges, n, max_attempts=10):
    """
    Finds a cycle in the graph defined by edges.
    Tries to find a longer cycle by sampling random edges and finding longest path between endpoints.
    Returns list of vertices in the cycle, or None.
    """
    if not edges:
        return None
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    best_cycle = None
    best_len = 0
    
    # Sample some edges to try to form cycles
    sampled_edges = random.sample(list(edges), min(len(edges), max_attempts))
    
    for u, v in sampled_edges:
        # Try to find longest path from u to v in graph without edge (u,v)
        # Simple DFS with pruning
        # State: (current_node, visited_set, path)
        # This can be exponential. For N=30, it might be risky.
        # Let's use a simpler heuristic: Just find ANY cycle using DFS.
        # The local search will improve it.
        pass
        
    # Fallback: Find any cycle using standard DFS
    visited = set()
    parent = {}
    
    def dfs(node):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True
            elif neighbor != parent.get(node, -1):
                # Back edge found
                # Reconstruct cycle
                cycle = [neighbor, node]
                curr = node
                while curr != neighbor:
                    curr = parent[curr]
                    cycle.append(curr)
                # cycle is [neighbor, node, ..., neighbor]
                # We need to reverse or adjust to get proper order
                # Actually, the path is neighbor -> ... -> node.
                # The cycle is node -> ... -> neighbor -> node?
                # Let's trace:
                # neighbor is ancestor of node.
                # Path: neighbor = p0, p1, ..., pk = node.
                # Cycle: p0, p1, ..., pk, p0.
                # My reconstruction:
                # Start with [neighbor, node].
                # curr = node.
                # Loop: curr = parent[node] = p_{k-1}. cycle.append(p_{k-1}).
                # ... until curr == neighbor.
                # So cycle list is [neighbor, node, p_{k-1}, ..., p1].
                # This is the path from neighbor to node, reversed?
                # No, parent[node] is the node that led to node.
                # So path is neighbor -> ... -> node.
                # The list is [neighbor, node, ..., ...].
                # We want the cycle in order.
                # Let's just return the set of vertices? No, we need the cycle structure to remove edges.
                # We can just return the list of vertices in the order they appear in the path from neighbor to node, plus the closing edge.
                # Path: v0=neighbor, v1, ..., vk=node.
                # Cycle: v0, v1, ..., vk, v0.
                # My list: [v0, vk, v_{k-1}, ..., v1].
                # Reverse it: [v1, ..., v_{k-1}, vk, v0].
                # This is a valid cycle.
                cycle.reverse()
                nonlocal best_cycle, best_len
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                return True
        return False

    for i in range(n):
        if i not in visited:
            parent[i] = -1
            if dfs(i):
                if best_cycle:
                    return best_cycle
                else:
                    # If DFS found a cycle, it should have set best_cycle
                    # But if it returned True, best_cycle is set.
                    # If it returned False, no cycle in this component.
                    pass
    
    return best_cycle

def greedy_decompose(edges, n):
    """
    Greedily extracts cycles from the edge set.
    """
    remaining_edges = set(edges)
    cycles = []
    
    while remaining_edges:
        cycle = find_cycle_heuristic(remaining_edges, n)
        if not cycle:
            break
        # Remove edges of the cycle from remaining_edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in remaining_edges:
                remaining_edges.remove(e)
        cycles.append(cycle)
        
    return cycles, list(remaining_edges)

def score(decomp):
    cycles, isolated = decomp
    return len(cycles) + len(isolated)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    results = []
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance_graph(n, p, seed)
        
        # Initial greedy
        cycles, isolated = greedy_decompose(edges, n)
        current_decomp = (cycles, isolated)
        current_score = score(current_decomp)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing 1 cycle
            for i in range(len(cycles)):
                c = cycles[i]
                new_iso = set(isolated)
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    new_iso.add(tuple(sorted((u, v))))
                
                new_cycles, new_isolated = greedy_decompose(list(new_iso), n)
                new_decomp = (new_cycles, new_isolated)
                new_score = score(new_decomp)
                
                if new_score < current_score:
                    current_decomp = new_decomp
                    current_score = new_score
                    cycles, isolated = current_decomp
                    improved = True
                    break # Restart the loop to pick up new state
            
            if not improved and len(cycles) >= 2:
                # Try removing 2 cycles
                for i in range(len(cycles)):
                    for j in range(i + 1, len(cycles)):
                        c1 = cycles[i]
                        c2 = cycles[j]
                        new_iso = set(isolated)
                        for c in (c1, c2):
                            for k in range(len(c)):
                                u = c[k]
                                v = c[(k + 1) % len(c)]
                                new_iso.add(tuple(sorted((u, v))))
                        
                        new_cycles, new_isolated = greedy_decompose(list(new_iso), n)
                        new_decomp = (new_cycles, new_isolated)
                        new_score = score(new_decomp)
                        
                        if new_score < current_score:
                            current_decomp = new_decomp
                            current_score = new_score
                            cycles, isolated = current_decomp
                            improved = True
                            break
                    if improved:
                        break
        
        # Format output
        # Pieces are either [u, v] or cycle vertices
        pieces = []
        for c in cycles:
            pieces.append(c)
        for e in isolated:
            pieces.append(list(e))
            
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import defaultdict

    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_heuristic(edges, n, max_attempts=20):
        if not edges:
            return None
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        best_cycle = None
        best_len = 0
        
        # Try to find a cycle using DFS
        visited = set()
        parent = {}
        
        def dfs(node):
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    parent[neighbor] = node
                    if dfs(neighbor):
                        return True
                elif neighbor != parent.get(node, -1):
                    # Back edge found
                    cycle = [neighbor, node]
                    curr = node
                    while curr != neighbor:
                        curr = parent[curr]
                        cycle.append(curr)
                    cycle.reverse()
                    nonlocal best_cycle, best_len
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    return True
            return False

        for i in range(n):
            if i not in visited:
                parent[i] = -1
                if dfs(i):
                    if best_cycle:
                        return best_cycle
        
        return best_cycle

    def greedy_decompose(edges, n):
        remaining_edges = set(edges)
        cycles = []
        
        while remaining_edges:
            cycle = find_cycle_heuristic(remaining_edges, n)
            if not cycle:
                break
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            cycles.append(cycle)
            
        return cycles, list(remaining_edges)

    def score(decomp):
        cycles, isolated = decomp
        return len(cycles) + len(isolated)

    start_time = time.time()
    deadline = start_time + 140.0
    
    results = []
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance_graph(n, p, seed)
        
        # Initial greedy
        cycles, isolated = greedy_decompose(edges, n)
        current_decomp = (cycles, isolated)
        current_score = score(current_decomp)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing 1 cycle
            for i in range(len(cycles)):
                c = cycles[i]
                new_iso = set(isolated)
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    new_iso.add(tuple(sorted((u, v))))
                
                new_cycles, new_isolated = greedy_decompose(list(new_iso), n)
                new_decomp = (new_cycles, new_isolated)
                new_score = score(new_decomp)
                
                if new_score < current_score:
                    current_decomp = new_decomp
                    current_score = new_score
                    cycles, isolated = current_decomp
                    improved = True
                    break
            
            if not improved and len(cycles) >= 2:
                # Try removing 2 cycles
                for i in range(len(cycles)):
                    for j in range(i + 1, len(cycles)):
                        c1 = cycles[i]
                        c2 = cycles[j]
                        new_iso = set(isolated)
                        for c in (c1, c2):
                            for k in range(len(c)):
                                u = c[k]
                                v = c[(k + 1) % len(c)]
                                new_iso.add(tuple(sorted((u, v))))
                        
                        new_cycles, new_isolated = greedy_decompose(list(new_iso), n)
                        new_decomp = (new_cycles, new_isolated)
                        new_score = score(new_decomp)
                        
                        if new_score < current_score:
                            current_decomp = new_decomp
                            current_score = new_score
                            cycles, isolated = current_decomp
                            improved = True
                            break
                    if improved:
                        break
        
        pieces = []
        for c in cycles:
            pieces.append(c)
        for e in isolated:
            pieces.append(list(e))
            
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END

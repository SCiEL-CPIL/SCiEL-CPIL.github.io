import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n, rng, num_trials=500):
        best_cycle = None
        best_len = 0
        
        for _ in range(num_trials):
            start = rng.randint(0, n - 1)
            if not adj[start]:
                continue
                
            path = [start]
            in_path = {start}
            
            stack = [(start, iter(adj[start]))]
            found = False
            
            while stack and not found:
                node, it = stack[-1]
                advanced = False
                for neighbor in it:
                    if neighbor in in_path:
                        idx = path.index(neighbor)
                        cycle = path[idx:]
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = list(cycle)
                        found = True
                        break
                    elif neighbor not in in_path:
                        if len(adj[neighbor]) <= 1:
                            continue
                        path.append(neighbor)
                        in_path.add(neighbor)
                        stack.append((neighbor, iter(adj[neighbor])))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    if path:
                        last = path.pop()
                        in_path.discard(last)
                        
        return best_cycle

    def decompose_graph(edges, n, rng):
        initial_edges = set(edges)
        best_pieces = None
        best_count = float('inf')
        
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
                
            current_edges = set(initial_edges)
            adj = {i: set() for i in range(n)}
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            pieces = []
            
            while current_edges:
                cycle = find_longest_cycle(adj, n, rng, num_trials=500)
                
                if cycle:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        current_edges.discard(edge)
                        adj[u].discard(v)
                        adj[v].discard(u)
                else:
                    u, v = next(iter(current_edges))
                    pieces.append([u, v])
                    current_edges.discard((u, v))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces
        
        return best_pieces if best_pieces is not None else [[u, v] for u, v in initial_edges]

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for idx, (n, p, seed) in enumerate(specs):
        if time.time() > deadline - 10.0:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(n, p, seed)
        rng = random.Random(seed + idx)
        pieces = decompose_graph(edges, n, rng)
        results.append(pieces)
        
    return results

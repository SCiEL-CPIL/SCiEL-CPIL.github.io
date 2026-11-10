import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    def get_cycle_lengths(edges, n=12):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def find_cycles_from(start):
            found = []
            def dfs(node, path, visited):
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(path) >= 3:
                        found.append(len(path))
                    elif neighbor not in visited and neighbor > start:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            dfs(start, [start], {start})
            return found
        
        for v in range(n):
            for c in find_cycles_from(v):
                cycle_lengths.add(c)
        
        return cycle_lengths
    
    def is_valid(edges, n=12):
        for u, v in edges:
            if u == v:
                return False
        
        edge_set = set()
        for u, v in edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d < 3:
                return False
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return True
            parent[px] = py
            return False
        
        has_cycle = False
        for u, v in edges:
            if union(u, v):
                has_cycle = True
                break
        
        return has_cycle
    
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges
    
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2
    
    # Try random 3-regular bipartite graphs on 12 vertices
    # Parts: A={0..5}, B={6..11}
    random.seed(42)
    
    for attempt in range(20000):
        if time.time() > deadline - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Start with base: a_i -> b_i, b_{i+1}, b_{i+2} mod 6
        # Apply random permutation to B side
        perm = list(range(6))
        random.shuffle(perm)
        
        edges = []
        for i in range(6):
            for j in [0, 1, 2]:
                a = i
                b = 6 + perm[(i + j) % 6]
                edges.append([a, b])
        
        edge_set = set()
        valid = True
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        edges = [list(e) for e in edge_set]
        
        if not is_valid(edges):
            continue
        
        cl = get_cycle_lengths(edges)
        if not cl:
            continue
        
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edges
            if score == 1:
                break
    
    return [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
# EVOLVE-BLOCK-END

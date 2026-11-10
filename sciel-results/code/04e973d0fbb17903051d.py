import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 12
    
    def get_cycle_lengths(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def find_cycles_from(start):
            found = []
            def dfs(node, path, visited):
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(path) >= 4:
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
    
    def is_valid(edges):
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
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
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has cycle (using Union-Find)
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
    
    # Start with known good: two disjoint K3,3 components
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2  # K3,3 has cycle lengths {4, 6}
    
    # === STRATEGY 1: Exhaustive search over 3-regular bipartite graphs ===
    # Parts: A={0..5}, B={6..11}
    # A 3-regular bipartite graph is defined by 3 disjoint perfect matchings
    # between A and B. There are 6! = 720 perfect matchings.
    # We need to choose 3 disjoint ones (as sets of edges).
    
    # Precompute all perfect matchings
    import itertools
    all_perms = list(itertools.permutations(range(6)))
    # Each permutation p defines matching: a_i -> b_{p[i]}
    
    # Check if two matchings are disjoint (no shared edge)
    def matchings_disjoint(p1, p2):
        # Edge (i, p1[i]) vs (i, p2[i]) - they share vertex i in A
        # Actually two matchings are disjoint as edge sets if for all i, p1[i] != p2[i]
        for i in range(6):
            if p1[i] == p2[i]:
                return False
        return True
    
    # For speed, precompute edge sets for each permutation
    perm_edge_sets = []
    for p in all_perms:
        es = frozenset((i, 6 + p[i]) for i in range(6))
        perm_edge_sets.append(es)
    
    # Find all triples of pairwise disjoint matchings
    # This is the key insight: enumerate all valid 3-regular bipartite graphs
    found_triples = 0
    checked = 0
    
    for i in range(720):
        if time.time() > deadline - 10:
            break
        pi = all_perms[i]
        ei = perm_edge_sets[i]
        
        for j in range(i + 1, 720):
            if time.time() > deadline - 10:
                break
            ej = perm_edge_sets[j]
            
            # Check disjoint
            if ei & ej:
                continue
            
            for k in range(j + 1, 720):
                ek = perm_edge_sets[k]
                
                if ei & ek or ej & ek:
                    continue
                
                # Valid triple found
                found_triples += 1
                edges = []
                for idx in [i, j, k]:
                    p = all_perms[idx]
                    for a in range(6):
                        edges.append([a, 6 + p[a]])
                
                # Convert to sorted unique edge list
                edge_set = set()
                for e in edges:
                    edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
                edges = [list(e) for e in edge_set]
                
                if len(edges) != 18:
                    continue
                
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
                        return [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
                
                checked += 1
    
    # === STRATEGY 2: Also try non-bipartite constructions ===
    # The Petersen graph has 10 vertices, girth 5, all cycles length 5
    # We can try to add 2 more vertices with degree >= 3
    
    # Try: Petersen graph (10 vertices) + 2 vertices connected to form degree >= 3
    # Petersen graph vertices 0-9
    petersen_edges = [
        [0,1],[1,2],[2,3],[3,4],[4,0],  # outer 5-cycle
        [5,6],[6,7],[7,8],[8,9],[9,5],  # inner 5-cycle
        [0,5],[1,6],[2,7],[3,8],[4,9]   # spokes
    ]
    
    # Need to add vertices 10, 11 with degree >= 3
    # Connect them to existing vertices to maintain triangle-free
    # Petersen is triangle-free, so we need to be careful
    
    # Try adding 10, 11 connected to 3 non-adjacent vertices each
    # This is complex, so try random approach
    
    random.seed(123)
    
    # Try random 3-regular graphs (not necessarily bipartite)
    for attempt in range(20000):
        if time.time() > deadline - 5:
            break
        
        # Generate random 3-regular graph on 12 vertices
        # Using configuration model
        degrees = [3] * 12
        half_edges = []
        for v in range(12):
            half_edges.extend([v] * 3)
        
        random.shuffle(half_edges)
        
        edges = []
        valid = True
        for i in range(0, 24, 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                valid = False
                break
            edges.append([u, v])
        
        if not valid:
            continue
        
        # Check no duplicate edges
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


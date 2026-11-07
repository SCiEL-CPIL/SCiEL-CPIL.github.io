from random import Random
import sys

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    # Initialize available edges
    initial_available_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            initial_available_edges.add((i, j))
            
    # We need to pack T20 down to T2. T1 is trivial.
    # Base indices for trees T2..T20 are 1..19
    base_indices = list(range(1, n))
    
    # Sort by size ascending
    base_indices.sort(key=lambda i: len(trees[i]))
    
    assignment = [None] * n
    assignment[0] = [0]  # T1
    
    # Try multiple times with different random seeds for host ordering
    for seed in range(100):
        rng = Random(seed)
        available_edges = set(initial_available_edges)
        assignment_attempt = [None] * n
        assignment_attempt[0] = [0]
        
        success = True
        
        for idx in base_indices:
            k = idx + 1
            tree_edges = trees[idx]
            if not tree_edges:
                continue
                
            adj = {i: [] for i in range(k)}
            deg = [0] * k
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                
            vertices = list(range(k))
            # Sort by degree descending to place high-degree nodes first
            vertices.sort(key=lambda x: deg[x], reverse=True)
            
            mapping = {}
            used_hosts = set()
            
            hosts = list(range(n))
            rng.shuffle(hosts)
            
            def backtrack(pos):
                if pos == len(vertices):
                    return True
                
                v = vertices[pos]
                # Try hosts in the shuffled order
                for h in hosts:
                    if h in used_hosts:
                        continue
                    
                    valid = True
                    for nb in adj[v]:
                        if nb in mapping:
                            host_nb = mapping[nb]
                            edge = (min(h, host_nb), max(h, host_nb))
                            if edge not in available_edges:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    mapping[v] = h
                    used_hosts.add(h)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    del mapping[v]
                    used_hosts.remove(h)
                
                return False
            
            if not backtrack(0):
                success = False
                break
            
            if success:
                assignment_attempt[idx] = [mapping[i] for i in range(k)]
                for u, v in tree_edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    available_edges.remove((min(hu, hv), max(hu, hv)))
        
        if success:
            return assignment_attempt
            
    # If all attempts fail, return the last partial attempt? 
    # The problem says "Any broken rule makes the whole result invalid".
    # So we must return a valid one. 
    # If we can't find a full packing, we should return a partial one that is valid?
    # "Entry k-1 is None or a list...". 
    # If we return a partial packing, the edges used must be disjoint.
    # The current code breaks on failure. 
    # We should track the best partial solution.
    
    best_assignment = [None] * n
    best_assignment[0] = [0]
    best_edges = 0
    
    for seed in range(100):
        rng = Random(seed)
        available_edges = set(initial_available_edges)
        assignment_attempt = [None] * n
        assignment_attempt[0] = [0]
        
        # We process in ascending order. If one fails, we stop.
        # But we want the max edges.
        # So we should try to pack as many as possible.
        
        for idx in base_indices:
            k = idx + 1
            tree_edges = trees[idx]
            if not tree_edges:
                continue
                
            adj = {i: [] for i in range(k)}
            deg = [0] * k
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                
            vertices = list(range(k))
            vertices.sort(key=lambda x: deg[x], reverse=True)
            
            mapping = {}
            used_hosts = set()
            
            hosts = list(range(n))
            rng.shuffle(hosts)
            
            def backtrack(pos):
                if pos == len(vertices):
                    return True
                
                v = vertices[pos]
                for h in hosts:
                    if h in used_hosts:
                        continue
                    
                    valid = True
                    for nb in adj[v]:
                        if nb in mapping:
                            host_nb = mapping[nb]
                            edge = (min(h, host_nb), max(h, host_nb))
                            if edge not in available_edges:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    mapping[v] = h
                    used_hosts.add(h)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    del mapping[v]
                    used_hosts.remove(h)
                
                return False
            
            if not backtrack(0):
                # Failed to embed this tree. 
                # We keep the previous successful embeddings.
                break
            
            assignment_attempt[idx] = [mapping[i] for i in range(k)]
            for u, v in tree_edges:
                hu = mapping[u]
                hv = mapping[v]
                available_edges.remove((min(hu, hv), max(hu, hv)))
                
        # Calculate score for this attempt
        current_edges = sum(len(trees[i]) for i in range(1, n) if assignment_attempt[i] is not None)
        if current_edges > best_edges:
            best_edges = current_edges
            best_assignment = assignment_attempt[:]
            
    return best_assignment

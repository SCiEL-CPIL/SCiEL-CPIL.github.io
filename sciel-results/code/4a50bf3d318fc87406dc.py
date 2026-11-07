import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); continue
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
    deadline = time.time() + 110
    trees = instance_trees(20, 611)
    
    all_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            all_edges.add((u, v))
            
    # We will try to find a full decomposition.
    # We use a randomized backtracking approach.
    
    def find_embedding(tree_edges, avail_edges, rng):
        k = len(set(v for e in tree_edges for v in e))
        if k == 0:
            return [0]
        
        adj = {i: [] for i in range(k)}
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        vertices = sorted(range(k), key=lambda x: -len(adj[x]))
        mapping = {}
        
        def backtrack(idx):
            if idx == len(vertices):
                return True
            tv = vertices[idx]
            candidates = list(range(20))
            
            # Filter by mapped neighbors
            for neighbor_tv in adj[tv]:
                if neighbor_tv in mapping:
                    host_neighbor = mapping[neighbor_tv]
                    candidates = [c for c in candidates if tuple(sorted((c, host_neighbor))) in avail_edges]
            
            # Filter by distinctness
            used_hosts = set(mapping.values())
            candidates = [c for c in candidates if c not in used_hosts]
            
            # Randomize order
            rng.shuffle(candidates)
            
            for host in candidates:
                mapping[tv] = host
                if backtrack(idx + 1):
                    return True
                del mapping[tv]
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    # Try multiple random restarts
    for attempt in range(1000):
        if time.time() > deadline:
            break
        rng = Random(attempt)
        used_edges = set()
        assignments = [None] * 20
        assignments[0] = [0] # T1
        
        success = True
        # Pack T19 down to T2
        for idx in range(18, 1, -1):
            if time.time() > deadline:
                success = False
                break
            t_edges = trees[idx]
            avail = all_edges - used_edges
            emb = find_embedding(t_edges, avail, rng)
            if emb is None:
                success = False
                break
            assignments[idx] = emb
            for u, v in t_edges:
                hu, hv = emb[u], emb[v]
                used_edges.add(tuple(sorted((hu, hv))))
                
        if not success:
            continue
            
        # Check T20
        t20_edges = trees[19]
        avail = all_edges - used_edges
        emb20 = find_embedding(t20_edges, avail, rng)
        if emb20 is not None:
            assignments[19] = emb20
            return assignments
            
    # If no full solution found, return best partial?
    # The problem implies we should return valid edge-disjoint embeddings.
    # If we didn't find a full one, we can return the partial one from the last attempt?
    # But we need to ensure it's valid.
    # Actually, if we break out of the loop, `assignments` might be from a failed attempt.
    # Let's just return None for unassigned trees in the last valid partial state?
    # Simpler: If we don't find a full solution, we return the seed solution (T20 only) to be safe?
    # No, we can return the partial packing from the last attempt that got furthest.
    
    # To be safe and simple, if we don't find a full solution, we return the seed solution.
    # But we can do better: return the partial packing from the last attempt.
    
    # Let's modify the loop to keep track of the best partial solution.
    
    best_assignments = [None] * 20
    best_assignments[19] = list(range(20)) # Seed solution
    best_edges = 19
    
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        rng = Random(attempt)
        used_edges = set()
        assignments = [None] * 20
        assignments[0] = [0]
        
        # Pack T19 down to T2
        for idx in range(18, 1, -1):
            if time.time() > deadline - 5:
                break
            t_edges = trees[idx]
            avail = all_edges - used_edges
            emb = find_embedding(t_edges, avail, rng)
            if emb is None:
                break
            assignments[idx] = emb
            for u, v in t_edges:
                hu, hv = emb[u], emb[v]
                used_edges.add(tuple(sorted((hu, hv))))
                
        # Check T20
        t20_edges = trees[19]
        avail = all_edges - used_edges
        emb20 = find_embedding(t20_edges, avail, rng)
        if emb20 is not None:
            assignments[19] = emb20
            return assignments
        
        # Update best partial
        current_edges = sum(len(trees[i]) for i in range(20) if assignments[i] is not None)
        if current_edges > best_edges:
            best_edges = current_edges
            best_assignments = assignments[:]
            
    return best_assignments

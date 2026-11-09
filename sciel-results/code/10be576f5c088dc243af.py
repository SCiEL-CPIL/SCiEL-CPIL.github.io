"""Packing trees T1..T20 into K20 using a model-and-optimize approach."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    
    # Total edges across all trees
    total_edges = sum(len(t) for t in trees)
    # K20 has 190 edges total
    
    def solve_one_restart(rng, order):
        """
        Embed trees in the given order using randomized backtracking.
        Returns list of 20 entries (None or list of ints) or None if failed.
        """
        n = 20
        used_edges = set()
        result = [None] * 20
        
        for k in order:
            tree_idx = k - 1
            edges_k = trees[tree_idx]
            
            if not edges_k:
                # T1: single vertex
                result[tree_idx] = [0]
                continue
            
            if k == 2:
                # Find an unused edge
                found = False
                for _ in range(5000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a != b and (min(a,b), max(a,b)) not in used_edges:
                        result[tree_idx] = [a, b]
                        used_edges.add((min(a,b), max(a,b)))
                        found = True
                        break
                if not found:
                    # Try systematic
                    for a in range(20):
                        for b in range(a+1, 20):
                            if (a, b) not in used_edges:
                                result[tree_idx] = [a, b]
                                used_edges.add((a, b))
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            # For tree with k vertices and k-1 edges
            # Use randomized backtracking with BFS ordering from a leaf
            # Build adjacency for the tree
            adj = [[] for _ in range(k)]
            for u, v in edges_k:
                adj[u].append(v)
                adj[v].append(u)
            
            # Find a leaf to start from
            leaf = None
            for v in range(k):
                if len(adj[v]) == 1:
                    leaf = v
                    break
            if leaf is None:
                leaf = 0
            
            # BFS ordering from leaf
            order_verts = []
            visited = [False] * k
            visited[leaf] = True
            queue = [leaf]
            while queue:
                node = queue.pop(0)
                order_verts.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # Backtracking with this ordering
            mapping = [None] * k
            mapping[leaf] = rng.randint(0, 19)
            
            def backtrack(pos):
                if time.time() > deadline - 5:
                    return False
                
                if pos == len(order_verts):
                    # Check all edges are valid
                    for u, v in edges_k:
                        eu, ev = mapping[u], mapping[v]
                        ekey = (min(eu, ev), max(eu, ev))
                        if ekey in used_edges:
                            return False
                    # Check injectivity
                    if len(set(mapping)) != k:
                        return False
                    # Save and add edges
                    for u, v in edges_k:
                        eu, ev = mapping[u], mapping[v]
                        used_edges.add((min(eu, ev), max(eu, ev)))
                    return True
                
                vertex = order_verts[pos]
                used_verts = set(mapping[:pos])
                
                # Get candidates
                candidates = []
                for v in range(20):
                    if v in used_verts:
                        continue
                    # Check if any edge from vertex to already-assigned neighbors would conflict
                    conflict = False
                    for nb in adj[vertex]:
                        if mapping[nb] is not None:
                            ekey = (min(v, mapping[nb]), max(v, mapping[nb]))
                            if ekey in used_edges:
                                conflict = True
                                break
                    if not conflict:
                        candidates.append(v)
                
                if not candidates:
                    return False
                
                rng.shuffle(candidates)
                
                for v in candidates:
                    mapping[vertex] = v
                    if backtrack(pos + 1):
                        return True
                    mapping[vertex] = None
                
                return False
            
            found = False
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                # Reset mapping
                mapping = [None] * k
                mapping[leaf] = rng.randint(0, 19)
                if backtrack(1):
                    found = True
                    break
            
            if not found:
                return None
            
            result[tree_idx] = list(mapping)
        
        return result
    
    # Try multiple restarts with different orderings and seeds
    best_result = None
    best_count = 0
    
    # First try: largest first
    order1 = list(range(20, 0, -1))
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        rng = Random(attempt * 1000 + 1)
        result = solve_one_restart(rng, order1)
        if result is not None:
            count = sum(1 for e in result if e is not None)
            if count > best_count:
                best_count = count
                best_result = result
                if best_count == 20:
                    break
    
    # If not all embedded, try smallest first
    if best_count < 20 and time.time() < deadline - 10:
        order2 = list(range(1, 21))
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            rng = Random(attempt * 1000 + 2)
            result = solve_one_restart(rng, order2)
            if result is not None:
                count = sum(1 for e in result if e is not None)
                if count > best_count:
                    best_count = count
                    best_result = result
                    if best_count == 20:
                        break
    
    # If still not all, try random orderings
    if best_count < 20 and time.time() < deadline - 10:
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            rng = Random(attempt * 1000 + 3)
            order = list(range(1, 21))
            rng.shuffle(order)
            result = solve_one_restart(rng, order)
            if result is not None:
                count = sum(1 for e in result if e is not None)
                if count > best_count:
                    best_count = count
                    best_result = result
                    if best_count == 20:
                        break
    
    # Convert to proper format
    final_result = []
    for i in range(20):
        if best_result is None or best_result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[i]])
    
    return final_result

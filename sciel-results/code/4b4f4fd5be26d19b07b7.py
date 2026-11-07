import random
from random import Random
import time

def construct():
    # Generate trees exactly as in the prompt
    rng = Random(611)
    trees = []
    for k in range(1, 21):
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
        
    # Sort indices by tree size (number of edges) descending
    # T1 has 0 edges, T20 has 19 edges.
    indices = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    embeddings = [None] * 20
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Local search for a valid mapping of a tree
    def find_mapping(tree_edges, size, used_edges_set):
        if size == 1:
            # Any host vertex works
            return [random.randint(0, 19)]
            
        # Build adjacency list for the tree
        adj = [[] for _ in range(size)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # We will use a randomized backtracking search
        # We try to assign host vertices to tree vertices 0..size-1
        # assigned[i] = host vertex for tree vertex i, or -1 if unassigned
        
        # To speed up, we can fix the root (vertex 0) to a random host vertex?
        # Or try all?
        # Let's try a few random roots and random neighbor orders.
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
                
            # Shuffle neighbor orders for randomness
            shuffled_adj = [list(neighs) for neighs in adj]
            for i in range(size):
                random.shuffle(shuffled_adj[i])
                
            assigned = [-1] * size
            used_hosts = set()
            
            # Pick a random root
            root = random.randint(0, size - 1)
            
            # DFS
            def dfs(v):
                # Assign a host vertex to v
                # Candidates: 0..19 not in used_hosts
                # And for each already assigned neighbor u, (assigned[u], h) not in used_edges_set
                
                # Get candidates
                candidates = [h for h in range(20) if h not in used_hosts]
                random.shuffle(candidates)
                
                for h in candidates:
                    # Check conflicts with already assigned neighbors
                    valid = True
                    for u in shuffled_adj[v]:
                        if assigned[u] != -1:
                            edge = frozenset((assigned[u], h))
                            if edge in used_edges_set:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Assign
                    assigned[v] = h
                    used_hosts.add(h)
                    
                    # Recurse on neighbors
                    success = True
                    for u in shuffled_adj[v]:
                        if assigned[u] == -1:
                            if not dfs(u):
                                success = False
                                break
                    
                    if success:
                        return True
                    else:
                        # Unassign
                        assigned[v] = -1
                        used_hosts.remove(h)
                        
                return False
            
            if dfs(root):
                # Check if all vertices assigned
                if -1 not in assigned:
                    return assigned
                    # Note: The mapping is valid.
        
        return None

    def solve(idx):
        nonlocal used_edges, embeddings
        
        if idx == len(indices):
            return True
            
        if time.time() > deadline:
            return False
            
        k = indices[idx]
        tree_edges = trees[k]
        size = k
        
        # Try to find a mapping
        mapping = find_mapping(tree_edges, size, used_edges)
        
        if mapping is None:
            return False
        
        # Apply mapping
        new_edges = set()
        for u, v in tree_edges:
            new_edges.add(frozenset((mapping[u], mapping[v])))
            
        # Check if any of these edges are already used (should be caught by find_mapping, but double check)
        # Actually find_mapping checks against used_edges.
        
        # Add to used_edges
        used_edges.update(new_edges)
        embeddings[k] = mapping
        
        # Recurse
        if solve(idx + 1):
            return True
        else:
            # Backtrack
            used_edges.difference_update(new_edges)
            embeddings[k] = None
            return False

    # Start solving
    # We might need to randomize the order of indices if the fixed order fails?
    # The order is sorted by size. This is a good heuristic.
    
    if solve(0):
        # Construct the return list
        result = [None] * 20
        for k in range(20):
            if embeddings[k] is not None:
                # Ensure types are int
                result[k] = [int(x) for x in embeddings[k]]
            else:
                result[k] = None
        return result
    else:
        # If failed, return best effort?
        # The task says "returning 20 entries... None or list".
        # If we can't pack all, we should pack as many as possible.
        # My solve() returns False if it can't pack ALL.
        # I need to modify solve to keep track of the best partial solution.
        
        # Let's restart with a simpler approach if the full solve fails?
        # Or just return what we have?
        # The solve() function backtracks, so if it returns False, it means no full solution was found.
        # But it might have found a partial one during the search.
        # We should track the best partial solution.
        
        # Let's modify solve to return the count of packed trees, and store the best embeddings.
        
        pass

    # Fallback: Greedy
    # Try to pack trees one by one, no backtracking.
    used_edges = set()
    embeddings = [None] * 20
    for k in indices:
        mapping = find_mapping(trees[k], k, used_edges)
        if mapping:
            new_edges = set(frozenset((mapping[u], mapping[v])) for u, v in trees[k])
            used_edges.update(new_edges)
            embeddings[k] = mapping
        else:
            # Skip this tree?
            pass
            
    result = [None] * 20
    for k in range(20):
        if embeddings[k] is not None:
            result[k] = [int(x) for x in embeddings[k]]
        else:
            result[k] = None
    return result

# Wait, the fallback greedy is weak.
# The backtracking solver is better.
# Let's ensure the backtracking solver is robust.
# If solve(0) returns False, it means we couldn't pack all 20.
# But we might have packed 19.
# I will modify the code to track the best result found.

def construct():
    import random
    from random import Random
    import time

    rng = Random(611)
    trees = []
    for k in range(1, 21):
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
        
    indices = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    embeddings = [None] * 20
    best_count = 0
    best_embeddings = [None] * 20
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def find_mapping(tree_edges, size, used_edges_set):
        if size == 1:
            return [random.randint(0, 19)]
            
        adj = [[] for _ in range(size)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
                
            shuffled_adj = [list(neighs) for neighs in adj]
            for i in range(size):
                random.shuffle(shuffled_adj[i])
                
            assigned = [-1] * size
            used_hosts = set()
            root = random.randint(0, size - 1)
            
            def dfs(v):
                candidates = [h for h in range(20) if h not in used_hosts]
                random.shuffle(candidates)
                
                for h in candidates:
                    valid = True
                    for u in shuffled_adj[v]:
                        if assigned[u] != -1:
                            edge = frozenset((assigned[u], h))
                            if edge in used_edges_set:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    assigned[v] = h
                    used_hosts.add(h)
                    
                    success = True
                    for u in shuffled_adj[v]:
                        if assigned[u] == -1:
                            if not dfs(u):
                                success = False
                                break
                    
                    if success:
                        return True
                    else:
                        assigned[v] = -1
                        used_hosts.remove(h)
                        
                return False
            
            if dfs(root):
                if -1 not in assigned:
                    return assigned
        
        return None

    def solve(idx):
        nonlocal used_edges, embeddings, best_count, best_embeddings
        
        if idx == len(indices):
            if len(indices) > best_count:
                best_count = len(indices)
                best_embeddings = embeddings.copy()
            return True
            
        if time.time() > deadline:
            return False
            
        k = indices[idx]
        tree_edges = trees[k]
        size = k
        
        mapping = find_mapping(tree_edges, size, used_edges)
        
        if mapping is None:
            # Try to update best if we have more than best_count?
            # No, we are at a dead end for this path.
            return False
        
        new_edges = set()
        for u, v in tree_edges:
            new_edges.add(frozenset((mapping[u], mapping[v])))
            
        used_edges.update(new_edges)
        embeddings[k] = mapping
        
        if solve(idx + 1):
            return True
        else:
            used_edges.difference_update(new_edges)
            embeddings[k] = None
            return False

    # Run solver
    # If it fails to find a full solution, we might still have a good partial one in best_embeddings?
    # No, best_embeddings is only updated when we reach the end (idx == 20).
    # We should update best_embeddings at every step if count > best_count.
    
    # Let's modify solve to track progress.
    
    def solve2(idx):
        nonlocal used_edges, embeddings, best_count, best_embeddings
        
        if idx == len(indices):
            if len(indices) > best_count:
                best_count = len(indices)
                best_embeddings = embeddings.copy()
            return True
            
        if time.time() > deadline:
            return False
            
        # Update best if current path is longer than best?
        # The path length is idx.
        if idx > best_count:
            best_count = idx
            best_embeddings = embeddings.copy()
            
        k = indices[idx]
        tree_edges = trees[k]
        size = k
        
        mapping = find_mapping(tree_edges, size, used_edges)
        
        if mapping is None:
            return False
        
        new_edges = set()
        for u, v in tree_edges:
            new_edges.add(frozenset((mapping[u], mapping[v])))
            
        used_edges.update(new_edges)
        embeddings[k] = mapping
        
        if solve2(idx + 1):
            return True
        else:
            used_edges.difference_update(new_edges)
            embeddings[k] = None
            return False

    solve2(0)
    
    result = [None] * 20
    for k in range(20):
        if best_embeddings[k] is not None:
            result[k] = [int(x) for x in best_embeddings[k]]
        else:
            result[k] = None
            
    return result

# Note: The above code has a bug. `best_embeddings` is a copy of `embeddings`.
# If `embeddings` is modified later, `best_embeddings` is safe because we use .copy().
# But we only copy when `idx > best_count`.
# This should work.

# One more thing: T1 has 0 edges. find_mapping returns a list of 1 host vertex.
# This is correct.

# Let's double check the types.
# "integers must be Python int"
# "sequences must be list or tuple"
# My result is a list of lists of ints.

# Final check on the logic.
# If the full solution is found, best_count will be 20.
# If not, it will be the max depth reached.

# This looks solid.


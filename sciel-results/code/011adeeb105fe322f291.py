import time
import sys
from itertools import permutations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0 # 20s margin
    
    # 1. Generate Trees
    def instance_trees(n=20, seed=611):
        from random import Random
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

    trees = instance_trees()
    
    # Result structure
    result = [None] * 20
    
    # 2. Fix T20 to identity
    # T20 is trees[19]
    # Vertices are 0..19
    result[19] = list(range(20))
    edges_20 = set()
    for u, v in trees[19]:
        edges_20.add(frozenset((u, v)))
        
    # 3. Fix T1 to vertex 0
    # T1 is trees[0]
    result[0] = [0]
    
    # 4. Define Available Edges
    # Total edges in K20
    all_edges = set()
    for i in range(20):
        for j in range(i+1, 20):
            all_edges.add(frozenset((i, j)))
            
    available_edges = all_edges - edges_20
    used_edges = set()
    
    # Trees to embed: T2..T19 (indices 1..18)
    # Sort by size descending to prune early
    tree_indices = list(range(1, 19))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Helper: Check if an embedding is valid
    def is_valid_embedding(k, mapping, current_used_edges):
        # mapping: list of length k, mapping vertex i to host vertex
        # Check injectivity
        if len(set(mapping)) != k:
            return False
        # Check edges
        for u, v in trees[k]:
            e = frozenset((mapping[u], mapping[v]))
            if e in current_used_edges:
                return False
            if e in edges_20:
                return False
        return True

    # CSP Solver for a single tree k
    def solve_tree(k, available_edges_set):
        # Variables: 0..k-1
        # Order by degree in T_k
        deg = [0] * k
        for u, v in trees[k]:
            deg[u] += 1
            deg[v] += 1
        vertices = list(range(k))
        vertices.sort(key=lambda v: deg[v], reverse=True)
        
        # Domains: 0..19
        # We will use backtracking
        assignment = {}
        
        def backtrack(idx):
            if idx == len(vertices):
                return True
            
            v = vertices[idx]
            
            # Get neighbors already assigned
            neighbors_assigned = []
            for u, w in trees[k]:
                if u == v and w in assignment:
                    neighbors_assigned.append(assignment[w])
                elif w == v and u in assignment:
                    neighbors_assigned.append(assignment[u])
            
            # Try all possible host vertices
            # Optimization: Only try vertices that don't conflict with assigned neighbors
            # and don't use used edges
            
            # To speed up, we can pre-filter candidates
            # But k is small, so simple loop is okay
            
            for host in range(20):
                if host in assignment.values():
                    continue
                
                # Check edges to assigned neighbors
                valid = True
                for nh in neighbors_assigned:
                    e = frozenset((host, nh))
                    if e in available_edges_set or e in edges_20:
                        # Wait, available_edges_set is the set of edges NOT in edges_20 and NOT in used_edges?
                        # No, available_edges_set passed in is the global available minus used.
                        # Let's pass the set of FORBIDDEN edges.
                        pass 
                
                # Let's restructure: pass forbidden_edges set.
                # Forbidden = edges_20 U used_edges_so_far
                # If e in forbidden, invalid.
                
                # Re-do logic inside backtrack with forbidden set
                pass
                
            return False

        # Better CSP implementation
        forbidden_edges = edges_20 | used_edges
        
        def backtrack2(idx):
            if idx == len(vertices):
                return True
            
            v = vertices[idx]
            
            # Determine valid hosts
            # A host h is valid if:
            # 1. h is not used by other vertices in this tree
            # 2. For every neighbor u of v in T_k that is already assigned to h_u,
            #    the edge {h, h_u} is NOT in forbidden_edges
            
            used_hosts = set(assignment.values())
            
            for h in range(20):
                if h in used_hosts:
                    continue
                
                valid = True
                # Check constraints with assigned neighbors
                for u, w in trees[k]:
                    if u == v and w in assignment:
                        h_u = assignment[w]
                        e = frozenset((h, h_u))
                        if e in forbidden_edges:
                            valid = False
                            break
                    elif w == v and u in assignment:
                        h_u = assignment[u]
                        e = frozenset((h, h_u))
                        if e in forbidden_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                assignment[v] = h
                
                if backtrack2(idx + 1):
                    return True
                
                # Unassign
                del assignment[v]
                
            return False

        if backtrack2(0):
            # Build mapping list
            mapping = [assignment[i] for i in range(k)]
            return mapping
        return None

    # Main Search
    def search(idx):
        nonlocal used_edges
        
        if time.time() > deadline:
            return False
            
        if idx == len(tree_indices):
            return True
            
        k = tree_indices[idx]
        
        # Solve for tree k
        # We need to find an embedding that doesn't conflict with used_edges
        # The solver above uses 'forbidden_edges' which is global.
        # We need to pass the current used_edges.
        
        # Modify solver to take forbidden set
        def solve_tree_with_forbidden(k, forbidden):
            deg = [0] * k
            for u, v in trees[k]:
                deg[u] += 1
                deg[v] += 1
            vertices = list(range(k))
            vertices.sort(key=lambda v: deg[v], reverse=True)
            
            assignment = {}
            
            def backtrack(idx):
                if idx == len(vertices):
                    return True
                
                v = vertices[idx]
                used_hosts = set(assignment.values())
                
                # Heuristic: try hosts that have fewer conflicts?
                # Just iterate 0..19
                
                for h in range(20):
                    if h in used_hosts:
                        continue
                    
                    valid = True
                    for u, w in trees[k]:
                        if u == v and w in assignment:
                            h_u = assignment[w]
                            e = frozenset((h, h_u))
                            if e in forbidden:
                                valid = False
                                break
                        elif w == v and u in assignment:
                            h_u = assignment[u]
                            e = frozenset((h, h_u))
                            if e in forbidden:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    assignment[v] = h
                    if backtrack(idx + 1):
                        return True
                    del assignment[v]
                
                return False
            
            if backtrack(0):
                return [assignment[i] for i in range(k)]
            return None
        
        mapping = solve_tree_with_forbidden(k, forbidden_edges)
        
        if mapping is None:
            return False
        
        # Update used edges
        new_edges = set()
        for u, v in trees[k]:
            e = frozenset((mapping[u], mapping[v]))
            new_edges.add(e)
        
        # Check if these edges are actually disjoint (should be by solver, but verify)
        if new_edges & forbidden_edges:
            return False
            
        used_edges.update(new_edges)
        result[k] = mapping
        
        if search(idx + 1):
            return True
        
        # Backtrack
        used_edges.difference_update(new_edges)
        result[k] = None
        
        return False

    # Start search
    if not search(0):
        # Fallback to seed solution
        return [None] * 19 + [list(range(20))]
        
    return result

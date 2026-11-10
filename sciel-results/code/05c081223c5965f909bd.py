import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list and edge set
    # We use a dictionary of sets for adjacency for fast updates
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in P['edges']:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    cycles_found = []
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    def get_edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def remove_edge(u, v):
        key = get_edge_key(u, v)
        if key in edge_set:
            edge_set.remove(key)
            adj[u].remove(v)
            adj[v].remove(u)
            return True
        return False

    def add_edge(u, v):
        key = get_edge_key(u, v)
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)

    def find_hamiltonian_cycle():
        """
        Finds a Hamiltonian cycle in the current remaining graph.
        Uses backtracking with constraint propagation (forced moves).
        """
        # We must visit all n vertices.
        # Start with vertex 0 to break symmetry.
        path = [0]
        visited = {0}
        
        # We need to track which edges are "reserved" for the cycle as we go.
        # Since we are removing edges from the global adj/edge_set only after success,
        # we need a local state for the search.
        # However, modifying global state is faster if we undo carefully.
        # Let's use a local copy of degrees or a set of "used" edges for the current path.
        
        # To make propagation easy, we'll work with the current adj structure 
        # but we need to be able to undo.
        # Simpler: Use a local set of 'available' edges? No, that's expensive to copy.
        # Let's stick to the previous approach of modifying global state and undoing, 
        # BUT add propagation.
        
        # State: path, visited
        # We will temporarily remove edges from adj/edge_set as we add them to the path.
        # This makes "degree" checks O(1).
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                first = path[0]
                key = get_edge_key(first, last)
                if key in edge_set:
                    return path[:]
                return None
            
            curr = path[-1]
            
            # Propagation: Check for forced moves
            # If any unvisited vertex has degree 1, it must connect to its only neighbor.
            # If any unvisited vertex has degree 0, fail.
            # If the current vertex has only 1 unvisited neighbor, it must go there.
            
            # Actually, simpler propagation:
            # For every unvisited vertex v:
            # deg(v) = number of neighbors in edge_set (which are effectively unvisited or path ends?)
            # No, edge_set contains ALL remaining edges.
            # We need to count neighbors that are *available* for the path.
            # Available neighbors of v are those u such that (u,v) in edge_set.
            # But if u is visited and not the start/end, we can't go there.
            
            # Let's implement a simple heuristic: Warnsdorff's rule with a tie-breaker.
            # And a pruning check:
            # If any unvisited vertex v has no edges to other unvisited vertices 
            # AND v is not the last vertex (which must connect to start), it's a dead end?
            # Not exactly, it could connect to start if it's the last one.
            
            # Let's try to find a vertex with degree 2 in the subgraph of unvisited vertices + current end.
            # This is complex. Let's stick to the effective Warnsdorff.
            
            neighbors = list(adj[curr])
            valid_neighbors = []
            for nxt in neighbors:
                if nxt not in visited:
                    valid_neighbors.append(nxt)
            
            if not valid_neighbors:
                return None

            # Heuristic: Sort by degree of neighbor in the remaining graph (Warnsdorff)
            # Smallest degree first.
            valid_neighbors.sort(key=lambda x: len(adj[x]))
            
            # Random tie-breaking to escape local minima
            random.shuffle(valid_neighbors) # Shuffling sorted list is okay for ties if we don't care about order of non-ties? 
            # Better: 
            # valid_neighbors.sort(key=lambda x: (len(adj[x]), random.random()))
            
            for nxt in valid_neighbors:
                # Move
                path.append(nxt)
                visited.add(nxt)
                
                # Remove edges incident to nxt from the "available" pool?
                # No, we only remove the edge (curr, nxt).
                # But we must ensure we don't use other edges of nxt later.
                # Since nxt is now visited, we won't enter it again.
                # But its other edges are still in edge_set, which might confuse degree counts?
                # Yes, len(adj[nxt]) includes edges to visited vertices.
                # This makes Warnsdorff's rule slightly inaccurate.
                
                # To be precise, we should remove ALL edges from nxt except the one we just used?
                # No, if we remove them, we can't undo easily if we backtrack.
                # We can just remove the edge (curr, nxt).
                
                edge_key = get_edge_key(curr, nxt)
                if edge_key not in edge_set:
                    # Should not happen
                    path.pop()
                    visited.remove(nxt)
                    continue
                    
                # Temporarily remove the edge
                edge_set.remove(edge_key)
                adj[curr].remove(nxt)
                adj[nxt].remove(curr)
                
                # Pruning Check:
                # 1. If any unvisited vertex is isolated in the remaining graph, fail.
                dead_end = False
                unvisited = [v for v in range(n) if v not in visited]
                if len(unvisited) > 0:
                    # The last unvisited vertex must be able to connect to path[0].
                    # Other unvisited vertices must have at least one unvisited neighbor.
                    for v in unvisited:
                        # Count neighbors in unvisited
                        unvisited_neighbors = 0
                        for u in adj[v]:
                            if u in unvisited:
                                unvisited_neighbors += 1
                        
                        # If v is the only unvisited vertex, it must connect to path[0]
                        if len(unvisited) == 1:
                            if get_edge_key(v, path[0]) not in edge_set:
                                dead_end = True
                                break
                        else:
                            # If it has no unvisited neighbors, it can't be placed
                            if unvisited_neighbors == 0:
                                dead_end = True
                                break
                
                if not dead_end:
                    res = backtrack()
                    if res is not None:
                        return res
                
                # Undo
                edge_set.add(edge_key)
                adj[curr].add(nxt)
                adj[nxt].add(curr)
                
                path.pop()
                visited.remove(nxt)
            
            return None
        
        return backtrack()

    # Try to find 3 edge-disjoint Hamiltonian cycles
    # The graph is 6-regular, so max 3 cycles.
    
    # We will try to find them sequentially.
    # To increase robustness, we can try multiple random restarts if the first attempt fails.
    
    max_attempts = 20
    attempts = 0
    
    while len(cycles_found) < 3 and attempts < max_attempts:
        if time.time() > deadline:
            break
            
        # Save state
        orig_edge_set = set(edge_set)
        orig_adj = {i: set(s) for i, s in adj.items()}
        
        # Try to find a cycle
        cycle = find_hamiltonian_cycle()
        
        if cycle is None:
            # Restore state
            edge_set = orig_edge_set
            adj = orig_adj
            attempts += 1
            # If we can't find any cycle, break
            if len(cycles_found) == 0:
                break
            continue
        
        # Verify the cycle is valid and edge-disjoint
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = get_edge_key(u, v)
            if key not in edge_set:
                valid = False
                break
        
        if valid:
            # Commit the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remove_edge(u, v)
            cycles_found.append(cycle)
            attempts = 0 # Reset attempts on success
        else:
            # Restore state
            edge_set = orig_edge_set
            adj = orig_adj
            attempts += 1

    # If we didn't find 3, we return what we have.
    # If we found 0, we must return at least 1 (nonempty list).
    # The fallback in the seed was [list(range(n))].
    # Let's ensure we return a valid result.
    
    if not cycles_found:
        # Fallback: Try to find any simple cycle? 
        # The task requires "simple cycles... use exactly the same vertex set".
        # If we can't find a Hamiltonian cycle, we can't satisfy the "same vertex set" 
        # requirement with multiple cycles, but we can return a single cycle 
        # that uses a subset? 
        # "All cycles must use exactly the same vertex set of size at least three".
        # If we return 1 cycle, it uses a vertex set S. 
        # If we return [list(range(n))], it uses S=V. 
        # Is [0..20] a valid cycle? 
        # Edges: (0,1), (1,2), ..., (19,20), (20,0).
        # Check if these edges exist in P.
        # (0,1) Yes. (1,2) Yes. (2,3) Yes. (3,4) Yes. (4,5) Yes. (5,6) Yes.
        # (6,7) Yes. (7,8) Yes. (8,9) Yes. (9,10) Yes. (10,11) Yes.
        # (11,12) Yes. (12,13) Yes. (13,14) Yes. (14,15) Yes. (15,16) Yes.
        # (16,17) Yes. (17,18) Yes. (18,19) Yes. (19,20) Yes. (20,0) Yes.
        # So [0, 1, ..., 20] is a valid Hamiltonian cycle.
        return [[int(i) for i in range(n)]]
        
    # Ensure types are correct
    result = []
    for c in cycles_found:
        result.append([int(v) for v in c])
        
    return result

# EVOLVE-BLOCK-END

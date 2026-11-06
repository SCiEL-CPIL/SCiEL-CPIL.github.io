import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    
    # We want to find the largest n <= 40 with a triangle-free graph and alpha <= 9.
    # We search for circulant graphs.
    
    for n in range(40, 25, -1):
        # Try multiple random restarts for this n
        for restart in range(500):
            if time.time() > deadline:
                # If we fail to find for n, we should probably return the best found so far?
                # But the task says "Maximize n". If we don't find one for 40, we must find one for 39, etc.
                # However, if we run out of time, we should return the best valid graph found in previous iterations?
                # The loop goes 40 down to 26. If we fail 40, we try 39.
                # If we run out of time inside the loop for a specific n, we should break and return the last valid one?
                # No, if we are checking n=40 and fail, we move to n=39.
                # If we run out of time while checking n=39, we haven't found a graph for 39 yet.
                # We should return the best graph found in a previous successful n?
                # But the loop structure is: find graph for n, return it.
                # If we don't find one for 40, we try 39.
                # If we find one for 39, we return it.
                # If we run out of time before finding any, we return the seed (n=26).
                
                # To be safe, let's keep track of the best graph found.
                pass
            
            max_shift = n // 2
            # Start with a random subset of shifts
            # Bias towards larger sets?
            # Let's start with all shifts and remove
            S = set(range(1, max_shift + 1))
            
            # Remove shifts to eliminate triangles
            # Condition: (S + S) intersect (-S) should be {0}
            # We remove shifts that cause conflicts.
            
            changed = True
            while changed:
                changed = False
                # Find a conflict
                # u, w in S such that (u+w)%n in -S and (u+w)%n != 0
                conflict_found = False
                # To speed up, we can just check all pairs
                # If n is small, this is fast.
                for u in list(S):
                    for w in list(S):
                        s = (u + w) % n
                        if s != 0 and s in (-S):
                            # Remove u or w
                            # Randomly choose which to remove
                            if random.random() < 0.5:
                                S.remove(u)
                            else:
                                S.remove(w)
                            changed = True
                            conflict_found = True
                            break
                    if conflict_found:
                        break
            
            # Now S is triangle-free.
            # Check independence number.
            # Build adjacency list
            adj = [set() for _ in range(n)]
            for u in S:
                for v in range(n):
                    if (v - u) % n in S or (u - v) % n in S:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Calculate max independent set size
            # Simple branch and bound
            # Or greedy upper bound?
            # For n=40, a simple recursive MIS is fast if graph is dense.
            
            def max_independent_set_size(adj, n):
                # Use a simple recursive approach with pruning
                # Order vertices by degree
                vertices = list(range(n))
                
                # Simple upper bound: number of remaining vertices
                # Better: greedy coloring of remaining graph?
                
                # Let's use a standard backtracking
                best = 0
                current_set = set()
                
                def dfs(remaining, current_size):
                    nonlocal best
                    if not remaining:
                        if current_size > best:
                            best = current_size
                        return
                    if current_size + len(remaining) <= best:
                        return
                    
                    # Pick a vertex
                    # Heuristic: pick vertex with highest degree in remaining?
                    # Or just first one
                    v = remaining[0]
                    rem_rest = remaining[1:]
                    
                    # Option 1: Include v
                    # Neighbors of v in remaining cannot be included
                    neighbors_in_rem = adj[v] & set(rem_rest)
                    new_rem = [x for x in rem_rest if x not in neighbors_in_rem]
                    dfs(new_rem, current_size + 1)
                    
                    # Option 2: Exclude v
                    dfs(rem_rest, current_size)
                
                dfs(vertices, 0)
                return best

            alpha = max_independent_set_size(adj, n)
            
            if alpha <= 9:
                # Found a valid graph
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # If alpha > 9, try another restart
            # To diversify, we can randomize the removal order or initial S
            # For now, the random removal choice provides some diversity.
            # We can also try starting with a random subset of S
            if restart % 10 == 0:
                # Try a sparser initial S
                S = set()
                for i in range(1, max_shift + 1):
                    if random.random() < 0.5:
                        S.add(i)
                # Re-run the removal loop
                changed = True
                while changed:
                    changed = False
                    conflict_found = False
                    for u in list(S):
                        for w in list(S):
                            s = (u + w) % n
                            if s != 0 and s in (-S):
                                if random.random() < 0.5:
                                    S.remove(u)
                                else:
                                    S.remove(w)
                                changed = True
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                
                # Re-check alpha
                adj = [set() for _ in range(n)]
                for u in S:
                    for v in range(n):
                        if (v - u) % n in S or (u - v) % n in S:
                            adj[u].add(v)
                            adj[v].add(u)
                
                alpha = max_independent_set_size(adj, n)
                if alpha <= 9:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if v in adj[u]:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}

    # If no graph found with n > 26, return the seed graph (Andrasfai 9)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

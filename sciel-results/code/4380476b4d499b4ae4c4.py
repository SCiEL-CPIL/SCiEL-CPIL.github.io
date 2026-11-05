import numpy as np
import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s for safety
    
    # 1. Generate permutations
    perms = list(itertools.permutations(range(1, n+1)))
    N = len(perms)
    
    # 2. Compute cost matrix
    # cost[i][j] = n - overlap(perms[i], perms[j])
    
    prefix_map = [{} for _ in range(n)]
    suffix_map = [{} for _ in range(n)]
    
    for idx, p in enumerate(perms):
        for k in range(1, n):
            pre = tuple(p[:k])
            suf = tuple(p[n-k:])
            if pre not in prefix_map[k-1]:
                prefix_map[k-1][pre] = []
            prefix_map[k-1][pre].append(idx)
            if suf not in suffix_map[k-1]:
                suffix_map[k-1][suf] = []
            suffix_map[k-1][suf].append(idx)
            
    cost = np.full((N, N), n, dtype=np.int8)
    
    for k in range(n-1, 0, -1):
        p_map = prefix_map[k-1]
        s_map = suffix_map[k-1]
        for suf, a_indices in s_map.items():
            if suf in p_map:
                b_indices = p_map[suf]
                for i in a_indices:
                    for j in b_indices:
                        if cost[i][j] == n:
                            cost[i][j] = n - k

    # 3. Nearest Neighbor Search
    best_tour = None
    best_cost = float('inf')
    
    # Try a few starting points
    # Use 0, 1, ..., 9 and some randoms
    starts = list(range(min(10, N)))
    # Add random starts if time permits
    # We'll just check time in the loop
    
    for start in starts:
        if time.time() > deadline:
            break
            
        visited = np.zeros(N, dtype=bool)
        visited[start] = True
        current = start
        tour = [start]
        current_cost = 0
        
        # Maintain unvisited list for faster min lookup
        unvisited = [i for i in range(N) if i != start]
        
        for _ in range(N - 1):
            if time.time() > deadline:
                break
                
            # Find min cost node in unvisited
            # min(unvisited, key=lambda x: cost[current, x])
            # This is O(|unvisited|)
            
            # Optimization: 
            # If we use numpy, we can do:
            # costs = cost[current, unvisited]
            # min_idx = np.argmin(costs)
            # next_node = unvisited[min_idx]
            # This is O(|unvisited|) but faster in C.
            
            # Let's use numpy for the min lookup.
            # unvisited is a list, convert to array?
            # Or just keep it as a numpy array and remove?
            # Removing from numpy array is slow.
            
            # Let's stick to Python list min for simplicity, 
            # but maybe it's too slow.
            
            # Let's try the numpy approach with a mask.
            # It's O(N) but C-speed.
            
            # row = cost[current]
            # masked_row = np.where(visited, 255, row)
            # next_node = np.argmin(masked_row)
            
            # This is O(N).
            # 5040 * 5040 = 25M ops.
            # C-speed, so ~0.1s?
            
            row = cost[current]
            # Create masked view
            # We can't modify row.
            # So we do:
            # next_node = np.argmin(np.where(visited, 255, row))
            
            # This allocates two arrays of size N.
            # 5040 * 8 bytes = 40KB.
            # 5040 iterations.
            # Total allocation ~ 200MB.
            # Might be ok.
            
            next_node = np.argmin(np.where(visited, 255, row))
            
            # Update
            visited[next_node] = True
            tour.append(next_node)
            current_cost += row[next_node]
            current = next_node
            
        # End of tour construction
        if len(tour) == N:
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour

    # If no tour found (shouldn't happen), fallback to standard
    if best_tour is None:
        # Fallback to simple sequential
        best_tour = list(range(N))
        
    # 4. Construct String
    word = list(perms[best_tour[0]])
    for i in range(1, N):
        prev_id = best_tour[i-1]
        curr_id = best_tour[i]
        overlap = n - cost[prev_id][curr_id]
        word.extend(perms[curr_id][overlap:])
        
    return "".join(map(str, word))

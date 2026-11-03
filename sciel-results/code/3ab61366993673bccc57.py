import time
import itertools
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Check time after perm generation
    if time.time() > deadline - 90:
        return _classic_construct(n)
    
    # Convert perms to numpy array for vectorized ops
    perm_array = np.array(perms, dtype=np.int8)  # shape (N, n)
    
    # Precompute cost matrix using vectorized approach
    # For each k from n down to 1, check if suffix of length k of perm i == prefix of length k of perm j
    # C[i][j] = n - max_k such that perm_array[i, n-k:] == perm_array[j, :k]
    
    C = np.full((N, N), n, dtype=np.int16)
    
    for k in range(1, n + 1):
        # suffixes: perm_array[:, n-k:] shape (N, k)
        # prefixes: perm_array[:, :k] shape (N, k)
        # For each i, find all j where perm_array[i, n-k:] == perm_array[j, :k]
        # We can do this by checking equality of the k-dimensional slices
        
        # Efficient: for each i, compute which j have matching prefix
        # Use broadcasting in chunks to avoid memory issues
        # N=5040, k up to 7. (5040, 5040, k) would be too large.
        # Instead, for each k, we can group by prefix value.
        
        # Create a dictionary mapping prefix tuple to list of indices
        prefix_dict = {}
        for j in range(N):
            key = tuple(perm_array[j, :k])
            if key not in prefix_dict:
                prefix_dict[key] = []
            prefix_dict[key].append(j)
        
        # For each i, find suffix and look up matching prefixes
        for i in range(N):
            suffix_key = tuple(perm_array[i, n-k:])
            if suffix_key in prefix_dict:
                for j in prefix_dict[suffix_key]:
                    if C[i, j] > n - k:
                        C[i, j] = n - k
    
    # Check time
    if time.time() > deadline - 30:
        return _classic_construct(n)
    
    # Nearest Neighbor with multiple restarts
    best_path = None
    best_len = float('inf')
    
    num_restarts = 1
    for restart in range(num_restarts):
        start = random.randint(0, N - 1)
        visited = np.zeros(N, dtype=bool)
        path = [start]
        visited[start] = True
        current = start
        
        for _ in range(N - 1):
            row = C[current]
            # Find unvisited j with min cost
            # Mask visited nodes with large value
            masked_row = row.copy()
            masked_row[visited] = 999
            best_next = int(np.argmin(masked_row))
            path.append(best_next)
            visited[best_next] = True
            current = best_next
        
        # Calculate length
        path_arr = np.array(path)
        edges = np.array([C[path_arr[i], path_arr[i+1]] for i in range(len(path)-1)])
        current_len = n + int(edges.sum())
        
        if current_len < best_len:
            best_len = current_len
            best_path = path
        
        # Check time for more restarts
        if time.time() > deadline - 40:
            break
        # Try more restarts if time permits
        if time.time() < deadline - 60 and restart < 4:
            num_restarts = restart + 2
    
    path = best_path
    C_list = C.tolist()
    
    # 2-Opt Local Search with prefix sums
    # Maintain edge costs and prefix sums
    path_arr = list(path)
    edge_costs = np.array([C[path_arr[i], path_arr[i+1]] for i in range(len(path_arr)-1)], dtype=np.int32)
    total_len = n + int(edge_costs.sum())
    
    def compute_prefix_sums(ec):
        return np.concatenate([[0], np.cumsum(ec)])
    
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 15 and pass_count < 5:
        improved = False
        pass_count += 1
        prefix_sums = compute_prefix_sums(edge_costs)
        
        for i in range(1, len(path_arr) - 2):
            if time.time() > deadline - 10:
                break
            prev_i = path_arr[i-1]
            curr_i = path_arr[i]
            
            # For j from i+1 to N-2
            # Old boundary cost: C[prev_i][curr_i] + C[path_arr[j]][path_arr[j+1]]
            # New boundary cost: C[prev_i][path_arr[j]] + C[curr_i][path_arr[j+1]]
            # Old internal: sum(edge_costs[i:j])
            # New internal: sum(C[path_arr[k+1]][path_arr[k]] for k in i..j-1)
            
            # We need reverse edge costs too
            # Precompute reverse cost lookup
            for j in range(i + 1, len(path_arr) - 1):
                if time.time() > deadline - 10:
                    break
                curr_j = path_arr[j]
                next_j = path_arr[j+1]
                
                old_boundary = C[prev_i, curr_i] + C[curr_j, next_j]
                new_boundary = C[prev_i, curr_j] + C[curr_i, next_j]
                
                old_internal = int(prefix_sums[j] - prefix_sums[i])
                # New internal: sum of C[path_arr[k+1]][path_arr[k]] for k=i..j-1
                # This is sum of reverse edges
                new_internal = 0
                for k in range(i, j):
                    new_internal += C[path_arr[k+1], path_arr[k]]
                
                delta = (new_boundary + new_internal) - (old_boundary + old_internal)
                
                if delta < 0:
                    # Apply the reversal
                    path_arr[i+1:j+1] = reversed(path_arr[i+1:j+1])
                    # Update edge costs
                    edge_costs[i-1] = C[path_arr[i-1], path_arr[i]]
                    for k in range(i, j):
                        edge_costs[k] = C[path_arr[k], path_arr[k+1]]
                    edge_costs[j] = C[path_arr[j], path_arr[j+1]]
                    total_len += delta
                    improved = True
                    # Recompute prefix sums for next i
                    prefix_sums = compute_prefix_sums(edge_costs)
                    break  # Restart i loop since path changed
        
    path = path_arr
    
    # Convert path to string
    result = list(perms[path[0]])
    for i in range(1, len(path)):
        prev = perms[path[i-1]]
        curr = perms[path[i]]
        # Find overlap
        for k in range(n, 0, -1):
            if prev[n-k:] == curr[:k]:
                result.extend(curr[k:])
                break
        else:
            result.extend(curr)
            
    return "".join(map(str, result))

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

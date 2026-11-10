import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
full_mask = (1 << N) - 1

# Build adjacency bitmasks
adj = [0] * N
for a, b in P['edges']:
    adj[a] |= (1 << b)
    adj[b] |= (1 << a)

# Complement adjacency for independent sets
cadj = [0] * N
for v in range(N):
    cadj[v] = full_mask ^ (1 << v) ^ adj[v]

# Precompute is_clique and is_indep for all subsets
is_clique = [False] * (1 << N)
is_indep = [False] * (1 << N)
is_clique[0] = True
is_indep[0] = True

# We can compute these using DP or direct check.
# Direct check is O(N * 2^N) which is ~200k, very fast.
for S in range(1, 1 << N):
    # Get lowest set bit
    lb = S & (-S)
    v = lb.bit_length() - 1
    rest = S ^ lb
    
    # S is clique if rest is clique and v is connected to all in rest
    if is_clique[rest]:
        if (adj[v] & rest) == rest:
            is_clique[S] = True
    else:
        is_clique[S] = False
        
    # S is independent if rest is independent and v is NOT connected to any in rest
    if is_indep[rest]:
        if (adj[v] & rest) == 0:
            is_indep[S] = True
    else:
        is_indep[S] = False

# Combine into is_valid
is_valid = [False] * (1 << N)
for S in range(1 << N):
    is_valid[S] = is_clique[S] or is_indep[S]

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # 55s safety margin within 60s limit
    
    # DP arrays
    size = 1 << N
    dp = [N + 1] * size
    parent = [0] * size
    dp[0] = 0
    
    # Precompute valid submasks for each mask to speed up lookup
    # For each mask S, store list of valid proper submasks T (T != S, T != 0)
    # Actually, we'll just use is_valid directly - it's O(1)
    
    # Fill DP table
    for S in range(1, size):
        if is_valid[S]:
            dp[S] = 1
            parent[S] = S
        else:
            # Find best submask T
            best_val = N + 1
            best_T = 0
            T = S
            while T:
                if is_valid[T]:
                    rem = S ^ T
                    val = dp[rem] + 1
                    if val < best_val:
                        best_val = val
                        best_T = T
                        if best_val == 2:
                            break
                T = (T - 1) & S
            
            dp[S] = best_val
            parent[S] = best_T
            
            # Deadline guard
            if time.time() > deadline:
                break
    
    # Reconstruct solution
    parts = []
    S = full_mask
    while S:
        T = parent[S]
        if T == 0:
            # Fallback: treat remaining as one part (shouldn't happen)
            T = S
        # Convert mask T to list of vertices
        part_vertices = []
        temp_T = T
        while temp_T:
            lb = temp_T & (-temp_T)
            v = lb.bit_length() - 1
            part_vertices.append(v)
            temp_T ^= lb
        parts.append(part_vertices)
        S ^= T
        
    # Ensure types are correct
    out = []
    for part in parts:
        out.append([int(x) for x in part])
        
    return out

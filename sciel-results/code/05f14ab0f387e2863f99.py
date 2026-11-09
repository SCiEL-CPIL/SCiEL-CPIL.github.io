import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    non_adj = [0] * n
    all_mask = (1 << n) - 1
    for v in range(n):
        non_adj[v] = all_mask ^ adj[v] ^ (1 << v)

    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        if is_independent[rest] and (rest & ~non_adj[v]) == 0:
            is_independent[mask] = True
            
    valid_part = [False] * (1 << n)
    for mask in range(1 << n):
        valid_part[mask] = is_clique[mask] or is_independent[mask]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # DP approach
    # dp[k][mask] = True if 'mask' can be covered by k valid parts
    # We iterate k from 1 to n
    # To save memory and time, we only need the previous layer
    # But we need to reconstruct the solution, so we store parent pointers or the actual submasks chosen.
    # parent[k][mask] = submask used for the k-th part covering 'mask'
    
    # Since n is small (14), we can use a list of dicts or arrays.
    # dp_prev[mask] = True/False
    # dp_curr[mask] = True/False
    # parent[k][mask] = submask
    
    # We want the minimum k.
    # Initialize dp for k=1
    # dp[1][mask] = valid_part[mask]
    
    # To reconstruct, we store the chosen submask for each (k, mask)
    # parent[k][mask] = sub
    # We only need to store this for the successful paths.
    
    # Optimization: Precompute list of valid submasks for each mask?
    # No, iterating submasks is fast enough.
    
    # Let's use a list of lists for parent pointers
    # parent[k][mask] = submask
    # We will store parents for all k up to the solution k.
    
    # However, we don't know k in advance.
    # We can store parents in a list `parents` where parents[k] is a dict or array.
    
    # Given 3^14 ~ 4.8M, and k up to 14, total states ~ 67M.
    # Storing an int for each state is ~ 268 MB. This might be high but likely okay.
    # Alternatively, we can re-run the DP for the found k to reconstruct, or store only the necessary info.
    # Better: Store `choice[k][mask]` = submask.
    
    # Let's try to be memory efficient.
    # We can just store the submask chosen for the *first* time a state becomes True.
    
    # Actually, for N=14, we can just use a 2D array of size (n+1) x (1<<n)
    # 15 * 16384 = 245,760 entries.
    # Each entry is an int (submask) or -1.
    # This is very small.
    
    parent = [[-1] * (1 << n) for _ in range(n + 1)]
    dp_prev = [False] * (1 << n)
    dp_curr = [False] * (1 << n)
    
    # k = 1
    for mask in range(1, 1 << n):
        if valid_part[mask]:
            dp_prev[mask] = True
            parent[1][mask] = mask # The part is the mask itself
    
    # Check if k=1 is enough
    if dp_prev[all_mask]:
        # Reconstruct
        parts = []
        mask = all_mask
        for k in range(1, 0, -1): # Just k=1
            sub = parent[k][mask]
            parts.append(sub)
            mask ^= sub
        # Convert to lists
        result = []
        for p_mask in parts:
            vertices = []
            for v in range(n):
                if p_mask & (1 << v):
                    vertices.append(v)
            result.append(vertices)
        return result

    for k in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        dp_curr = [False] * (1 << n)
        # For each mask, check if it can be formed by a valid sub + prev_mask
        # We iterate mask from 1 to all_mask
        for mask in range(1, 1 << n):
            # If dp_prev[mask] is already true, we don't need k parts, we could use fewer.
            # But we are looking for *exactly* k? No, we want *at most* k?
            # The problem is "minimize number of parts".
            # If dp_prev[mask] is True, then mask can be covered by k-1 parts.
            # So it can be covered by k parts (by adding an empty part? No, parts must be non-empty).
            # Wait, the definition of dp[k][mask] is "can be covered by exactly k non-empty valid parts".
            # If it can be covered by k-1, can it be covered by k?
            # Only if one of the parts can be split into two valid parts.
            # Not necessarily. E.g., a clique of size 2 can be 1 part (clique) or 2 parts (two independent sets of size 1).
            # So dp[k][mask] is NOT simply OR of dp[k-1][mask].
            # We must compute it properly.
            
            # However, if we find a solution with k parts, that's a valid upper bound.
            # We iterate k from 1 upwards. The first k where dp[k][all_mask] is True is the minimum.
            
            # Iterate submasks of mask
            # Optimization: Only consider submasks that are valid_part
            # We can precompute valid submasks? No.
            # Just iterate.
            sub = mask
            while sub:
                if valid_part[sub]:
                    rest = mask ^ sub
                    if rest == 0:
                        # This would imply k=1 case, handled earlier or sub=mask
                        # If k>1, rest must be non-empty?
                        # If rest is 0, then mask is covered by 1 part (sub).
                        # This is only valid for k=1.
                        # For k>1, we need rest to be coverable by k-1 parts.
                        # If rest is 0, dp[k-1][0] should be False (0 parts for non-empty set? No, 0 parts for empty set is True).
                        # Let's define dp[0][0] = True, dp[0][non-zero] = False.
                        pass
                    if rest != 0 and dp_prev[rest]:
                        dp_curr[mask] = True
                        parent[k][mask] = sub
                        break
                sub = (sub - 1) & mask
            
            # Also, if mask itself is a valid part, it can be covered by 1 part.
            # But we are at k >= 2.
            # If valid_part[mask] is True, it is covered by 1 part.
            # Can it be covered by k parts?
            # Yes, if we split it. But we are looking for *existence*.
            # If dp_prev[mask] is True, it means mask can be covered by k-1 parts.
            # This doesn't directly help dp_curr[mask] (k parts) unless we can split one part.
            # The submask iteration handles this: if we pick sub=mask, rest=0.
            # We need dp[k-1][0] to be True.
            # Let's set dp[0][0] = True.
            
        # Check if we solved all_mask
        if dp_curr[all_mask]:
            # Reconstruct
            parts = []
            mask = all_mask
            for i in range(k, 0, -1):
                sub = parent[i][mask]
                parts.append(sub)
                mask ^= sub
            # Convert to lists
            result = []
            for p_mask in parts:
                vertices = []
                for v in range(n):
                    if p_mask & (1 << v):
                        vertices.append(v)
                result.append(vertices)
            return result
            
        dp_prev = dp_curr

    # Fallback: Greedy (should not be reached)
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END

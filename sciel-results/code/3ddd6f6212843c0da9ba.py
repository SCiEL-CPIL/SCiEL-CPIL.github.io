"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute validity for all subsets: is_clique or is_independent
    # valid[mask] = True if mask is a clique or an independent set
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        is_clique = True
        is_independent = True
        
        # Check clique
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            remaining = mask ^ v_bit
            if remaining & ~adj[v]:
                is_clique = False
                break
            m ^= v_bit
        
        if not is_clique:
            # Check independent
            m = mask
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                remaining = mask ^ v_bit
                if remaining & adj[v]:
                    is_independent = False
                    break
                m ^= v_bit
        
        valid[mask] = is_clique or is_independent

    # Backtracking search for a given k
    def solve(k):
        # parts is a list of bitmasks representing the current partition parts
        parts = []
        
        def backtrack(remaining, parts):
            if remaining == 0:
                return True
            
            # Heuristic: pick the vertex in 'remaining' that has the fewest neighbors in 'remaining'
            # This is often a good constraint for coloring/partitioning.
            # Alternatively, pick the vertex with the fewest valid placements.
            # Let's try minimum degree in the remaining subgraph.
            min_deg = n + 1
            best_v = -1
            r = remaining
            while r:
                v_bit = r & (-r)
                v_idx = v_bit.bit_length() - 1
                deg = bin(remaining & adj[v_idx]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v_idx
                r ^= v_bit
            
            if best_v == -1:
                return False
            
            v_bit = 1 << best_v
            
            # Option 1: Try to add best_v to an existing part
            # We can prune symmetric parts (e.g., if two parts are identical, only try one)
            # But for simplicity and speed with n=14, we just iterate.
            # Optimization: If we've already tried adding to a part with the same mask, skip others.
            seen_masks = set()
            for i in range(len(parts)):
                p_mask = parts[i]
                if p_mask in seen_masks:
                    continue
                seen_masks.add(p_mask)
                
                new_p_mask = p_mask | v_bit
                if valid[new_p_mask]:
                    parts[i] = new_p_mask
                    if backtrack(remaining ^ v_bit, parts):
                        return True
                    parts[i] = p_mask
            
            # Option 2: Start a new part (if we haven't reached limit k)
            if len(parts) < k:
                parts.append(v_bit)
                if backtrack(remaining ^ v_bit, parts):
                    return True
                parts.pop()
            
            return False
        
        if backtrack(full_mask, parts):
            return parts
        return None

    # Iterate k from 1 to n
    best_parts = None
    for k in range(1, n + 1):
        res = solve(k)
        if res is not None:
            best_parts = res
            break
    
    # If no solution found (should not happen as k=n is always valid), fallback
    if best_parts is None:
        best_parts = [[i] for i in range(n)]
        
    # Convert bitmasks to lists of vertices
    result = []
    for p_mask in best_parts:
        part = []
        m = p_mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            part.append(v)
            m ^= v_bit
        result.append(part)
        
    return result

# EVOLVE-BLOCK-END

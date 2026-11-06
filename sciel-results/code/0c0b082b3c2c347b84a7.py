import time
import random
import numpy as np

def _popcount(x):
    return bin(x).count('1')

def _get_bit_indices(x, count):
    """Get up to 'count' indices of set bits in x."""
    indices = []
    while x and len(indices) < count:
        # Get lowest set bit
        lsb = x & -x
        idx = (lsb.bit_length() - 1)
        indices.append(idx)
        x ^= lsb
    return indices

def _find_k222_bitset(adj, n):
    """
    Find a K2,2,2 subgraph in the graph represented by adj (list of ints).
    Returns (u, v, w, x, y, z) or None.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au & (1 << v)):
                continue
            
            # Common neighbors of u and v
            common = au & adj[v]
            # Remove u and v
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            if _popcount(common) < 4:
                continue
            
            # We need to find w, x in common such that they have at least 2 common neighbors in common
            # that are also in common (y, z)
            # Iterate pairs w, x from common
            # To speed up, get list of candidates
            candidates = _get_bit_indices(common, n)
            
            for i in range(len(candidates)):
                w = candidates[i]
                aw = adj[w]
                for j in range(i + 1, len(candidates)):
                    x = candidates[j]
                    ax = adj[x]
                    
                    # Common neighbors of w and x
                    common_wx = aw & ax
                    # Must be in common (neighbors of u and v)
                    common_wx_in_common = common_wx & common
                    
                    if _popcount(common_wx_in_common) >= 2:
                        # Find two such neighbors y, z
                        yz = _get_bit_indices(common_wx_in_common, 2)
                        if len(yz) == 2:
                            y = yz[0]
                            z = yz[1]
                            return (u, v, w, x, y, z)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try n from 30 down to 17
    # 30 is the theoretical max for score improvement (30/17 > 29/17)
    # If we find 30, we stop.
    
    for n in range(30, 16, -1):
        if time.time() > deadline:
            break
            
        found = False
        best_adj = None
        
        # Number of restarts depends on time remaining
        # For n=30, it's hard. For n=29, it's easier.
        max_restarts = 50 if n == 30 else 100
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            # We track the current state
            current_adj = adj[:]
            
            # If n is small, we can try more aggressive search
            # If n is 30, we have less time per restart
            
            steps = 0
            max_steps = 5000 if n <= 29 else 2000
            
            while steps < max_steps:
                if time.time() > deadline:
                    break
                
                # Check color 0
                k222 = _find_k222_bitset(current_adj, n)
                if k222 is not None:
                    u, v, w, x, y, z = k222
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    # Pick a random edge to flip
                    i, j = random.choice(edges)
                    if i > j:
                        i, j = j, i
                    
                    # Flip
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
                    steps += 1
                    continue
                
                # Check color 1 (complement)
                # Construct adj1 on the fly? Or just invert logic.
                # adj1[i] = full_mask ^ adj[i]
                # But we need to exclude self loops.
                # full_mask = (1 << n) - 1
                # adj1[i] = full_mask ^ adj[i] ^ (1 << i)
                
                adj1 = [0] * n
                full_mask = (1 << n) - 1
                for i in range(n):
                    adj1[i] = (full_mask ^ current_adj[i]) & ~(1 << i)
                
                k222_1 = _find_k222_bitset(adj1, n)
                if k222_1 is not None:
                    u, v, w, x, y, z = k222_1
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    i, j = random.choice(edges)
                    if i > j:
                        i, j = j, i
                    
                    # Flip in current_adj (which changes color 1 edge to color 0)
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
                    steps += 1
                    continue
                
                # If no K222 in either, we found a valid coloring
                found = True
                best_adj = current_adj
                break
                
                # If we get stuck (no K222 found but maybe we missed one due to bug? No, find is exhaustive)
                # Then it is valid.
                
            if found:
                break
        
        if found:
            # Convert adj to colors
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': n, 'colors': colors}
            
    # Fallback to n=17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

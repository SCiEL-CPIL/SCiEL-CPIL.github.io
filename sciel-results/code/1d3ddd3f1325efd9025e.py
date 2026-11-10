import random
import time
from itertools import combinations

def is_valid_graph(n, adj):
    """
    adj: list of integers where adj[u] is a bitmask of neighbors of u.
    Returns True if no subset of size >= 6 induces a regular subgraph.
    """
    # Iterate over all subsets of size >= 6
    # This is expensive if done naively for large n.
    # For n=20, total subsets ~ 1M.
    # For each subset, check regularity.
    
    # Optimization: 
    # If a subset S is regular, all v in S have same degree in S.
    # We can skip subsets that contain a known regular subset? No, disjointness doesn't help directly.
    
    # Let's just check all.
    # Bit operations are fast in Python if we use integers.
    
    # Precompute popcount? Python 3.10+ has int.bit_count().
    
    for mask in range(1, 1 << n):
        m = mask.bit_count()
        if m < 6:
            continue
        
        # Check if induced subgraph on mask is regular
        # Get the first vertex in mask to determine the target degree
        # u = (mask & -mask).bit_length() - 1
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        target_deg = (adj[u] & mask).bit_count()
        
        # Check all other vertices in mask
        # We can iterate through bits of mask
        rem_mask = mask ^ lsb
        valid = True
        while rem_mask:
            lsb2 = rem_mask & -rem_mask
            v = lsb2.bit_length() - 1
            deg_v = (adj[v] & mask).bit_count()
            if deg_v != target_deg:
                valid = False
                break
            rem_mask ^= lsb2
        
        if valid:
            return False
            
    return True

def construct():
    # Try to find a graph on n=20
    # If found, return it.
    # If not found in reasonable time, try n=19, etc.
    # But n=20 is known to exist.
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Try n=21 first? The prompt says "Maximize n".
    # "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
    # This implies 21 is the target if possible.
    
    for n in [21, 20]:
        # Random search
        # p=0.5 is a good start.
        # We can also try to "fix" a graph.
        
        # Simple random restarts
        found = None
        attempts = 0
        while time.time() - start_time < time_limit and found is None:
            # Generate random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check validity
            # To make it faster, we can check smaller subsets first?
            # No, we must check all.
            # But we can exit early on the first violation.
            # The probability of a random graph having NO regular induced subgraph of size 6 is likely low but non-zero.
            
            if is_valid_graph(n, adj):
                found = adj
                break
            attempts += 1
            
        if found:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (found[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: return a trivial valid graph (n=5) if we fail to find larger ones
    # This should not happen for n=20 within 100s if the density is right.
    return {'n': 5, 'edges': []}

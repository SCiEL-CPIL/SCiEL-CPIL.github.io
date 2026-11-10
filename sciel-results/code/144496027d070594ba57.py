import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Try n=21 first
    for n in [21, 20]:
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Using adjacency list of sets or bitmasks
        # Bitmask: adj[i] is an integer with bits for neighbors
        adj = [0] * n
        
        # Random init
        random.seed(42) # Fixed seed for reproducibility in search? Or random?
        # Let's use a fixed seed for the first attempt, then random if it fails?
        # The task says "Maximize n". A specific valid graph is enough.
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Helper to check if a subset (as bitmask) induces a regular graph
        def is_regular(mask, k):
            # Get vertices in mask
            # We need induced degrees
            degs = []
            # Iterate through bits in mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                # Induced degree of u is popcount(adj[u] & mask)
                deg = (adj[u] & mask).bit_count()
                degs.append(deg)
                m ^= lsb
            
            # Check if all degs are equal
            if not degs:
                return False
            first = degs[0]
            for d in degs[1:]:
                if d != first:
                    return False
            return True

        # Main search loop
        found_valid = False
        while time.time() < deadline:
            # Check for regular induced subgraphs
            violation_found = False
            # Check sizes 6 to n
            for k in range(6, n+1):
                if time.time() > deadline:
                    break
                
                # Generate combinations of size k
                # Optimization: only check if k=6? No, must check all.
                # But if we find one of size 6, we fix it and restart.
                # So we can just check size 6 first. If clean, check 7, etc.
                
                for comb in itertools.combinations(range(n), k):
                    # Build mask
                    mask = 0
                    for v in comb:
                        mask |= (1 << v)
                    
                    if is_regular(mask, k):
                        # Found a violation
                        violation_found = True
                        # Flip an edge inside this subset
                        # Pick two random vertices u, v from comb
                        # Ensure they are distinct
                        if len(comb) < 2:
                            continue # k>=6 so this won't happen
                        
                        # Try to flip an edge that breaks regularity
                        # Any internal edge flip breaks regularity of this specific subset
                        # because degrees change by +/- 1 for two vertices, keeping others same.
                        # Unless the graph was not regular? But we checked it is.
                        # So any internal flip works.
                        
                        # Pick random pair
                        u, v = random.sample(comb, 2)
                        
                        # Check if edge exists
                        if adj[u] & (1 << v):
                            # Remove edge
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        
                        break # Break inner loop, restart k loop
                if violation_found:
                    break # Break k loop, restart main while loop
            
            if not violation_found:
                # No violations found
                found_valid = True
                break
        
        if found_valid:
            # Construct edge list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            
            return {'n': n, 'edges': edges}
            
    # Fallback: return a known small valid graph or empty
    # The empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}

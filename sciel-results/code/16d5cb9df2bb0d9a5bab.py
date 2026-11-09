import time
import random

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    # Try n=20 down to 5
    for n in range(20, 4, -1):
        if time.time() - start_time > timeout:
            break
        
        # Try to find a valid graph for this n
        # Use random restarts
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Generate random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            for step in range(2000):
                if time.time() - start_time > timeout:
                    break
                
                violation_mask = find_violation_mask(adj, n)
                if violation_mask is None:
                    # Verify fully just in case (should be redundant)
                    if verify_graph_mask(adj, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        # Should not happen if find_violation is correct
                        break
                
                # Pick a random edge in the violation mask to flip
                # Extract vertices
                verts = []
                m = violation_mask
                while m:
                    l = m & -m
                    v = l.bit_length() - 1
                    verts.append(v)
                    m ^= l
                
                if len(verts) < 2:
                    break # Should not happen for size >= 6
                
                u, v = random.sample(verts, 2)
                
                # Flip edge u-v
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Fallback
    return {'n': 5, 'edges': []}

def find_violation_mask(adj, n):
    # Iterate all masks
    # Optimization: Start with smaller masks?
    # It's hard to iterate masks in order of popcount easily without precomputation.
    # But random access is fine.
    # We can just iterate 0 to 2^n - 1.
    # To find a violation quickly, maybe check random masks first?
    # No, systematic is better for completeness.
    # But we can stop early.
    
    # Let's try iterating in a random order?
    # No, let's just iterate.
    # For n=20, 2^20 = 1,048,576.
    # Looping 1M times in Python takes ~0.1-0.2s.
    # If we do this 100 times, it's 10-20s.
    # This is acceptable.
    
    for mask in range(1, 1 << n):
        if mask.bit_count() < 6:
            continue
        
        if is_regular_induced_mask(mask, adj):
            return mask
            
    return None

def is_regular_induced_mask(mask, adj):
    # Get first vertex
    l = mask & -mask
    v0 = l.bit_length() - 1
    deg0 = (adj[v0] & mask).bit_count()
    
    # Check others
    m = mask ^ l
    while m:
        l = m & -m
        v = l.bit_length() - 1
        if (adj[v] & mask).bit_count() != deg0:
            return False
        m ^= l
        
    return True

def verify_graph_mask(adj, n):
    for mask in range(1, 1 << n):
        if mask.bit_count() < 6:
            continue
        if is_regular_induced_mask(mask, adj):
            return False
    return True

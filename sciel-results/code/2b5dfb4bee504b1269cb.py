import time
import random

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj_int: list of integers representing adjacency.
    u, v: vertices to connect. Assumes not already connected.
    """
    # Condition 1: Common neighbors
    # If |N(u) & N(v)| > 1, adding (u,v) creates C4?
    # If N(u) & N(v) has {w1, w2}, then u-v-w1-w2-u is C4.
    # So if (adj_int[u] & adj_int[v]).bit_count() > 1: return True
    
    # Condition 2: Edges between N(u) and N(v)
    # If a in N(u), b in N(v), and a~b, then u-a-b-v-u is C4.
    # We need to check if there is any edge between N(u) and N(v).
    # Note: u and v are not in N(u) or N(v) in the standard sense (no self-loops).
    # But N(u) contains v? No, we are adding u-v, so v is not in N(u) yet.
    # So we check edges between the sets N(u) and N(v).
    
    nu = adj_int[u]
    nv = adj_int[v]
    
    # Check common neighbors
    common = nu & nv
    if common & (common - 1): # More than 1 bit set
        return True
        
    # Check edges between N(u) and N(v)
    # Iterate over neighbors of u
    # For each x in N(u), check if N(x) intersects N(v)
    # But we must exclude v from N(v) check? No, v is not in N(v).
    # We must exclude u from N(u) check? No.
    
    temp = nu
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length() - 1
        # Check if N(x) intersects N(v)
        if adj_int[x] & nv:
            return True
        temp ^= lsb
        
    return False

def _construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        # p should be high enough to have many edges, but not too high
        # Target degree is min_deg_req.
        # Let's try p = 0.5 to 0.9
        p = random.uniform(0.4, 0.8)
        
        adj_int = [0] * n
        degrees = [0] * n
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Prune C4s
        # We need to find C4s and remove edges.
        # A C4 exists if there is a pair (u,v) with |N(u) & N(v)| >= 2.
        # Or more generally, the condition for NO C4 is |N(u) & N(v)| <= 1 for all u,v.
        # So we just need to eliminate all pairs with >= 2 common neighbors.
        
        changed = True
        while changed:
            if time.time() - start_time > time_limit:
                return None
            changed = False
            # Find a pair with >= 2 common neighbors
            # To speed up, we can check pairs randomly or systematically.
            # Systematic is O(n^2 * bit_ops). n=47, so ~1000 pairs.
            # We can iterate and break on first found.
            
            for u in range(n):
                for v in range(u+1, n):
                    common = adj_int[u] & adj_int[v]
                    if common & (common - 1): # >= 2 bits
                        # Found C4. Remove an edge.
                        # Pick a common neighbor w
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        
                        # Remove an edge incident to the highest degree vertex among u, v, w
                        # Actually, the C4 is u-w-...-v-w?
                        # The common neighbor w is part of the C4.
                        # The C4 is u-w-x-v-u where x is another common neighbor.
                        # Let's pick x.
                        common_rest = common ^ lsb
                        lsb2 = common_rest & -common_rest
                        x = lsb2.bit_length() - 1
                        
                        # Vertices involved: u, v, w, x
                        # Edges in C4: (u,w), (w,v), (v,x), (x,u)
                        # We can remove any of these.
                        # Let's remove the one incident to the highest degree vertex.
                        candidates = [
                            (u, w, degrees[u]),
                            (w, v, degrees[w]),
                            (v, x, degrees[v]),
                            (x, u, degrees[x])
                        ]
                        # Sort by degree descending
                        candidates.sort(key=lambda t: t[2], reverse=True)
                        a, b, _ = candidates[0]
                        
                        # Remove edge (a, b)
                        adj_int[a] &= ~(1 << b)
                        adj_int[b] &= ~(1 << a)
                        degrees[a] -= 1
                        degrees[b] -= 1
                        changed = True
                        break
                if changed:
                    break
                    
        # After pruning, check min degree
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
            
        # If min_deg is close, try to add edges
        # This part is optional but might help
        # For simplicity, just return None and try next random init
        # Or, we can try to add edges to low degree vertices
        
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 1:
            break
            
        min_deg_req = max(1, n - 39)
        t_limit = min(20.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}

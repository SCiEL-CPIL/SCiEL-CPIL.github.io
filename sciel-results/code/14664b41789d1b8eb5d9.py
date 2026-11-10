import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 18
    # adj[i] is bitmask of neighbors of i
    adj = [0] * n
    
    # Initialize with a random K4-free graph or complete bipartite
    # Let's start with K9,9 (score 18? No, K9,9 is bipartite, so triangle-free size 18).
    # We want to minimize triangle-free size.
    # Start with a dense random graph and remove edges to make it K4-free?
    # Or start with K6,6,6 and mutate.
    
    # Let's initialize with K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    def is_k4_free(a):
        for u in range(n):
            for v in range(u + 1, n):
                if a[u] & (1 << v):
                    common = a[u] & a[v]
                    # Check if common induces an edge
                    # For each w in common, check if a[w] & common has any bit other than w?
                    # Actually, if a[w] & common is not 0, there is an edge w-x in common.
                    # Since w is in common, a[w] & common will include x if (w,x) is an edge.
                    # It will NOT include w because a[w] doesn't have bit w.
                    # So if a[w] & common != 0, then there is an edge in the subgraph induced by common.
                    temp = common
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        if a[w] & common:
                            return False
                        temp ^= lsb
        return True

    def popcount(x):
        return bin(x).count('1')
        
    def estimate_max_tf_size(a):
        # Returns a lower bound on the largest triangle-free induced subgraph size
        # by finding large triangle-free subsets via greedy deletion
        best_size = 0
        full_mask = (1 << n) - 1
        
        # Try a few random orders
        for _ in range(10):
            current = full_mask
            # Greedy deletion
            # We need to check if current is triangle-free.
            # If not, remove a vertex from a triangle.
            while True:
                # Find a triangle in current
                found = False
                temp_u = current
                while temp_u:
                    lsb_u = temp_u & -temp_u
                    u = lsb_u.bit_length() - 1
                    temp_u ^= lsb_u
                    
                    # Neighbors of u in current
                    neigh_u = a[u] & current
                    temp_v = neigh_u
                    while temp_v:
                        lsb_v = temp_v & -temp_v
                        v = lsb_v.bit_length() - 1
                        temp_v ^= lsb_v
                        
                        # Common neighbors of u and v in current
                        common = a[u] & a[v] & current
                        if common:
                            # Found triangle (u, v, w) where w is in common
                            # Remove one of them randomly
                            # Let's remove u, v, or the first w
                            choices = [u, v]
                            lsb_w = common & -common
                            w = lsb_w.bit_length() - 1
                            choices.append(w)
                            rem = random.choice(choices)
                            current &= ~(1 << rem)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
            
            sz = popcount(current)
            if sz > best_size:
                best_size = sz
                
        return best_size

    def score(a):
        if not is_k4_free(a):
            return 1000  # Penalty
        return estimate_max_tf_size(a)

    current_adj = adj[:]
    current_score = score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        new_adj = current_adj[:]
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            if new_adj[u] & (1 << v):
                # Remove edge
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                # Add edge
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
        
        new_score = score(new_adj)
        
        if new_score < current_score or random.random() < pow(2.71828, (current_score - new_score) / T):
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final polish: try to remove edges to reduce score further?
    # Or just return best_adj
    
    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges


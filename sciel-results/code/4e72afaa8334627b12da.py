import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def count_k4(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Common neighbors
                    common = adj[i] & adj[j]
                    # Count edges in common neighbors
                    # Iterate over k in common
                    k = common
                    while k:
                        l = k & -k
                        k_idx = l.bit_length() - 1
                        k >>= 1 # Not quite, need to handle bit positions correctly
                        # This loop is tricky. Let's use a simpler O(n^2) check for common neighbors.
                        pass
        # Simpler O(n^3) for initial count, but we want incremental.
        # For initial count, O(n^3) is fine for n<=40.
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1): continue
                for k in range(j+1, n):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1): continue
                    # Check l > k
                    common_l = adj[i] & adj[j] & adj[k]
                    # Mask out bits <= k
                    mask = (1 << n) - 1
                    mask_k = mask ^ ((1 << (k+1)) - 1)
                    count += popcount(common_l & mask_k)
        return count

    def count_i6(adj, n):
        # Count I6 by counting K6 in complement
        # Complement adj: ~adj[i] & mask
        mask = (1 << n) - 1
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1: continue # Edge in G, so non-edge in ~G? No.
                # If edge in G, they are NOT connected in ~G.
                # We want K6 in ~G. So i,j must be connected in ~G, i.e., NOT connected in G.
                # So if (adj[i] >> j) & 1 is 0, proceed.
                for k in range(j+1, n):
                    if (adj[i] >> k) & 1 or (adj[j] >> k) & 1: continue
                    for l in range(k+1, n):
                        if (adj[i] >> l) & 1 or (adj[j] >> l) & 1 or (adj[k] >> l) & 1: continue
                        # Check m > l
                        common_m = (~adj[i]) & (~adj[j]) & (~adj[k]) & (~adj[l]) & mask
                        mask_l = mask ^ ((1 << (l+1)) - 1)
                        count += popcount(common_m & mask_l)
        return count

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_n = 0
    best_result = None
    
    # Try to find a valid graph for n=20, 25, 30, 35
    # We'll use Local Search to minimize cost = K4 + I6
    # If cost == 0, it's valid.
    
    for target_n in [25, 20, 17]:
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        p = 0.4
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        current_k4 = count_k4(adj, target_n)
        current_i6 = count_i6(adj, target_n)
        current_cost = current_k4 + current_i6
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        
        while T > min_T and time.time() < deadline - 5:
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            # Calculate delta for flipping (u,v)
            # Case 1: (u,v) is an edge. We remove it.
            # Case 2: (u,v) is not an edge. We add it.
            
            if (adj[u] >> v) & 1:
                # Remove edge
                # Delta K4: - (number of K4 containing u,v)
                # K4 containing u,v: u,v + w,z where w,z in N(u)∩N(v) and (w,z) is edge.
                common = adj[u] & adj[v]
                # Count edges in common
                delta_k4 = 0
                # Iterate over w in common
                w = common
                while w:
                    l = w & -w
                    w_idx = l.bit_length() - 1
                    w &= w - 1
                    # Check if w_idx > v? No, just count all pairs.
                    # We need to count edges (w,z) where z is in common and z != w.
                    # To avoid double counting, count z > w_idx.
                    # But we are removing the edge, so we lose these triangles.
                    # Each K4 is counted once by the edge (u,v) and the pair (w,z).
                    # So we sum over w in common, count edges (w,z) with z in common, z > w.
                    z = common & ((1 << target_n) - 1)
                    # Mask out w and bits < w
                    mask_z = z ^ ((1 << (w_idx + 1)) - 1)
                    delta_k4 += popcount(adj[w_idx] & mask_z)
                
                # Delta I6: + (number of I6 containing u,v)
                # I6 containing u,v: u,v + 4 others w,x,y,z in N^c(u)∩N^c(v) forming I4 in G.
                # This is K4 in ~G.
                # Non-neighbors in G are neighbors in ~G.
                non_common = (~adj[u]) & (~adj[v]) & ((1 << target_n) - 1)
                # We need to count K4 in ~G induced by non_common.
                # This is expensive to calculate exactly every step.
                # Approximation: We can just recalculate I6 if K4 is low?
                # Or use a heuristic.
                # For now, let's just recalculate I6 if we are close to 0.
                # Actually, let's just use full recalculation for I6 as it's rare to have many I6s in a dense graph?
                # No, I6s are common in sparse graphs.
                
                # Let's stick to full recalc for safety and simplicity, but optimize the loop.
                # The previous attempt was slow.
                # Let's try a different approach:
                # Use a "WalkSAT" style where we flip edges that reduce the number of violations.
                # Violations: K4 and I6.
                
                # Revert flip for now and try a simpler heuristic:
                # Just flip random edges and check validity every 100 steps?
                # That's what the seed does.
                
                # Let's try to improve the seed's Local Search by making the cost function cheaper.
                # Cost = K4 + I6.
                # We can sample? No, exact check required.
                
                # Let's just use the full recalc but in C-speed? Not possible.
                # Let's limit n to 20 for fast search.
                
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_cost = count_k4(adj, target_n) + count_i6(adj, target_n)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost == 0:
                    return {'n': target_n, 'edges': to_edges(adj, target_n)}
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
            steps += 1
            if steps % 1000 == 0:
                 if time.time() > deadline - 5:
                     break

    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': to_edges(adj, 17)}

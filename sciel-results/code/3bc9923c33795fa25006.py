import random
import time
import math

def construct():
    """
    Finds a 5-coloring of K_n with no monochromatic C4 for the largest possible n.
    Uses local search with simulated annealing.
    """
    
    def edge_index(u, v):
        if u > v: u, v = v, u
        return u * n - (u * (u + 1)) // 2 + v - u - 1

    def decode_edge(idx):
        # Find u, v such that idx = u*n - u*(u+1)/2 + v - u - 1
        # This is inverse of the above
        # u is approx sqrt(2*idx)
        # Better: precompute or binary search
        # For n<=29, simple loop is fine
        for u in range(n):
            # max idx for u is when v=n-1
            # idx_max = u*n - u*(u+1)//2 + (n-1) - u - 1
            # if idx <= idx_max:
            #    v = u + 1 + (idx - (u*n - u*(u+1)//2))
            #    return u, v
            base = u * n - (u * (u + 1)) // 2
            if idx < base + (n - 1 - u):
                v = u + 1 + (idx - base)
                return u, v
        return 0, 1 # fallback

    def count_c4_full(colors, n):
        # colors is list of length n(n-1)/2
        # Build adjacency masks
        # masks[c][u] is bitmask of neighbors of u with color c
        masks = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_index(i, j)
                c = colors[idx]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        
        cost = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # count common neighbors of i and j in color c
                    common = masks[c][i] & masks[c][j]
                    cnt = common.bit_count()
                    cost += cnt * (cnt - 1) // 2
        return cost

    def local_search(n, time_limit):
        start_time = time.time()
        best_cost = float('inf')
        best_colors = None
        
        # Try multiple restarts
        max_restarts = 20
        while time.time() - start_time < time_limit and max_restarts > 0:
            max_restarts -= 1
            
            # Random init
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Build masks
            masks = [[0] * n for _ in range(5)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = edge_index(i, j)
                    c = colors[idx]
                    masks[c][i] |= (1 << j)
                    masks[c][j] |= (1 << i)
            
            # Initial cost
            cost = count_c4_full(colors, n)
            
            if cost == 0:
                return colors
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
            
            T = 10.0 # Initial temperature
            T_min = 0.1
            step = 0
            
            while time.time() - start_time < time_limit:
                step += 1
                
                # Pick random edge
                idx = random.randint(0, n * (n - 1) // 2 - 1)
                u, v = decode_edge(idx)
                
                c_old = colors[idx]
                c_new = random.randint(0, 4)
                if c_old == c_new:
                    continue
                
                # Calculate delta
                # Delta = C(|N_new(u) & N_new(v)|, 2) - C(|N_old(u) & N_old(v)|, 2)
                
                # Note: The intersection N(u) & N(v) does NOT include u or v.
                # Changing color of (u,v) does not change the set of OTHER common neighbors.
                # So we just need the count of common neighbors for c_old and c_new.
                
                common_old = masks[c_old][u] & masks[c_old][v]
                cnt_old = common_old.bit_count()
                delta_old = cnt_old * (cnt_old - 1) // 2
                
                common_new = masks[c_new][u] & masks[c_new][v]
                cnt_new = common_new.bit_count()
                delta_new = cnt_new * (cnt_new - 1) // 2
                
                delta = delta_new - delta_old
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    # Update masks
                    masks[c_old][u] &= ~(1 << v)
                    masks[c_old][v] &= ~(1 << u)
                    masks[c_new][u] |= (1 << v)
                    masks[c_new][v] |= (1 << u)
                    
                    colors[idx] = c_new
                    cost += delta
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                        
                    if cost == 0:
                        return colors
                else:
                    # Reject
                    pass
                
                # Cool down
                T *= 0.999
                if T < T_min:
                    T = T_min
                    
        return best_colors if best_cost < float('inf') else None

    # Try n from 26 down to 11
    # Allocate time wisely. 120s total limit.
    # Let's try 26 with 20s, 25 with 15s, etc.
    
    # Actually, just try 26 first. If it fails, try 25, etc.
    # To be safe, we'll use a smaller time limit per n to ensure we return something.
    
    n_target = 26
    time_per_n = 10.0
    
    for n in range(26, 10, -1):
        # Adjust time limit based on remaining time?
        # For simplicity, fixed 10s per n.
        # If we find a solution, return immediately.
        
        # Use a bit more time for larger n?
        # Let's just try.
        
        res = local_search(n, time_per_n)
        if res is not None:
            # Verify
            # The local search returns best_colors which might not be 0 cost if it didn't reach 0.
            # Wait, local_search returns `colors` if cost==0 inside the loop.
            # If it finishes without cost==0, it returns best_colors which has cost > 0.
            # We only want to return if cost is 0.
            
            # Re-check cost of res
            if count_c4_full(res, n) == 0:
                return {'n': n, 'colors': res}
            
    # If nothing found, return the seed solution for n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

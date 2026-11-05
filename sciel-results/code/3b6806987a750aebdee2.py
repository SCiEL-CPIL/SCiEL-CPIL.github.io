import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()

    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    best_n = 1
    best_colors = [0]
    
    # Try high n first to maximize score
    # We know n=26 is possible, but finding it is hard.
    # We iterate from 26 down.
    for target_n in range(26, 7, -1):
        if time_left() < 15:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        
        # Adjust restarts based on time
        # We want to spend most of the time on the largest n we can attempt
        # If we fail 26, we should have time left for 25, etc.
        # But usually, if 26 is hard, 25 is easier.
        
        # Strategy:
        # 1. Try 26 with a limited time budget (e.g. 30s)
        # 2. If found, return.
        # 3. If not, try 25 with remaining time.
        
        # To implement this, we need to know how much time to allocate.
        # Let's allocate proportional to 1/n or just fixed chunks?
        # Fixed chunks are safer.
        # 26: 30s, 25: 20s, 24: 15s, 23: 10s, 22: 5s...
        # Total: 80s. Leaves 35s buffer.
        
        time_budgets = {
            26: 30,
            25: 20,
            24: 15,
            23: 10,
            22: 5,
            21: 5,
            20: 5,
            19: 5,
            18: 5,
            17: 5,
            16: 5,
            15: 5,
            14: 5,
            13: 5,
            12: 5,
            11: 5,
            10: 5,
            9: 5,
            8: 5,
            7: 5
        }
        
        t_limit = time_budgets.get(n, 5)
        if time_left() < t_limit:
             t_limit = max(time_left() - 2, 1)
        
        start_time = time.time()
        
        max_restarts = 100 # We'll control via time
        
        for restart in range(max_restarts):
            if time.time() - start_time > t_limit:
                break
            
            # Random Initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = [[0] * n for _ in range(5)]
            for idx in range(num_edges):
                u, v, e_idx = edge_list[idx]
                c = colors[idx]
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Simulated Annealing
            # Parameters tuned for speed and effectiveness
            temp = 1.0
            cooling = 0.9999
            min_temp = 0.01
            
            # Adaptive cooling?
            # Or fixed steps?
            # Time-based is better.
            
            stuck_count = 0
            max_stuck = 5000
            
            while time.time() - start_time < t_limit:
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    # Restart SA from best state? Or just break to new restart?
                    # Breaking to new restart is simpler.
                    break
                
                ei = random.randint(0, num_edges - 1)
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # Remove from c_old
                # Iterate over neighbors of u and v in c_old
                # This is O(n) per step
                
                # Optimization: Precompute neighbor lists?
                # Bitmask iteration is fast.
                
                # For each w in N(c_old, u):
                #   If w is connected to v in c_old? No, we are removing (u,v).
                #   Wait, the energy term is sum over pairs (i,j) of C(common(i,j), 2).
                #   Removing (u,v) from c_old:
                #   1. It affects common neighbors of (u, w) for w in N(c_old, v) \ {u}
                #      Specifically, v is a common neighbor of u and w.
                #      Removing (u,v) means v is no longer a neighbor of u in c_old.
                #      So common(u, w) decreases by 1 if w is connected to v in c_old.
                #   2. It affects common neighbors of (v, w) for w in N(c_old, u) \ {v}
                #      Similarly, u is no longer a neighbor of v.
                #      So common(v, w) decreases by 1 if w is connected to u in c_old.
                
                # Let's use the bitmask logic from the current program which seemed okay.
                
                mask_u_old = adj[c_old][u]
                mask_v_old = adj[c_old][v]
                
                # Iterate over bits in mask_u_old (neighbors of u in c_old)
                # Note: mask_u_old does NOT contain v yet? 
                # Wait, we are REMOVING (u,v). So currently (u,v) IS in c_old.
                # So mask_u_old contains v. mask_v_old contains u.
                
                # We need to iterate over w != u, v.
                
                # For each w in N(c_old, v) (excluding u):
                #   common(u, w) currently includes v (since v is neighbor of u and w).
                #   After removal, v is not neighbor of u.
                #   So common(u, w) decreases by 1.
                
                # Let's iterate over bits in mask_v_old.
                # mask_v_old has u and other neighbors.
                # We want w in mask_v_old such that w != u.
                
                # We can clear bit u from mask_v_old to get candidates.
                candidates_v = mask_v_old & ~(1 << u)
                
                while candidates_v:
                    # Extract lowest set bit
                    lsb = candidates_v & -candidates_v
                    w = lsb.bit_length() - 1
                    candidates_v ^= lsb
                    
                    # w is a neighbor of v in c_old.
                    # Check if w is neighbor of u in c_old?
                    # If yes, then v is a common neighbor of u and w.
                    # Removing (u,v) removes v from common(u,w).
                    if mask_u_old & (1 << w):
                        # v was a common neighbor.
                        # Old common count k = (mask_u_old & adj[c_old][w]).bit_count()
                        # Note: mask_u_old contains v. adj[c_old][w] contains v (since w~v).
                        # So k includes v.
                        # New common count k' = k - 1.
                        # Energy change: -C(k,2) + C(k-1,2) = -C(k,2) + C(k,2) - (k-1) = -(k-1)
                        # So delta -= (k - 1)
                        
                        k = (mask_u_old & adj[c_old][w]).bit_count()
                        delta -= (k - 1)
                
                # Similarly for neighbors of u
                candidates_u = mask_u_old & ~(1 << v)
                
                while candidates_u:
                    lsb = candidates_u & -candidates_u
                    w = lsb.bit_length() - 1
                    candidates_u ^= lsb
                    
                    # w is neighbor of u in c_old.
                    # Check if w is neighbor of v in c_old?
                    if mask_v_old & (1 << w):
                        # u was a common neighbor of v and w.
                        # Removing (u,v) removes u from common(v,w).
                        # delta -= (k - 1) where k is old common count.
                        
                        k = (mask_v_old & adj[c_old][w]).bit_count()
                        delta -= (k - 1)
                
                # Add to c_new
                mask_u_new = adj[c_new][u]
                mask_v_new = adj[c_new][v]
                
                # Note: (u,v) is NOT in c_new yet.
                # So mask_u_new does NOT contain v. mask_v_new does NOT contain u.
                
                # For each w in N(c_new, v):
                #   If w is neighbor of u in c_new?
                #   If yes, v is NOT currently a common neighbor of u and w (since u-v not in c_new).
                #   After adding (u,v), v becomes a common neighbor.
                #   common(u,w) increases by 1.
                #   Energy change: C(k+1,2) - C(k,2) = k.
                #   So delta += k.
                
                candidates_v_new = mask_v_new
                while candidates_v_new:
                    lsb = candidates_v_new & -candidates_v_new
                    w = lsb.bit_length() - 1
                    candidates_v_new ^= lsb
                    
                    if mask_u_new & (1 << w):
                        k = (mask_u_new & adj[c_new][w]).bit_count()
                        delta += k
                        
                candidates_u_new = mask_u_new
                while candidates_u_new:
                    lsb = candidates_u_new & -candidates_u_new
                    w = lsb.bit_length() - 1
                    candidates_u_new ^= lsb
                    
                    if mask_v_new & (1 << w):
                        k = (mask_v_new & adj[c_new][w]).bit_count()
                        delta += k
                        
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    # Update adj
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break

        if found:
            break

    # Fallback if nothing found (should not happen for n<20)
    if best_n < 20:
        # Use a simple deterministic construction for small n if possible?
        # Or just return what we have.
        # The loop should have found something for n<=20.
        pass

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")

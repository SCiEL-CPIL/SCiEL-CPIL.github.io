"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    n = 40
    best_edges = []
    best_count = 0
    
    def creates_c6(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6."""
        # Neighbors of u excluding v
        nu_mask = adj_bits[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj_bits[v] & ~(1 << u)
        
        # Iterate over potential x1 in N(u)
        temp_u = nu_mask
        while temp_u:
            lsb_u = temp_u & -temp_u
            x1 = lsb_u.bit_length() - 1
            temp_u ^= lsb_u
            
            # Iterate over potential x4 in N(v)
            temp_v = nv_mask
            while temp_v:
                lsb_v = temp_v & -temp_v
                x4 = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if x1 == x4:
                    continue
                
                # x2 must be in N(x1) \ {u, v, x4}
                # x3 must be in N(x4) \ {u, v, x1}
                # And x2 ~ x3
                
                mask_x1 = adj_bits[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x4)
                mask_x4 = adj_bits[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
                
                # Check if any neighbor of x1 (in mask_x1) is connected to any neighbor of x4 (in mask_x4)
                # We can iterate over the smaller mask
                if mask_x1.bit_count() < mask_x4.bit_count():
                    temp_a = mask_x1
                    mask_b = mask_x4
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x2 = lsb_a.bit_length() - 1
                        if (adj_bits[x2] & mask_b) != 0:
                            return True
                        temp_a ^= lsb_a
                else:
                    temp_a = mask_x4
                    mask_b = mask_x1
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x3 = lsb_a.bit_length() - 1
                        if (adj_bits[x3] & mask_b) != 0:
                            return True
                        temp_a ^= lsb_a
        return False

    def greedy_add(adj_bits, current_edges, current_count):
        """Try to add all possible non-C6-creating edges in random order."""
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    candidates.append((u, v))
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 5.0:
                break
            if not creates_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
        return current_edges, current_count

    def simulated_annealing(adj_bits, current_edges, current_count):
        """Simulated Annealing to escape local optima."""
        if not current_edges:
            return current_edges, current_count
            
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
                
            # Pick a random edge to remove
            i = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[i]
            
            # Remove edge
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            current_edges.pop(i)
            new_count = current_count - 1
            
            # Try to add a random non-existent edge
            # To keep it fast, we pick a random pair and check if it's valid
            # If not valid, we might just restore the original edge
            # Better: Try to add the removed edge back OR a new random edge
            
            # Strategy: Try to add a random missing edge. If it fits, we have a swap.
            # If it doesn't fit, we restore the original edge (neutral move) or accept with prob.
            
            # Let's try to add a random missing edge
            attempts = 0
            added = False
            added_u, added_v = -1, -1
            
            while attempts < 10 and not added:
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    attempts += 1
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                    
                if (adj_bits[ru] >> rv) & 1:
                    attempts += 1
                    continue
                    
                # Check if adding (ru, rv) creates C6
                if not creates_c6(adj_bits, ru, rv):
                    # Add it
                    adj_bits[ru] |= (1 << rv)
                    adj_bits[rv] |= (1 << ru)
                    current_edges.append([ru, rv])
                    new_count += 1
                    added = True
                    added_u, added_v = ru, rv
                    break
                else:
                    attempts += 1
            
            if added:
                delta = new_count - current_count
                if delta >= 0 or random.random() < pow(2.718, delta / T):
                    current_count = new_count
                else:
                    # Revert
                    adj_bits[added_u] &= ~(1 << added_v)
                    adj_bits[added_v] &= ~(1 << added_u)
                    # Remove the last added edge (which is [added_u, added_v])
                    # Note: current_edges still has the old edge removed, and new edge added.
                    # We need to restore old edge and remove new edge.
                    current_edges.pop() # Remove [added_u, added_v]
                    current_edges.append([u, v]) # Restore [u, v]
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    current_count = current_count # No change in count effectively, but we reverted state
            else:
                # Could not add any new edge, so restore the removed one
                current_edges.append([u, v])
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_count = current_count # Reverted to original count
            
            T *= cooling
            
        return current_edges, current_count

    # Main loop
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with 8 K5s
        adj_bits = [0] * n
        current_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    current_edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        current_count = len(current_edges)
        
        # Greedy phase
        current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        # SA phase
        if time.time() < deadline - 5.0:
            current_edges, current_count = simulated_annealing(adj_bits, current_edges, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges)
    
    return best_edges
# EVOLVE-BLOCK-END

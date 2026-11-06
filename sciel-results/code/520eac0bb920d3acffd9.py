"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    n = 40
    
    def creates_c6(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6 using bit operations."""
        nu_mask = adj_bits[u] & ~(1 << v)
        nv_mask = adj_bits[v] & ~(1 << u)
        
        temp_u = nu_mask
        while temp_u:
            lsb_u = temp_u & -temp_u
            x1 = lsb_u.bit_length() - 1
            temp_u ^= lsb_u
            
            mask_x1 = adj_bits[x1] & ~(1 << u) & ~(1 << v)
            
            temp_v = nv_mask
            while temp_v:
                lsb_v = temp_v & -temp_v
                x4 = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if x1 == x4:
                    continue
                
                mask_x4 = adj_bits[x4] & ~(1 << u) & ~(1 << v)
                
                mask_x1_eff = mask_x1 & ~(1 << x4)
                mask_x4_eff = mask_x4 & ~(1 << x1)
                
                if mask_x1_eff.bit_count() < mask_x4_eff.bit_count():
                    temp_a = mask_x1_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x2 = lsb_a.bit_length() - 1
                        if (adj_bits[x2] & mask_x4_eff) != 0:
                            return True
                        temp_a ^= lsb_a
                else:
                    temp_a = mask_x4_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x3 = lsb_a.bit_length() - 1
                        if (adj_bits[x3] & mask_x1_eff) != 0:
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

    def local_search(adj_bits, current_edges, current_count):
        """Local search: remove 1-2 edges, try to add multiple new edges."""
        if not current_edges:
            return current_edges, current_count
            
        improved = True
        max_rounds = 80
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Try removing 1 or 2 random edges
            k_remove = random.choice([1, 1, 2])  # Mostly remove 1, sometimes 2
            indices = random.sample(range(len(current_edges)), k_remove)
            
            removed = []
            for idx in sorted(indices, reverse=True):
                u, v = current_edges[idx]
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                removed.append(current_edges.pop(idx))
            
            # Try to add as many edges as possible
            added = []
            for a in range(n):
                if time.time() > deadline - 5.0:
                    break
                for b in range(a + 1, n):
                    if not (adj_bits[a] >> b) & 1:
                        if not creates_c6(adj_bits, a, b):
                            adj_bits[a] |= (1 << b)
                            adj_bits[b] |= (1 << a)
                            added.append((a, b))
            
            new_count = current_count + len(added)
            
            if len(added) > k_remove:
                for a, b in added:
                    current_edges.append([a, b])
                current_count = new_count
                improved = True
            else:
                # Revert
                for u, v in removed:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    current_edges.append([u, v])
                for a, b in added:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                    for j in range(len(current_edges) - 1, -1, -1):
                        if current_edges[j] == [a, b]:
                            current_edges.pop(j)
                            break
        
        return current_edges, current_count

    def simulated_annealing(adj_bits, current_edges, current_count):
        """Simulated Annealing with structured moves."""
        if not current_edges:
            return current_edges, current_count
            
        T = 3.0
        cooling = 0.997
        min_T = 0.05
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
                
            # Randomly remove 1-2 edges
            k_remove = random.choice([1, 1, 2])
            indices = random.sample(range(len(current_edges)), k_remove)
            
            removed = []
            for idx in sorted(indices, reverse=True):
                u, v = current_edges[idx]
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                removed.append(current_edges.pop(idx))
            
            new_count = current_count - k_remove
            
            # Try to add edges
            added_edges = []
            attempts = 0
            max_add_attempts = 40
            
            while attempts < max_add_attempts and len(added_edges) < 4:
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
                    
                if not creates_c6(adj_bits, ru, rv):
                    adj_bits[ru] |= (1 << rv)
                    adj_bits[rv] |= (1 << ru)
                    current_edges.append([ru, rv])
                    added_edges.append((ru, rv))
                    new_count += 1
                attempts += 1
            
            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718, delta / T):
                current_count = new_count
            else:
                for a, b in added_edges:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                    for j in range(len(current_edges) - 1, -1, -1):
                        if current_edges[j] == [a, b]:
                            current_edges.pop(j)
                            break
                for u, v in removed:
                    current_edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
            
            T *= cooling
            
        return current_edges, current_count

    def init_seed(restart_idx):
        """Create diverse initial seeds."""
        adj_bits = [0] * n
        current_edges = []
        
        if restart_idx % 4 == 0:
            # 8 K5s
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u = 5 * k + a
                        v = 5 * k + b
                        current_edges.append([u, v])
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
        elif restart_idx % 4 == 1:
            # Random sparse graph
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.08:
                        current_edges.append([u, v])
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
        elif restart_idx % 4 == 2:
            # Turan-like: partition into 5 groups of 8, connect within groups
            group_size = 8
            for g in range(5):
                for a in range(group_size):
                    for b in range(a + 1, group_size):
                        u = g * group_size + a
                        v = g * group_size + b
                        current_edges.append([u, v])
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
        else:
            # Random + some structure
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.12:
                        current_edges.append([u, v])
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            # Add a few K4s
            for k in range(5):
                base = k * 8
                for a in range(4):
                    for b in range(a + 1, 4):
                        u = base + a
                        v = base + b
                        if not (adj_bits[u] >> v) & 1:
                            current_edges.append([u, v])
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
        
        return adj_bits, current_edges

    # Main loop
    best_edges = []
    best_count = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with diverse seed
        adj_bits, current_edges = init_seed(restart)
        current_count = len(current_edges)
        
        # Greedy phase
        current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        # Local search phase
        if time.time() < deadline - 8.0:
            current_edges, current_count = local_search(adj_bits, current_edges, current_count)
        
        # SA phase
        if time.time() < deadline - 5.0:
            current_edges, current_count = simulated_annealing(adj_bits, current_edges, current_count)
        
        # Final greedy pass
        if time.time() < deadline - 3.0:
            current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges)
    
    return best_edges
# EVOLVE-BLOCK-END

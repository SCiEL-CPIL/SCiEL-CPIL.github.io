import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    def time_ok():
        return time.time() < deadline
    
    # Build Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    def has_k5_bitset(n, adj_bits):
        """Fast K5 check using bitsets."""
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & adj_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def has_ind5_bitset(n, adj_bits):
        """Fast independent 5-set check using bitsets."""
        comp_bits = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & comp_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def is_valid(n, adj_bits):
        return not has_k5_bitset(n, adj_bits) and not has_ind5_bitset(n, adj_bits)
    
    # Verify seed
    if not is_valid(n, adj_bits):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return n, edges
    
    best_n = n
    best_adj_bits = list(adj_bits)
    
    def compute_k4s(n, adj_bits):
        """Compute all K4s in the graph."""
        k4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    k4s.append((i, j, k, l))
        return k4s
    
    def compute_ind4s(n, adj_bits):
        """Compute all Ind4s in the graph."""
        full_mask = (1 << n) - 1
        comp_bits = [0] * n
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        
        ind4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    ind4s.append((i, j, k, l))
        return ind4s
    
    def try_add_vertex_optimized(n, adj_bits, k4s, ind4s, max_attempts=10000):
        """Try to add a vertex using precomputed K4s and Ind4s."""
        all_mask = (1 << n) - 1
        
        for attempt in range(max_attempts):
            if not time_ok():
                break
            
            # Random neighbor mask with slight bias
            if random.random() < 0.3:
                # Try to be more structured
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neighbor_mask |= (1 << i)
            else:
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neighbor_mask |= (1 << i)
            
            non_neighbor_mask = all_mask ^ neighbor_mask
            
            # Check K4s: no K4 should be entirely in neighbors
            valid = True
            for k4 in k4s:
                if all((neighbor_mask >> v) & 1 for v in k4):
                    valid = False
                    break
            
            if valid:
                # Check Ind4s: no Ind4 should be entirely in non-neighbors
                for ind4 in ind4s:
                    if all((non_neighbor_mask >> v) & 1 for v in ind4):
                        valid = False
                        break
            
            if valid:
                # Full validation
                new_adj_bits = list(adj_bits)
                new_adj_bits.append(neighbor_mask)
                for i in range(n):
                    if (neighbor_mask >> i) & 1:
                        new_adj_bits[i] |= (1 << n)
                
                if is_valid(n + 1, new_adj_bits):
                    return True, new_adj_bits
        
        return False, None
    
    current_n = n
    current_adj_bits = list(adj_bits)
    current_k4s = compute_k4s(current_n, current_adj_bits)
    current_ind4s = compute_ind4s(current_n, current_adj_bits)
    
    for _ in range(15):
        if not time_ok():
            break
        success, new_adj_bits = try_add_vertex_optimized(
            current_n, current_adj_bits, current_k4s, current_ind4s
        )
        if success:
            current_adj_bits = new_adj_bits
            current_n += 1
            best_n = current_n
            best_adj_bits = list(current_adj_bits)
            # Recompute K4s and Ind4s
            current_k4s = compute_k4s(current_n, current_adj_bits)
            current_ind4s = compute_ind4s(current_n, current_adj_bits)
        else:
            break
    
    # Try local search: flip bits in the last added vertex's adjacency
    if best_n > n:
        if time_ok():
            for _ in range(5000):
                if not time_ok():
                    break
                
                # Try flipping one bit in the last vertex's adjacency
                last_v = best_n - 1
                new_adj_bits = list(best_adj_bits)
                
                # Pick a random vertex to flip
                v = random.randint(0, last_v - 1)
                new_adj_bits[last_v] ^= (1 << v)
                new_adj_bits[v] ^= (1 << last_v)
                
                if is_valid(best_n, new_adj_bits):
                    # Try to add another vertex
                    k4s = compute_k4s(best_n, new_adj_bits)
                    ind4s = compute_ind4s(best_n, new_adj_bits)
                    success, added = try_add_vertex_optimized(best_n, new_adj_bits, k4s, ind4s, max_attempts=2000)
                    if success:
                        best_adj_bits = added
                        best_n += 1
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges

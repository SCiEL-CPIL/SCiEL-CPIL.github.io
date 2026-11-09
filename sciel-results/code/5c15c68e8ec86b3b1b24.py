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
    # Use bitset representation: adj_bits[i] is an int with bit j set if i~j
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    def has_k4_in_set(n, adj_bits, mask):
        """Check if the induced subgraph on vertices in mask has a K4."""
        verts = [i for i in range(n) if (mask >> i) & 1]
        if len(verts) < 4:
            return False
        for idx_i in range(len(verts)):
            vi = verts[idx_i]
            for idx_j in range(idx_i + 1, len(verts)):
                vj = verts[idx_j]
                if (adj_bits[vi] >> vj) & 1:
                    common = adj_bits[vi] & adj_bits[vj] & mask
                    for idx_k in range(idx_j + 1, len(verts)):
                        vk = verts[idx_k]
                        if (common >> vk) & 1:
                            common2 = common & adj_bits[vk]
                            for idx_l in range(idx_k + 1, len(verts)):
                                vl = verts[idx_l]
                                if (common2 >> vl) & 1:
                                    return True
        return False
    
    def has_ind4_in_set(n, adj_bits, mask):
        """Check if the induced subgraph on vertices in mask has an independent 4-set (K4 in complement)."""
        verts = [i for i in range(n) if (mask >> i) & 1]
        if len(verts) < 4:
            return False
        full_mask = (1 << n) - 1
        comp_adj = [full_mask ^ adj_bits[i] ^ (1 << i) for i in range(n)]
        for idx_i in range(len(verts)):
            vi = verts[idx_i]
            for idx_j in range(idx_i + 1, len(verts)):
                vj = verts[idx_j]
                if (comp_adj[vi] >> vj) & 1:
                    common = comp_adj[vi] & comp_adj[vj] & mask
                    for idx_k in range(idx_j + 1, len(verts)):
                        vk = verts[idx_k]
                        if (common >> vk) & 1:
                            common2 = common & comp_adj[vk]
                            for idx_l in range(idx_k + 1, len(verts)):
                                vl = verts[idx_l]
                                if (common2 >> vl) & 1:
                                    return True
        return False
    
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
        """Fast independent 5-set check using bitsets (clique in complement)."""
        full_mask = (1 << n) - 1
        comp_bits = [(full_mask ^ adj_bits[i]) & ~(1 << i) for i in range(n)]
        
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
        # Fallback: return edges from Paley
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return n, edges
    
    best_n = n
    best_adj_bits = list(adj_bits)
    
    def try_add_vertex(n, adj_bits, max_attempts=5000):
        """Try to add a vertex with valid neighbors."""
        all_mask = (1 << n) - 1
        full_comp = [(all_mask ^ adj_bits[i]) & ~(1 << i) for i in range(n)]
        
        # Try several density biases
        densities = [0.5, 0.45, 0.55, 0.4, 0.6, 0.5]
        
        for attempt in range(max_attempts):
            if not time_ok():
                break
            
            p = densities[attempt % len(densities)]
            
            # Random neighbor mask
            neighbor_mask = 0
            for i in range(n):
                if random.random() < p:
                    neighbor_mask |= (1 << i)
            
            non_neighbor_mask = all_mask ^ neighbor_mask
            
            # Check: induced subgraph on neighbors has no K4
            if neighbor_mask.bit_count() >= 4:
                if has_k4_in_set(n, adj_bits, neighbor_mask):
                    continue
            
            # Check: induced subgraph on non-neighbors has no Ind4
            if non_neighbor_mask.bit_count() >= 4:
                if has_ind4_in_set(n, adj_bits, non_neighbor_mask):
                    continue
            
            # Passed quick checks, do full validation
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
    
    # Try multiple restarts of the addition process
    for restart in range(20):
        if not time_ok():
            break
        
        # Reset to current best and try to add more vertices
        work_n = current_n
        work_adj = list(current_adj_bits)
        
        for _ in range(10):
            if not time_ok():
                break
            success, new_adj_bits = try_add_vertex(work_n, work_adj)
            if success:
                work_adj = new_adj_bits
                work_n += 1
            else:
                break
        
        if work_n > best_n:
            best_n = work_n
            best_adj_bits = list(work_adj)
            current_n = work_n
            current_adj_bits = work_adj
        else:
            # If we couldn't improve, try a different random seed for next restart
            pass
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges

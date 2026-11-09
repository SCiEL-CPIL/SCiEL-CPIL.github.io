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
    
    def has_k5_bitset(n, adj_bits):
        """Fast K5 check using bitsets."""
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    # Remove i and j from common (they're not in each other's neighbor sets in the induced subgraph sense)
                    # Actually, we need k,l,m in common that form a triangle
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & adj_bits[l]
                                    # Check if there's an m > l in common3
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def has_ind5_bitset(n, adj_bits):
        """Fast independent 5-set check using bitsets (clique in complement)."""
        # Complement adjacency
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
        
        for attempt in range(max_attempts):
            if not time_ok():
                break
            
            # Random neighbor mask
            if random.random() < 0.5:
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neighbor_mask |= (1 << i)
            else:
                # Slightly biased
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.45:
                        neighbor_mask |= (1 << i)
            
            non_neighbor_mask = all_mask ^ neighbor_mask
            
            # Check: induced subgraph on neighbors has no K4
            # Check: induced subgraph on non-neighbors has no Ind4 (i.e., K4 in complement)
            
            # Quick size check
            if neighbor_mask.bit_count() >= 4:
                # Check for K4 in neighbor subgraph
                has_k4 = False
                verts = [i for i in range(n) if (neighbor_mask >> i) & 1]
                if len(verts) >= 4:
                    for idx_i in range(len(verts)):
                        vi = verts[idx_i]
                        for idx_j in range(idx_i + 1, len(verts)):
                            vj = verts[idx_j]
                            if (adj_bits[vi] >> vj) & 1:
                                common = adj_bits[vi] & adj_bits[vj] & neighbor_mask
                                for idx_k in range(idx_j + 1, len(verts)):
                                    vk = verts[idx_k]
                                    if (common >> vk) & 1:
                                        common2 = common & adj_bits[vk]
                                        for idx_l in range(idx_k + 1, len(verts)):
                                            vl = verts[idx_l]
                                            if (common2 >> vl) & 1:
                                                has_k4 = True
                                                break
                                        if has_k4:
                                            break
                                if has_k4:
                                    break
                            if has_k4:
                                break
                        if has_k4:
                            break
                if has_k4:
                    continue
            
            if non_neighbor_mask.bit_count() >= 4:
                # Check for Ind4 in non-neighbor subgraph = K4 in complement on non-neighbors
                has_ind4 = False
                verts = [i for i in range(n) if (non_neighbor_mask >> i) & 1]
                if len(verts) >= 4:
                    # Build complement adjacency for these vertices
                    for idx_i in range(len(verts)):
                        vi = verts[idx_i]
                        for idx_j in range(idx_i + 1, len(verts)):
                            vj = verts[idx_j]
                            if not ((adj_bits[vi] >> vj) & 1):
                                common = (all_mask ^ adj_bits[vi]) & (all_mask ^ adj_bits[vj]) & non_neighbor_mask
                                for idx_k in range(idx_j + 1, len(verts)):
                                    vk = verts[idx_k]
                                    if (common >> vk) & 1 and not ((adj_bits[vi] >> vk) & 1) and not ((adj_bits[vj] >> vk) & 1):
                                        common2 = common & (all_mask ^ adj_bits[vk])
                                        for idx_l in range(idx_k + 1, len(verts)):
                                            vl = verts[idx_l]
                                            if (common2 >> vl) & 1 and not ((adj_bits[vi] >> vl) & 1) and not ((adj_bits[vj] >> vl) & 1) and not ((adj_bits[vk] >> vl) & 1):
                                                has_ind4 = True
                                                break
                                        if has_ind4:
                                            break
                                if has_ind4:
                                    break
                            if has_ind4:
                                break
                        if has_ind4:
                            break
                if has_ind4:
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
    
    for _ in range(10):
        if not time_ok():
            break
        success, new_adj_bits = try_add_vertex(current_n, current_adj_bits)
        if success:
            current_adj_bits = new_adj_bits
            current_n += 1
            best_n = current_n
            best_adj_bits = list(current_adj_bits)
        else:
            break
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges

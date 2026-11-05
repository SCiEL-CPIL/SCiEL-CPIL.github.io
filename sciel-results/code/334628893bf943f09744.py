import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    def find_violation(adj, n):
        """Find K5 or independent set of size 5. Returns (type, vertices) or None."""
        # Check for K5
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                    w_bits = common
                    while w_bits:
                        w_bit = w_bits & -w_bits
                        w = w_bit.bit_length() - 1
                        w_bits ^= w_bit
                        
                        common2 = common & adj[w] & ~((1 << u) | (1 << v) | (1 << w))
                        x_bits = common2
                        while x_bits:
                            x_bit = x_bits & -x_bits
                            x = x_bit.bit_length() - 1
                            x_bits ^= x_bit
                            
                            common3 = common2 & adj[x] & ~((1 << u) | (1 << v) | (1 << w) | (1 << x))
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('clique', [u, v, w, x, y])
        
        # Check for Independent Set of size 5
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
            
        for u in range(n):
            for v in range(u + 1, n):
                if non_adj[u] & (1 << v):
                    common = non_adj[u] & non_adj[v] & ~((1 << u) | (1 << v))
                    w_bits = common
                    while w_bits:
                        w_bit = w_bits & -w_bits
                        w = w_bit.bit_length() - 1
                        w_bits ^= w_bit
                        
                        common2 = common & non_adj[w] & ~((1 << u) | (1 << v) | (1 << w))
                        x_bits = common2
                        while x_bits:
                            x_bit = x_bits & -x_bits
                            x = x_bit.bit_length() - 1
                            x_bits ^= x_bit
                            
                            common3 = common2 & non_adj[x] & ~((1 << u) | (1 << v) | (1 << w) | (1 << x))
                            if common3:
                                y_bit = common3 & -common3
                                y = y_bit.bit_length() - 1
                                return ('independent', [u, v, w, x, y])
                                
        return None

    def count_violations(adj, n):
        """Count number of K5 and independent 5-sets (for scoring)."""
        count = 0
        # Count K5s
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                    w_bits = common
                    while w_bits:
                        w_bit = w_bits & -w_bits
                        w = w_bit.bit_length() - 1
                        w_bits ^= w_bit
                        
                        common2 = common & adj[w] & ~((1 << u) | (1 << v) | (1 << w))
                        x_bits = common2
                        while x_bits:
                            x_bit = x_bits & -x_bits
                            x = x_bit.bit_length() - 1
                            x_bits ^= x_bit
                            
                            common3 = common2 & adj[x] & ~((1 << u) | (1 << v) | (1 << w) | (1 << x))
                            if common3:
                                count += 1
                                break  # Only count once per (u,v,w,x)
                    else:
                        continue
                    break
                else:
                    continue
            else:
                continue
            break
        
        # Count independent 5-sets
        non_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
            
        for u in range(n):
            for v in range(u + 1, n):
                if non_adj[u] & (1 << v):
                    common = non_adj[u] & non_adj[v] & ~((1 << u) | (1 << v))
                    w_bits = common
                    while w_bits:
                        w_bit = w_bits & -w_bits
                        w = w_bit.bit_length() - 1
                        w_bits ^= w_bit
                        
                        common2 = common & non_adj[w] & ~((1 << u) | (1 << v) | (1 << w))
                        x_bits = common2
                        while x_bits:
                            x_bit = x_bits & -x_bits
                            x = x_bit.bit_length() - 1
                            x_bits ^= x_bit
                            
                            common3 = common2 & non_adj[x] & ~((1 << u) | (1 << v) | (1 << w) | (1 << x))
                            if common3:
                                count += 1
                                break
                    else:
                        continue
                    break
                else:
                    continue
            else:
                continue
            break
        
        return count

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def local_search(adj, n, time_limit):
        """Improve graph by flipping edges to eliminate violations."""
        deadline = time.time() + time_limit
        while time.time() < deadline:
            viol = find_violation(adj, n)
            if viol is None:
                return adj, True
            
            vtype, verts = viol
            # Try all pairs from the violating set
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    if vtype == 'clique':
                        # Remove edge
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    # Check if this fix works
                    new_viol = find_violation(adj, n)
                    if new_viol is None:
                        return adj, True
                    elif new_viol is not None:
                        # Revert if it created a new violation
                        if vtype == 'clique':
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            
            # If no single flip fixed it, try a random flip
            if time.time() > deadline - 0.5:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj, find_violation(adj, n) is None

    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    best_n = q
    best_adj = adj[:]
    
    # Try to grow by adding vertices
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline - 30:
            break
            
        success = False
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Create new graph with random connections for new vertex
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj[i]
            
            new_idx = target_n - 1
            # Use biased probability based on degree considerations
            for i in range(current_n):
                if random.random() < 0.45:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            # Local search with more time
            local_deadline = min(time.time() + 3.0, deadline - 5)
            new_adj, valid = local_search(new_adj, target_n, local_deadline - time.time())
            
            if valid:
                current_n = target_n
                adj = new_adj
                best_n = current_n
                best_adj = adj[:]
                success = True
                break
        
        if not success:
            # Try more attempts with different random seeds
            for extra_attempt in range(10):
                if time.time() > deadline - 10:
                    break
                    
                new_adj = [0] * target_n
                for i in range(current_n):
                    new_adj[i] = adj[i]
                
                new_idx = target_n - 1
                for i in range(current_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << new_idx)
                        new_adj[new_idx] |= (1 << i)
                
                local_deadline = min(time.time() + 2.0, deadline - 5)
                new_adj, valid = local_search(new_adj, target_n, local_deadline - time.time())
                
                if valid:
                    current_n = target_n
                    adj = new_adj
                    best_n = current_n
                    best_adj = adj[:]
                    success = True
                    break
            
            if not success:
                break  # If we can't grow, stop trying higher N

    # Final verification of best_adj
    if find_violation(best_adj, best_n) is not None:
        # Fallback to Paley
        best_n = 37
        best_adj = [0] * 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)

    return best_n, get_edges(best_adj, best_n)

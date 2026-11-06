import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin

    def find_violation_fast(adj, n):
        """Fast check for K5 or independent 5-set using bit operations."""
        # Check for K5 using nested loops with early termination
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if au & (1 << v):
                    common = au & adj[v] & ~((1 << u) | (1 << v))
                    # Need 3 more vertices from common that form a triangle
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
                                y = (common3 & -common3).bit_length() - 1
                                return ('clique', [u, v, w, x, y])
        
        # Check for independent 5-set
        all_mask = (1 << n) - 1
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = all_mask & ~(1 << i) & ~adj[i]
        
        for u in range(n):
            nu = non_adj[u]
            for v in range(u + 1, n):
                if nu & (1 << v):
                    common = nu & non_adj[v] & ~((1 << u) | (1 << v))
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
                                y = (common3 & -common3).bit_length() - 1
                                return ('independent', [u, v, w, x, y])
        
        return None

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def try_flip(adj, n, u, v):
        """Flip edge (u,v) and return new adj or None if same."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def local_search(adj, n, time_limit):
        """Improve graph by flipping edges to eliminate violations."""
        deadline = time.time() + time_limit
        no_improve_count = 0
        
        while time.time() < deadline:
            viol = find_violation_fast(adj, n)
            if viol is None:
                return adj, True
            
            vtype, verts = viol
            improved = False
            
            # Try all pairs from the violating set
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    # Save state
                    had_edge = bool(adj[u] & (1 << v))
                    
                    # Flip
                    if had_edge:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    # Check if this fix works
                    new_viol = find_violation_fast(adj, n)
                    if new_viol is None:
                        return adj, True
                    elif new_viol is not None:
                        # Revert
                        if had_edge:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            
            # If no single flip from violation worked, try random flips
            for _ in range(20):
                if time.time() > deadline - 0.5:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                try_flip(adj, n, u, v)
                new_viol = find_violation_fast(adj, n)
                if new_viol is None:
                    return adj, True
                # Revert
                try_flip(adj, n, u, v)
            
            no_improve_count += 1
            if no_improve_count > 100:
                break
        
        return adj, find_violation_fast(adj, n) is None

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
        if time.time() > deadline - 40:
            break
            
        success = False
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            # Create new graph with new vertex
            new_adj = [0] * target_n
            for i in range(current_n):
                new_adj[i] = adj[i]
            
            new_idx = target_n - 1
            
            # Use degree-aware connection probability
            # Vertices with higher degree in the existing graph get slightly lower connection probability
            # to avoid creating large cliques
            degrees = [bin(adj[i]).count('1') for i in range(current_n)]
            avg_degree = sum(degrees) / current_n if current_n > 0 else 0
            
            for i in range(current_n):
                # Base probability, adjusted by relative degree
                base_prob = 0.45
                if avg_degree > 0:
                    # Lower probability for high-degree vertices
                    rel_degree = degrees[i] / avg_degree
                    prob = base_prob * (1.0 - 0.1 * (rel_degree - 1.0))
                    prob = max(0.2, min(0.6, prob))
                else:
                    prob = base_prob
                
                if random.random() < prob:
                    new_adj[i] |= (1 << new_idx)
                    new_adj[new_idx] |= (1 << i)
            
            # Local search
            remaining_time = deadline - time.time() - 10
            if remaining_time > 1:
                local_deadline = min(time.time() + 2.5, deadline - 10)
                new_adj, valid = local_search(new_adj, target_n, local_deadline - time.time())
                
                if valid:
                    current_n = target_n
                    adj = new_adj
                    best_n = current_n
                    best_adj = adj[:]
                    success = True
                    break
        
        if not success:
            # Try a few more with different strategies
            for extra_attempt in range(15):
                if time.time() > deadline - 15:
                    break
                    
                new_adj = [0] * target_n
                for i in range(current_n):
                    new_adj[i] = adj[i]
                
                new_idx = target_n - 1
                
                # Alternative strategy: connect to vertices that form independent sets with new vertex
                # First, find which vertices are non-adjacent to each other (potential independent sets)
                # Connect new vertex to break those
                non_adj_pairs = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if not (adj[i] & (1 << j)):
                            non_adj_pairs.append((i, j))
                
                # With some probability, connect to one endpoint of a random non-adjacent pair
                if non_adj_pairs and random.random() < 0.3:
                    i, j = random.choice(non_adj_pairs)
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << new_idx)
                        new_adj[new_idx] |= (1 << i)
                    else:
                        new_adj[j] |= (1 << new_idx)
                        new_adj[new_idx] |= (1 << j)
                
                # Then fill rest randomly
                for i in range(current_n):
                    if not (new_adj[i] & (1 << new_idx)):
                        if random.random() < 0.4:
                            new_adj[i] |= (1 << new_idx)
                            new_adj[new_idx] |= (1 << i)
                
                remaining_time = deadline - time.time() - 10
                if remaining_time > 1:
                    local_deadline = min(time.time() + 2.0, deadline - 10)
                    new_adj, valid = local_search(new_adj, target_n, local_deadline - time.time())
                    
                    if valid:
                        current_n = target_n
                        adj = new_adj
                        best_n = current_n
                        best_adj = adj[:]
                        success = True
                        break
            
            if not success:
                # Try one more time with a fresh random initialization for the new vertex
                if time.time() < deadline - 20:
                    new_adj = [0] * target_n
                    for i in range(current_n):
                        new_adj[i] = adj[i]
                    
                    new_idx = target_n - 1
                    # Completely random
                    for i in range(current_n):
                        if random.random() < 0.5:
                            new_adj[i] |= (1 << new_idx)
                            new_adj[new_idx] |= (1 << i)
                    
                    local_deadline = min(time.time() + 3.0, deadline - 10)
                    new_adj, valid = local_search(new_adj, target_n, local_deadline - time.time())
                    
                    if valid:
                        current_n = target_n
                        adj = new_adj
                        best_n = current_n
                        best_adj = adj[:]
                        success = True
            
            if not success:
                break  # If we can't grow, stop trying higher N

    # Final verification of best_adj
    if find_violation_fast(best_adj, best_n) is not None:
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

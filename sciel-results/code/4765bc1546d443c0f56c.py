import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid_fast(adj, changed_u, changed_v):
        """
        Check if adding edge (changed_u, changed_v) creates a K3,3.
        """
        nu = adj[changed_u] & ~(1 << changed_v)
        nv = adj[changed_v] & ~(1 << changed_u)
        
        bits_nv = nv
        while bits_nv:
            low_a = bits_nv & (-bits_nv)
            a = low_a.bit_length() - 1
            bits_nv ^= low_a
            
            bits_b = nv & ~(1 << a)
            while bits_b:
                low_b = bits_b & (-bits_b)
                b = low_b.bit_length() - 1
                bits_b ^= low_b
                
                common_ab = adj[a] & adj[b]
                candidates = nu & common_ab
                candidates &= ~(1 << a)
                candidates &= ~(1 << b)
                
                if candidates.bit_count() >= 2:
                    return False
        
        return True

    def is_valid_full(adj):
        """
        Full check for K3,3.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """
        Find a violating triple and return (i, j, k, common_neighbours) or None.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k, cijk)
        return None

    def make_valid(adj):
        """Remove edges until valid."""
        for _ in range(1000):
            if is_valid_full(adj):
                return True
            
            viol = find_violation(adj)
            if viol is None:
                return True
            i, j, k, cijk = viol
            x = (cijk & (-cijk)).bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
        
        return is_valid_full(adj)

    def greedy_add(adj):
        added = 0
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_fast(adj, u, v):
                    added += 1
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        return added

    def swap_local_search(adj, time_limit):
        """
        Local search using swap moves (remove one edge, add another).
        Accept moves that maintain validity and don't decrease edge count.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        t_start = time.time()
        
        while time.time() < t_start + time_limit:
            # Pick random existing edge to remove
            u1 = random.randint(0, n-1)
            if current_adj[u1] == 0:
                continue
            v1 = (current_adj[u1] & (-current_adj[u1])).bit_length() - 1
            
            # Pick random non-edge to add
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if current_adj[u2] & (1 << v2):
                continue
            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                continue
            
            # Apply move
            old_u1, old_v1 = current_adj[u1], current_adj[v1]
            old_u2, old_v2 = current_adj[u2], current_adj[v2]
            
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            
            # Check validity
            if is_valid_fast(current_adj, u2, v2):
                # Valid, edge count unchanged
                pass
            else:
                # Revert
                current_adj[u1] = old_u1
                current_adj[v1] = old_v1
                current_adj[u2] = old_u2
                current_adj[v2] = old_v2
                continue
            
            # Try to greedily add after swap
            greedy_add(current_adj)
            new_edges = count_edges(current_adj)
            
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = current_adj[:]
        
        return best_adj, best_edges

    def simulated_annealing(adj, time_limit):
        """
        Simulated annealing with add/remove/swap moves.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        
        t_start = time.time()
        T = 3.0
        cooling_rate = 0.99995
        
        while time.time() < t_start + time_limit:
            move_type = random.random()
            
            if move_type < 0.4:
                # Add or remove a random edge
                u, v = random.choice(all_pairs)
                if current_adj[u] & (1 << v):
                    # Remove
                    old_u, old_v = current_adj[u], current_adj[v]
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    new_edges = current_edges - 1
                    delta_e = 1
                    if random.random() < math.exp(-delta_e / T):
                        current_edges = new_edges
                    else:
                        current_adj[u] = old_u
                        current_adj[v] = old_v
                else:
                    # Add
                    old_u, old_v = current_adj[u], current_adj[v]
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    if is_valid_fast(current_adj, u, v):
                        new_edges = current_edges + 1
                        current_edges = new_edges
                    else:
                        current_adj[u] = old_u
                        current_adj[v] = old_v
            
            elif move_type < 0.7:
                # Swap
                u1 = random.randint(0, n-1)
                if current_adj[u1] == 0:
                    T *= cooling_rate
                    continue
                v1 = (current_adj[u1] & (-current_adj[u1])).bit_length() - 1
                
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    T *= cooling_rate
                    continue
                if current_adj[u2] & (1 << v2):
                    T *= cooling_rate
                    continue
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    T *= cooling_rate
                    continue
                
                old_u1, old_v1 = current_adj[u1], current_adj[v1]
                old_u2, old_v2 = current_adj[u2], current_adj[v2]
                
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                if is_valid_fast(current_adj, u2, v2):
                    pass
                else:
                    current_adj[u1] = old_u1
                    current_adj[v1] = old_v1
                    current_adj[u2] = old_u2
                    current_adj[v2] = old_v2
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
            
            # Update best
            cur_edges = count_edges(current_adj)
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = current_adj[:]
            
            # Periodic greedy add
            if random.random() < 0.01:
                greedy_add(current_adj)
                cur_edges = count_edges(current_adj)
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_adj = current_adj[:]
        
        return best_adj, best_edges

    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    random.seed(42)
    
    # Strategy:
    # 1. Generate diverse valid seeds from different starting densities and structures.
    # 2. Pick the best seed.
    # 3. Run swap-based local search + simulated annealing on the best seed.
    
    candidates = []
    
    # Try various starting densities
    for p in [0.7, 0.6, 0.5, 0.8, 0.4]:
        if time.time() > deadline - 60:
            break
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        if not make_valid(adj):
            continue
        
        greedy_add(adj)
        c = count_edges(adj)
        candidates.append((c, adj))
    
    # Try circulant-like structures
    for shift in [3, 4, 5, 6]:
        if time.time() > deadline - 50:
            break
        adj = [0] * n
        for i in range(n):
            for d in range(1, n):
                if d <= shift or (n - d) <= shift:
                    j = (i + d) % n
                    if i < j:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        if not make_valid(adj):
            continue
        
        greedy_add(adj)
        c = count_edges(adj)
        candidates.append((c, adj))
    
    # Try complete graph start with aggressive repair
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    if make_valid(adj):
        greedy_add(adj)
        c = count_edges(adj)
        candidates.append((c, adj))
    
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_start = candidates[0][1]
        update_best(best_start)
        
        # Run swap-based local search
        remaining = deadline - 20 - time.time()
        if remaining > 5:
            ls_time = min(remaining // 2, 30)
            ls_adj, ls_count = swap_local_search(best_start, ls_time)
            update_best(ls_adj)
            
            # Run simulated annealing on the best result
            remaining = deadline - 10 - time.time()
            if remaining > 5:
                sa_adj, sa_count = simulated_annealing(ls_adj, remaining)
                update_best(sa_adj)
    
    # Fallback
    if not best_edges:
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % 2 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if not is_valid_full(adj):
            return [[i, i+1] for i in range(n-1)]
        return adj_to_edges(adj)
        
    return best_edges

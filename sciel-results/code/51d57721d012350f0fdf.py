import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def elapsed():
        return time.time() - start_time

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_k4(n, adj):
        """Count K4s in graph with adjacency bitmasks."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Count edges within common neighbors
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        count += (adj[x] & common & ((1 << n) - 1)).bit_count()
                    count //= 2
        return count

    def has_k6_in_complement(adj, n):
        """Check if there's an independent set of size 6 in G (K6 in complement)."""
        # Use backtracking with pruning
        def bt(candidates, depth):
            if depth == 6:
                return True
            if len(candidates) < depth:
                return False
            # Pruning: need at least (depth - 1) more from remaining
            for idx in range(len(candidates)):
                v = candidates[idx]
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                if len(new_cands) >= depth - 1:
                    if bt(new_cands, depth + 1):
                        return True
            return False
        return bt(list(range(n)), 0)

    def is_valid(n, adj):
        if count_k4(n, adj) != 0:
            return False
        if has_k6_in_complement(adj, n):
            return False
        return True

    def delta_k4(u, v, adj, n):
        """Change in K4 count when flipping edge (u,v)."""
        is_edge = bool((adj[u] >> v) & 1)
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        # Count edges in common
        deg_sum = 0
        c = common
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            deg_sum += (adj[x] & common).bit_count()
        edges_in_common = deg_sum // 2
        
        if is_edge:
            return -edges_in_common  # Removing edge destroys K4s
        else:
            return edges_in_common   # Adding edge creates K4s

    def compute_k4_count(adj, n):
        """Full K4 count."""
        return count_k4(n, adj)

    def compute_cost_fast(adj, n):
        """Fast cost: K4 count + I6 violation check (boolean)."""
        k4c = count_k4(n, adj)
        i6v = 1 if has_k6_in_complement(adj, n) else 0
        return k4c + 1000 * i6v  # Weight I6 heavily

    # Known 35-vertex graph structure - we'll use random init but try to extend
    # Start with multiple attempts at n=35, then try n=36
    
    best_valid = None
    
    def try_size(target_n, max_restarts=3, time_per_restart=25.0):
        """Try to find valid graph of given size."""
        nonlocal best_valid, deadline, start_time
        total_deadline = deadline - 5  # Leave buffer for final validation
        
        for restart in range(max_restarts):
            if elapsed() > total_deadline:
                break
            
            # Initialize with random graph
            adj = [0] * target_n
            # Use density around 0.5 for balanced approach
            base_density = 0.45 + random.random() * 0.2
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < base_density:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = compute_cost_fast(adj, target_n)
            best_cost = current_cost
            best_adj = adj[:]
            
            # Simulated Annealing with incremental K4 updates
            T = 5.0
            cooling = 0.999
            min_T = 0.1
            
            restart_deadline = min(start_time + elapsed() + time_per_restart, total_deadline)
            
            while T > min_T and elapsed() < restart_deadline:
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Flip edge
                is_edge = bool((adj[u] >> v) & 1)
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Compute new cost
                new_k4 = count_k4(target_n, adj)
                new_i6 = 1 if has_k6_in_complement(adj, target_n) else 0
                new_cost = new_k4 + 1000 * new_i6
                
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj[:]
                        
                        # Early exit if valid
                        if best_cost == 0:
                            if is_valid(target_n, best_adj):
                                if best_valid is None or target_n > best_valid[0]:
                                    best_valid = (target_n, best_adj[:])
                                return True
                else:
                    # Revert
                    if is_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                T *= cooling
            
            # After SA, check best found
            if best_cost == 0 and is_valid(target_n, best_adj):
                if best_valid is None or target_n > best_valid[0]:
                    best_valid = (target_n, best_adj[:])
                return True
            
            # Greedy repair attempt
            if elapsed() < total_deadline - 10:
                adj = best_adj[:]
                improved = True
                repair_deadline = min(elapsed() + 8, total_deadline - 5)
                while improved and elapsed() < repair_deadline:
                    improved = False
                    best_delta = 0
                    best_u = -1
                    best_v = -1
                    
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            dk4 = delta_k4(u, v, adj, target_n)
                            # For I6, we can't easily compute delta, so use heuristic
                            # Just use K4 delta as primary signal
                            if dk4 < best_delta:
                                best_delta = dk4
                                best_u = u
                                best_v = v
                                improved = True
                    
                    if best_u != -1:
                        if (adj[best_u] >> best_v) & 1:
                            adj[best_u] &= ~(1 << best_v)
                            adj[best_v] &= ~(1 << best_u)
                        else:
                            adj[best_u] |= (1 << best_v)
                            adj[best_v] |= (1 << best_u)
                
                if is_valid(target_n, adj):
                    if best_valid is None or target_n > best_valid[0]:
                        best_valid = (target_n, adj[:])
                    return True

    # Try n=35 first (known achievable)
    if elapsed() < deadline - 30:
        try_size(35, max_restarts=4, time_per_restart=22.0)
    
    # If we found valid 35, try to extend to 36
    if best_valid is not None and best_valid[0] >= 35 and elapsed() < deadline - 15:
        # Start from best valid 35-graph and try to add one vertex
        base_n, base_adj = best_valid
        try_size(36, max_restarts=2, time_per_restart=18.0)
    
    # Fallback: try smaller sizes if nothing found
    if best_valid is None:
        for n_try in [30, 28, 26, 24, 22, 20, 17]:
            if elapsed() > deadline - 10:
                break
            try_size(n_try, max_restarts=2, time_per_restart=15.0)
            if best_valid is not None:
                break
    
    # Final fallback: Paley graph on 17 vertices
    if best_valid is None:
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_valid(17, adj):
            best_valid = (17, adj)
    
    if best_valid is not None:
        n, adj = best_valid
        return {'n': n, 'edges': get_edges(adj, n)}
    
    # Ultimate fallback
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': get_edges(adj, 17)}

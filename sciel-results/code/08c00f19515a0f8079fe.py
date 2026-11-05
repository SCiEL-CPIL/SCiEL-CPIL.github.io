import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    n_pairs = len(all_pairs)
    
    best_edges = []
    best_score = -1
    
    # Use integer-encoded adjacency for fast bit operations
    # adj_bits[v] is an integer where bit u is set if edge (u,v) exists
    adj_bits = [0] * n
    
    def get_adj_matrix(adj_bits):
        """Convert adj_bits to 2D list."""
        adj = [[0]*n for _ in range(n)]
        for v in range(n):
            for u in range(n):
                if u != v and (adj_bits[v] >> u) & 1:
                    adj[v][u] = 1
        return adj
    
    def compute_cn_bits(adj_bits):
        """Compute common neighbor counts using bit operations."""
        cn = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                common = adj_bits[u] & adj_bits[v]
                # Remove u and v from consideration (they shouldn't be neighbors of themselves)
                common &= ~(1 << u)
                common &= ~(1 << v)
                cnt = bin(common).count('1')
                cn[u][v] = cnt
                cn[v][u] = cnt
        return cn
    
    def get_penalty(cn):
        """Sum of max(0, cn[u][v] - 2) for all u < v."""
        pen = 0
        for u in range(n):
            for v in range(u+1, n):
                if cn[u][v] > 2:
                    pen += cn[u][v] - 2
        return pen
    
    def can_add_edge(adj_bits, cn, u, v):
        """Check if edge (u,v) can be added without violating K2,3-free constraint."""
        # For each common neighbor w of u and v, we need cn[u][w] < 2 and cn[v][w] < 2
        # Actually, adding edge (u,v) increases cn[u][w] by 1 if w is neighbor of v,
        # and cn[v][w] by 1 if w is neighbor of u.
        # So we need: for all w neighbor of v (w != u), cn[u][w] <= 1
        #              for all w neighbor of u (w != v), cn[v][w] <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj_bits[v] >> w) & 1:
                if cn[u][w] >= 2:
                    return False
            if (adj_bits[u] >> w) & 1:
                if cn[v][w] >= 2:
                    return False
        return True
    
    def try_greedy_fill(adj_bits, cn, max_time):
        """Greedily add edges until no more can be added."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try edges in random order for diversity
            pairs_to_try = all_pairs[:]
            random.shuffle(pairs_to_try)
            for u, v in pairs_to_try:
                if (adj_bits[u] >> v) & 1:
                    continue
                if can_add_edge(adj_bits, cn, u, v):
                    # Add edge (u, v)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    # Update cn
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (adj_bits[v] >> w) & 1:
                            cn[u][w] += 1
                            cn[w][u] += 1
                        if (adj_bits[u] >> w) & 1:
                            cn[v][w] += 1
                            cn[w][v] += 1
                    improved = True
        m = sum(bin(adj_bits[v]).count('1') for v in range(n)) // 2
        return adj_bits, cn, m
    
    def simulated_annealing(adj_bits_init, cn_init, time_budget):
        """Run simulated annealing on the given initial graph."""
        adj_bits = adj_bits_init[:]
        cn = [row[:] for row in cn_init]
        m = sum(bin(adj_bits[v]).count('1') for v in range(n)) // 2
        penalty = get_penalty(cn)
        
        best_m = m
        best_adj_bits = adj_bits[:]
        best_cn = [row[:] for row in cn]
        best_score_local = m - 10 * penalty
        
        T = 5.0
        cooling = 0.997
        min_T = 0.01
        
        steps_per_temp = 150
        
        while T > min_T and time.time() < time_budget:
            for _ in range(steps_per_temp):
                if time.time() > time_budget:
                    break
                
                # Choose move type: 70% single flip, 30% swap
                if random.random() < 0.7:
                    # Single edge flip
                    u, v = random.choice(all_pairs)
                    adding = ((adj_bits[u] >> v) & 1) == 0
                    
                    # Calculate delta penalty
                    delta_pen = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (adj_bits[v] >> w) & 1:
                            old_cn = cn[u][w]
                            new_cn = old_cn + (1 if adding else -1)
                            delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                        if (adj_bits[u] >> w) & 1:
                            old_cn = cn[v][w]
                            new_cn = old_cn + (1 if adding else -1)
                            delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                    
                    delta_m = 1 if adding else -1
                    delta_score = delta_m - 10 * delta_pen
                    
                    if delta_score > 0 or random.random() < math.exp(delta_score / max(T, 0.001)):
                        # Apply flip
                        if adding:
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                        else:
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                        
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if (adj_bits[v] >> w) & 1:
                                cn[u][w] += (1 if adding else -1)
                                cn[w][u] += (1 if adding else -1)
                            if (adj_bits[u] >> w) & 1:
                                cn[v][w] += (1 if adding else -1)
                                cn[w][v] += (1 if adding else -1)
                        
                        m += delta_m
                        penalty += delta_pen
                        
                        current_score = m - 10 * penalty
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_m = m
                            best_adj_bits = adj_bits[:]
                            best_cn = [row[:] for row in cn]
                else:
                    # Swap: remove one edge, add another
                    # Pick a random edge to remove
                    edges_present = [(u, v) for u, v in all_pairs if (adj_bits[u] >> v) & 1]
                    if not edges_present:
                        continue
                    u1, v1 = random.choice(edges_present)
                    
                    # Pick a random non-edge to add
                    non_edges = [(u, v) for u, v in all_pairs if not ((adj_bits[u] >> v) & 1)]
                    if not non_edges:
                        continue
                    u2, v2 = random.choice(non_edges)
                    
                    # Check if swap is valid (after removal, can we add?)
                    # First simulate removal
                    old_cn_u1v1 = cn[u1][v1]
                    
                    # Tentatively remove (u1, v1)
                    adj_bits[u1] &= ~(1 << v1)
                    adj_bits[v1] &= ~(1 << u1)
                    for w in range(n):
                        if w == u1 or w == v1:
                            continue
                        if (adj_bits[v1] >> w) & 1:
                            cn[u1][w] -= 1
                            cn[w][u1] -= 1
                        if (adj_bits[u1] >> w) & 1:
                            cn[v1][w] -= 1
                            cn[w][v1] -= 1
                    m -= 1
                    penalty -= max(0, old_cn_u1v1 - 2)
                    
                    # Now check if we can add (u2, v2)
                    if can_add_edge(adj_bits, cn, u2, v2):
                        # Add edge (u2, v2)
                        adj_bits[u2] |= (1 << v2)
                        adj_bits[v2] |= (1 << u2)
                        for w in range(n):
                            if w == u2 or w == v2:
                                continue
                            if (adj_bits[v2] >> w) & 1:
                                cn[u2][w] += 1
                                cn[w][u2] += 1
                            if (adj_bits[u2] >> w) & 1:
                                cn[v2][w] += 1
                                cn[w][v2] += 1
                        m += 1
                        # Update penalty for new edges
                        for w in range(n):
                            if w == u2 or w == v2:
                                continue
                            if (adj_bits[v2] >> w) & 1:
                                old = cn[u2][w] - 1
                                new = cn[u2][w]
                                penalty += max(0, new - 2) - max(0, old - 2)
                            if (adj_bits[u2] >> w) & 1:
                                old = cn[v2][w] - 1
                                new = cn[v2][w]
                                penalty += max(0, new - 2) - max(0, old - 2)
                        
                        current_score = m - 10 * penalty
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_m = m
                            best_adj_bits = adj_bits[:]
                            best_cn = [row[:] for row in cn]
                    else:
                        # Revert the removal
                        adj_bits[u1] |= (1 << v1)
                        adj_bits[v1] |= (1 << u1)
                        for w in range(n):
                            if w == u1 or w == v1:
                                continue
                            if (adj_bits[v1] >> w) & 1:
                                cn[u1][w] += 1
                                cn[w][u1] += 1
                            if (adj_bits[u1] >> w) & 1:
                                cn[v1][w] += 1
                                cn[w][v1] += 1
                        m += 1
                        penalty += max(0, old_cn_u1v1 - 2)
            
            T *= cooling
        
        # Greedy fill on the best found
        adj_bits_final, cn_final, m_final = try_greedy_fill(best_adj_bits, best_cn, time_budget)
        return adj_bits_final, cn_final, m_final
    
    # Strategy 1: Multiple random initializations with SA
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with varying density
        p = 0.25 + 0.08 * restart
        adj_bits = [0] * n
        for u, v in all_pairs:
            if random.random() < p:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        cn = compute_cn_bits(adj_bits)
        
        time_budget = deadline - 5
        adj_bits_final, cn_final, m_final = simulated_annealing(adj_bits, cn, time_budget)
        
        # Verify validity
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u, v in all_pairs if (adj_bits_final[u] >> v) & 1]
    
    # Strategy 2: Start from sparse graph
    if time.time() < deadline - 10:
        adj_bits = [0] * n
        for _ in range(25):
            u, v = random.choice(all_pairs)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        cn = compute_cn_bits(adj_bits)
        
        time_budget = deadline - 5
        adj_bits_final, cn_final, m_final = simulated_annealing(adj_bits, cn, time_budget)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u, v in all_pairs if (adj_bits_final[u] >> v) & 1]
    
    # Strategy 3: Circulant-like initialization
    if time.time() < deadline - 10:
        adj_bits = [0] * n
        for i in range(n):
            for j in range(1, 5):
                v = (i + j) % n
                if i != v:
                    adj_bits[i] |= (1 << v)
                    adj_bits[v] |= (1 << i)
        
        cn = compute_cn_bits(adj_bits)
        
        time_budget = deadline - 5
        adj_bits_final, cn_final, m_final = simulated_annealing(adj_bits, cn, time_budget)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u, v in all_pairs if (adj_bits_final[u] >> v) & 1]
    
    # Strategy 4: Complete bipartite-like structure (K_{10,10} is K2,3-free? No, it has many common neighbors)
    # Try a different structured approach: random regular-ish graph
    if time.time() < deadline - 10:
        adj_bits = [0] * n
        # Try to create a graph where each vertex has degree ~8
        # Use a greedy approach to build it
        target_deg = 8
        deg = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.5:
                if deg[u] < target_deg and deg[v] < target_deg:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
        
        cn = compute_cn_bits(adj_bits)
        
        time_budget = deadline - 5
        adj_bits_final, cn_final, m_final = simulated_annealing(adj_bits, cn, time_budget)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn_final[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m_final > best_score:
            best_score = m_final
            best_edges = [[u, v] for u, v in all_pairs if (adj_bits_final[u] >> v) & 1]
    
    # Final polish: try to add more edges to the best solution
    if best_edges and time.time() < deadline - 2:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        cn = compute_cn_bits(adj_bits)
        adj_bits, cn, m = try_greedy_fill(adj_bits, cn, deadline - 1)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if cn[u][v] > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and m > best_score:
            best_score = m
            best_edges = [[u, v] for u, v in all_pairs if (adj_bits[u] >> v) & 1]
    
    if not best_edges:
        return []
    
    return best_edges

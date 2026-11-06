"""K2,3-free graph construction on 20 vertices using Simulated Annealing."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    total_possible = n * (n - 1) // 2  # 190
    
    # Precompute all possible edges as indices
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def count_violations(adj_bits):
        """Count number of pairs (u,v) with >= 3 common neighbors."""
        viol = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = bin(adj_bits[i] & adj_bits[j]).count('1')
                if common >= 3:
                    viol += 1
        return viol
    
    def is_valid(adj_bits):
        """Check K2,3-freeness: no pair has >= 3 common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                common = bin(adj_bits[i] & adj_bits[j]).count('1')
                if common >= 3:
                    return False
        return True
    
    def edge_count(adj_bits):
        """Count edges from adjacency bitmasks."""
        count = 0
        for i in range(n):
            count += bin(adj_bits[i]).count('1')
        return count // 2
    
    def remove_violations(adj_bits, edges_set, rng):
        """Remove edges until the graph is K2,3-free."""
        changed = True
        while changed:
            changed = False
            # Find a violating pair
            for i in range(n):
                for j in range(i + 1, n):
                    common = bin(adj_bits[i] & adj_bits[j]).count('1')
                    if common >= 3:
                        # Remove a random edge among common neighbors
                        common_mask = adj_bits[i] & adj_bits[j]
                        common_list = []
                        mask = common_mask
                        while mask:
                            low = mask & (-mask)
                            w = low.bit_length() - 1
                            common_list.append(w)
                            mask ^= low
                        # Remove edge (i, w) or (j, w) randomly
                        w = rng.choice(common_list)
                        if rng.random() < 0.5:
                            u, v = i, w
                        else:
                            u, v = j, w
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        edges_set.discard((min(u,v), max(u,v)))
                        changed = True
                        break
                if changed:
                    break
        return adj_bits, edges_set
    
    def sa_search(adj_bits, edges_set, rng, time_budget):
        """
        Simulated Annealing to maximize edges while maintaining K2,3-freeness.
        Moves: add a random missing edge, remove a random present edge,
        or swap (remove one, add another).
        Acceptance: only accept if new state is valid (0 violations) AND 
        has more edges, or accept with probability exp(-delta/T) for equal/fewer.
        """
        T0 = 5.0
        Tend = 0.01
        alpha = 0.995  # cooling rate
        
        best_edges = set(edges_set)
        best_adj = list(adj_bits)
        best_count = edge_count(adj_bits)
        
        cur_adj = list(adj_bits)
        cur_edges = set(edges_set)
        cur_count = edge_count(cur_adj)
        
        T = T0
        no_improve = 0
        
        while time.time() < time_budget:
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add a random missing edge
                missing = [(u, v) for (u, v) in all_edges if (u, v) not in cur_edges]
                if not missing:
                    T *= alpha
                    continue
                u, v = rng.choice(missing)
                # Check if adding (u,v) keeps graph valid
                # After adding: common(u,v) must be <= 2
                common_uv = bin(cur_adj[u] & cur_adj[v]).count('1')
                if common_uv > 2:
                    T *= alpha
                    continue
                # Check all w in N(u) \ {v}: common(v,w) must stay <= 2
                # After adding u-v, v becomes common neighbor of u and w
                valid = True
                nu = cur_adj[u]
                w = nu
                while w:
                    low = w & (-w)
                    wi = low.bit_length() - 1
                    w ^= low
                    if wi == v:
                        continue
                    common_vw = bin(cur_adj[v] & cur_adj[wi]).count('1')
                    if common_vw > 1:
                        valid = False
                        break
                if valid:
                    nv = cur_adj[v]
                    w = nv
                    while w:
                        low = w & (-w)
                        wi = low.bit_length() - 1
                        w ^= low
                        if wi == u:
                            continue
                        common_uw = bin(cur_adj[u] & cur_adj[wi]).count('1')
                        if common_uw > 1:
                            valid = False
                            break
                if not valid:
                    T *= alpha
                    continue
                
                # Apply move
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
                cur_edges.add((u, v))
                new_count = cur_count + 1
                
                # Always accept improvement
                if new_count > best_count:
                    best_count = new_count
                    best_edges = set(cur_edges)
                    best_adj = list(cur_adj)
                    no_improve = 0
                else:
                    no_improve += 1
                
                cur_count = new_count
                
            elif move_type < 0.8:
                # Try to remove a random present edge (to allow restructuring)
                if not cur_edges:
                    T *= alpha
                    continue
                u, v = rng.choice(list(cur_edges))
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                cur_edges.discard((u, v))
                cur_count -= 1
                no_improve += 1
                
            else:
                # Swap: remove one edge, add another
                if not cur_edges:
                    T *= alpha
                    continue
                u1, v1 = rng.choice(list(cur_edges))
                missing = [(u, v) for (u, v) in all_edges if (u, v) not in cur_edges]
                if not missing:
                    T *= alpha
                    continue
                u2, v2 = rng.choice(missing)
                
                # Apply removal
                cur_adj[u1] &= ~(1 << v1)
                cur_adj[v1] &= ~(1 << u1)
                cur_edges.discard((u1, v1))
                
                # Try to add
                common_uv = bin(cur_adj[u2] & cur_adj[v2]).count('1')
                if common_uv <= 2:
                    valid = True
                    nu = cur_adj[u2]
                    w = nu
                    while w:
                        low = w & (-w)
                        wi = low.bit_length() - 1
                        w ^= low
                        if wi == v2:
                            continue
                        common_vw = bin(cur_adj[v2] & cur_adj[wi]).count('1')
                        if common_vw > 1:
                            valid = False
                            break
                    if valid:
                        nv = cur_adj[v2]
                        w = nv
                        while w:
                            low = w & (-w)
                            wi = low.bit_length() - 1
                            w ^= low
                            if wi == u2:
                                continue
                            common_uw = bin(cur_adj[u2] & cur_adj[wi]).count('1')
                            if common_uw > 1:
                                valid = False
                                break
                    if valid:
                        cur_adj[u2] |= (1 << v2)
                        cur_adj[v2] |= (1 << u2)
                        cur_edges.add((u2, v2))
                        # Net change in count is 0
                    else:
                        # Revert the removal
                        cur_adj[u1] |= (1 << v1)
                        cur_adj[v1] |= (1 << u1)
                        cur_edges.add((u1, v1))
                else:
                    # Revert the removal
                    cur_adj[u1] |= (1 << v1)
                    cur_adj[v1] |= (1 << u1)
                    cur_edges.add((u1, v1))
            
            T *= alpha
            if T < Tend:
                T = Tend
        
        return best_adj, best_edges, best_count
    
    # Main: multiple restarts with SA
    best_edges_overall = []
    best_count_overall = 0
    
    rng = random.Random(42)
    
    # Restart 1: Start from a moderately dense random graph, clean up, then SA
    if time.time() < deadline - 5:
        adj_bits = [0] * n
        edges_set = set()
        # Add random edges at ~50% density
        for (u, v) in all_edges:
            if rng.random() < 0.5:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges_set.add((u, v))
        # Remove violations
        adj_bits, edges_set = remove_violations(adj_bits, edges_set, rng)
        # SA
        t_budget = deadline - 10
        adj_bits, edges_set, cnt = sa_search(adj_bits, edges_set, rng, t_budget)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_edges_overall = list(edges_set)
    
    # Restart 2: Start from sparser random
    if time.time() < deadline - 5:
        adj_bits = [0] * n
        edges_set = set()
        for (u, v) in all_edges:
            if rng.random() < 0.3:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges_set.add((u, v))
        adj_bits, edges_set = remove_violations(adj_bits, edges_set, rng)
        t_budget = deadline - 5
        adj_bits, edges_set, cnt = sa_search(adj_bits, edges_set, rng, t_budget)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_edges_overall = list(edges_set)
    
    # Restart 3: Start from a structured graph (e.g., complement of a matching)
    if time.time() < deadline - 5:
        adj_bits = [0] * n
        edges_set = set()
        # Start with all edges except a random matching
        matching = list(range(n))
        rng.shuffle(matching)
        matched_pairs = set()
        for i in range(0, n - 1, 2):
            u, v = matching[i], matching[i+1]
            matched_pairs.add((min(u,v), max(u,v)))
        for (u, v) in all_edges:
            if (u, v) not in matched_pairs:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges_set.add((u, v))
        adj_bits, edges_set = remove_violations(adj_bits, edges_set, rng)
        t_budget = deadline - 3
        adj_bits, edges_set, cnt = sa_search(adj_bits, edges_set, rng, t_budget)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_edges_overall = list(edges_set)
    
    # Final cleanup: verify and return
    if best_edges_overall:
        # Verify validity
        adj_bits = [0] * n
        for u, v in best_edges_overall:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        if not is_valid(adj_bits):
            # Fallback: remove violations
            edges_set = set(best_edges_overall)
            adj_bits, edges_set = remove_violations(adj_bits, edges_set, rng)
            best_edges_overall = list(edges_set)
    
    return best_edges_overall

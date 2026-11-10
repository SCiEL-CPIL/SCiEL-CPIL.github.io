import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    total_possible = n * (n - 1) // 2
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[u] & adj[wi]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[v] & adj[wi]) > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
        
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def greedy_build(adj):
        # Randomly add edges while maintaining K2,3-free property
        order = list(range(n * (n - 1) // 2))
        random.shuffle(order)
        # Map index to (u,v) pair
        pair_map = {}
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                pair_map[idx] = (u, v)
                idx += 1
                
        for i in order:
            if time.time() > deadline:
                break
            u, v = pair_map[i]
            if can_add(adj, u, v):
                add_edge(adj, u, v)

    def local_search(adj, max_iters=5000):
        # Try to improve by adding edges
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try to add a random missing edge
            for _ in range(100):
                if time.time() > deadline:
                    return adj
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break
        return adj

    def simulated_annealing(adj, max_iters=20000, initial_temp=1.0, cooling=0.9995):
        # Simulated annealing: accept worse solutions with probability exp(-delta/temp)
        # Moves: add edge, remove edge, or swap (remove one, add another)
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random missing edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (current_adj[u] >> v) & 1:
                    continue
                if can_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                # Note: We don't revert if can_add fails, we just skip.
                # We don't "accept worse" for adds because adds are always better if valid.
                # SA is mainly for swaps/removals to escape local optima.
                
            elif move_type < 0.7:
                # Try to remove a random edge (to escape local optima)
                # Pick a random existing edge
                all_edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        lsb = v_mask & (-v_mask)
                        v_mask ^= lsb
                        v = lsb.bit_length() - 1
                        if u < v:
                            all_edges.append((u, v))
                if not all_edges:
                    continue
                u, v = random.choice(all_edges)
                remove_edge(current_adj, u, v)
                new_edges = current_edges - 1
                delta = new_edges - current_edges # -1
                # Accept with probability exp(-delta/temp) = exp(1/temp)
                # But we want to accept *worse* solutions.
                # Standard SA: accept if rand < exp(-Delta E / T).
                # Here Delta E = -1 (energy increases if we minimize energy, but we maximize edges).
                # Let's stick to: accept if rand < exp(-(-1)/T) = exp(1/T).
                # This is > 1, so always accept? No, that's for minimizing.
                # If we maximize, removing an edge is bad. We should accept with small prob.
                # Prob = exp(-1/T) is small when T is small.
                # The previous code used exp(-delta/T) where delta = -1, so exp(1/T).
                # This is huge. That was the bug.
                # Correct logic for maximizing: accept worse (remove) with prob exp(-1/T).
                if random.random() < math.exp(-1.0 / temp):
                    current_edges = new_edges
                else:
                    add_edge(current_adj, u, v)
                    
            else:
                # Swap: remove one edge, try to add another
                # Pick a random existing edge to remove
                all_edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        lsb = v_mask & (-v_mask)
                        v_mask ^= lsb
                        v = lsb.bit_length() - 1
                        if u < v:
                            all_edges.append((u, v))
                if not all_edges:
                    continue
                u_rem, v_rem = random.choice(all_edges)
                
                # Pick a random missing edge to add
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add or (u_add == u_rem and v_add == v_rem) or (u_add == v_rem and v_add == u_rem):
                    continue
                if (current_adj[u_add] >> v_add) & 1:
                    continue
                    
                # Remove the first edge
                remove_edge(current_adj, u_rem, v_rem)
                
                # Check if we can add the second edge
                if can_add(current_adj, u_add, v_add):
                    add_edge(current_adj, u_add, v_add)
                    # Net change: 0 edges.
                    # Accept swap with probability 0.5 to explore, or based on some metric?
                    # Since edge count is same, we accept to change structure.
                    if random.random() < 0.5:
                        pass
                    else:
                        remove_edge(current_adj, u_add, v_add)
                        add_edge(current_adj, u_rem, v_rem)
                else:
                    # Revert
                    add_edge(current_adj, u_rem, v_rem)
                    
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
                
        return best_adj, best_edges

    # Multi-start approach with simulated annealing
    best_adj = [0] * n
    best_edges = 0
    
    num_starts = 30
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        greedy_build(adj)
        adj = local_search(adj)
        
        # Apply simulated annealing
        sa_adj, sa_edges = simulated_annealing(adj)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]
            
    # Convert best_adj to edge list
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END

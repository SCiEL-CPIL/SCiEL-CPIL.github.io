import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    n = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute edge index for fast lookup
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def is_k33_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is K3,3-free.
        A K3,3 exists if there are 3 vertices A={a1,a2,a3} and 3 vertices B={b1,b2,b3}
        such that all 9 edges between A and B exist.
        Equivalent: No 3 vertices have 3 common neighbours.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij == 0:
                    continue
                # If common_ij has < 3 bits, it can't be part of a K3,3 with k
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def count_violations(adj):
        """
        Count the number of triples (i,j,k) such that |N(i) ∩ N(j) ∩ N(k)| >= 3.
        This is a smooth-ish surrogate for the constraint.
        """
        count = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij == 0:
                    continue
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        count += 1
        return count

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def greedy_construct(p_order):
        """
        Build a valid graph by adding edges in a specific order if they don't violate K3,3.
        p_order is a list of (u,v) pairs.
        """
        adj = [0] * n
        
        for u, v in p_order:
            if (adj[u] >> v) & 1:
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_k33_free(adj):
                # Keep it
                pass
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def local_search_add(adj):
        """
        Try to add any missing edge that keeps the graph K3,3-free.
        Returns (adj, added) where added is bool.
        """
        missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
        if not missing:
            return adj, False
        
        random.shuffle(missing)
        for u, v in missing:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_k33_free(adj):
                return adj, True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj, False

    def local_search_swap(adj):
        """
        Try a swap: remove one edge, add another.
        Returns (adj, swapped) where swapped is bool.
        """
        existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
        missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
        
        if not existing or not missing:
            return adj, False
        
        random.shuffle(existing)
        random.shuffle(missing)
        
        # Limit attempts for speed
        max_attempts = 50
        attempts = 0
        for u1, v1 in existing:
            if attempts >= max_attempts:
                break
            for u2, v2 in missing:
                if attempts >= max_attempts:
                    break
                attempts += 1
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    continue
                
                # Perform swap
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                if is_k33_free(adj):
                    return adj, True
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
        return adj, False

    def simulated_annealing(adj):
        """
        Simulated Annealing to optimize edge count while maintaining K3,3-freeness.
        Moves: add edge, remove edge, swap edges.
        Energy = -edge_count + penalty * violations (but we enforce violations=0 strictly).
        Actually, we only accept moves that result in a valid graph (violations=0).
        So it's a random walk in the space of valid graphs.
        We accept if new_edge_count >= old_edge_count (always if valid).
        To escape local minima, we might allow removing an edge (decreasing count) with some probability.
        """
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling = 0.995
        
        T = T_start
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T <= T_end:
                break
                
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add an edge
                u, v = random.choice(all_edges)
                if (current_adj[u] >> v) & 1:
                    continue
                new_adj = current_adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if is_k33_free(new_adj):
                    new_edges = current_edges + 1
                    # Always accept improvement
                    current_adj = new_adj
                    current_edges = new_edges
                else:
                    # Reject invalid move
                    pass
                    
            elif move_type < 0.6:
                # Try to remove an edge (to escape local minima)
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (current_adj[u] >> v) & 1]
                if not existing:
                    continue
                u, v = random.choice(existing)
                new_adj = current_adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                # Removal always keeps graph valid (subgraph of valid graph)
                new_edges = current_edges - 1
                # Accept with probability exp((new_edges - current_edges) / T) = exp(-1/T)
                # Since new_edges < current_edges, delta = -1
                prob = random.random()
                if prob < (T / (T + 1.0)): # Simplified acceptance
                    current_adj = new_adj
                    current_edges = new_edges
                    
            else:
                # Try a swap
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (current_adj[u] >> v) & 1]
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (current_adj[u] >> v) & 1]
                
                if existing and missing:
                    u1, v1 = random.choice(existing)
                    u2, v2 = random.choice(missing)
                    
                    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                        continue
                        
                    new_adj = current_adj[:]
                    new_adj[u1] &= ~(1 << v1)
                    new_adj[v1] &= ~(1 << u1)
                    new_adj[u2] |= (1 << v2)
                    new_adj[v2] |= (1 << u2)
                    
                    if is_k33_free(new_adj):
                        new_edges = current_edges # Swap keeps count same
                        # Accept with probability exp(0) = 1.0? 
                        # To allow exploration, we accept with prob 1.0 if valid.
                        current_adj = new_adj
                        # current_edges unchanged
                    
            T *= cooling
            
        return current_adj

    best_edges = 0
    best_adj = None

    # Strategy: 
    # 1. Greedy construction with random restarts
    # 2. Local search (add edges)
    # 3. Simulated Annealing to escape local minima
    
    while time.time() < deadline - 10.0:
        # Create a random order of edges
        random_order = all_edges[:]
        random.shuffle(random_order)
        
        # Greedy phase
        adj = greedy_construct(random_order)
        
        # Local Search phase: aggressively add edges
        improved = True
        while improved and time.time() < deadline - 10.0:
            adj, improved = local_search_add(adj)
            
        # Simulated Annealing phase
        adj = simulated_annealing(adj)
        
        # Final local search after SA
        improved = True
        while improved and time.time() < deadline - 10.0:
            adj, improved = local_search_add(adj)
            
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
        # If we have a high score, do more SA on the best one
        if best_edges > 70 and time.time() < deadline - 5.0:
            adj_copy = best_adj[:]
            adj_copy = simulated_annealing(adj_copy)
            improved = True
            while improved and time.time() < deadline - 5.0:
                adj_copy, improved = local_search_add(adj_copy)
            ec = edge_count(adj_copy)
            if ec > best_edges:
                best_edges = ec
                best_adj = adj_copy[:]

    if best_adj:
        return get_edges(best_adj)
    else:
        # Fallback: empty graph is valid
        return []

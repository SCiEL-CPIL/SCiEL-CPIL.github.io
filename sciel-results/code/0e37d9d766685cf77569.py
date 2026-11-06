import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def creates_c4(adj, u, v):
        """Check if adding edge (u,v) creates a C4."""
        au = adj[u]
        av = adj[v]
        
        if au == 0 or av == 0:
            return False
            
        temp = au
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if adj[w] & av:
                return True
            temp ^= lsb
            
        return False

    def is_c4_free(adj, n):
        """Verify the entire graph is C4-free."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check all common neighbors of i and j
                    common = adj[i] & adj[j]
                    if common:
                        # If there are 2+ common neighbors, we have a C4
                        temp = common
                        count = 0
                        while temp:
                            temp &= temp - 1
                            count += 1
                            if count >= 2:
                                return False
        return True

    def min_degree(adj, n):
        """Compute minimum degree."""
        min_deg = n
        for i in range(n):
            temp = adj[i]
            deg = 0
            while temp:
                temp &= temp - 1
                deg += 1
            if deg < min_deg:
                min_deg = deg
        return min_deg

    def try_finite_field_construction(n, p, field_size, time_limit):
        """
        Try to construct a C4-free graph using finite field structure.
        
        For GF(p^k), we can create a graph where vertices are elements of the field,
        and two vertices are adjacent if their difference is in a specific set.
        
        This is a heuristic approach - we try different parameter combinations.
        """
        if time.time() > time_limit:
            return None
        
        # Strategy 1: Use a Paley-type construction with modifications
        # For n=47, try to build a graph where most vertices have high degree
        # but avoid C4s by careful edge selection
        
        adj = [0] * n
        degrees = [0] * n
        
        # Start with a random greedy construction, then try to optimize
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > time_limit:
                return None
            if not creates_c4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
        
        # Local search to improve minimum degree
        min_deg_req = n - 39
        ls_deadline = min(time.time() + 2.0, time_limit)
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > ls_deadline:
                break
                
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                break
                
            # Find vertices with minimum degree
            min_vertices = [i for i in range(n) if degrees[i] == min_deg]
            u = random.choice(min_vertices)
            
            # Try to add an edge to u
            non_neigh = [v for v in range(n) if v != u and not (adj[u] & (1 << v))]
            random.shuffle(non_neigh)
            
            added = False
            for v in non_neigh:
                if not creates_c4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break
                    
            if not added:
                # Kick: remove a random edge
                if degrees[u] > 0:
                    neigh = []
                    temp = adj[u]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        neigh.append(w)
                        temp ^= lsb
                    
                    if neigh:
                        w = random.choice(neigh)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        degrees[u] -= 1
                        degrees[w] -= 1
        
        # Check if we succeeded
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            # Verify C4-free
            if is_c4_free(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return edges
        
        return None

    def try_simulated_annealing(n, time_limit):
        """
        Use simulated annealing to find a C4-free graph with high minimum degree.
        """
        min_deg_req = n - 39
        if min_deg_req <= 0:
            return []
        
        # Initialize with random graph
        adj = [0] * n
        degrees = [0] * n
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        # Start with a sparse random graph
        num_initial = int(n * (n-1) * 0.15 / 2)
        for u, v in all_edges[:num_initial]:
            if not creates_c4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
        
        # Simulated annealing
        best_adj = [a[:] for a in adj]
        best_degrees = degrees[:]
        best_min_deg = min(degrees)
        
        temperature = 10.0
        cooling = 0.999
        sa_deadline = time_limit - 5
        
        for step in range(100000):
            if time.time() > sa_deadline:
                break
                
            # Pick a random edge operation
            op = random.random()
            
            if op < 0.6:
                # Try to add an edge
                u, v = random.randint(0, n-1), random.randint(0, n-1)
                if u == v or (adj[u] & (1 << v)):
                    continue
                
                if not creates_c4(adj, u, v):
                    # Always accept if it improves min degree or doesn't hurt
                    old_min = min(degrees)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    new_min = min(degrees)
                    
                    if new_min >= old_min:
                        if new_min > best_min_deg:
                            best_min_deg = new_min
                            best_adj = [a[:] for a in adj]
                            best_degrees = degrees[:]
                    else:
                        # Accept with probability based on temperature
                        if random.random() < np.exp((old_min - new_min) / temperature):
                            if new_min > best_min_deg:
                                best_min_deg = new_min
                                best_adj = [a[:] for a in adj]
                                best_degrees = degrees[:]
                        else:
                            # Reject
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            degrees[u] -= 1
                            degrees[v] -= 1
                else:
                    # Adding creates C4 - only accept if we remove another edge
                    pass
            else:
                # Try to remove an edge
                u, v = random.randint(0, n-1), random.randint(0, n-1)
                if u == v or not (adj[u] & (1 << v)):
                    continue
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                degrees[u] -= 1
                degrees[v] -= 1
        
        # Check best solution
        if best_min_deg >= min_deg_req:
            if is_c4_free(best_adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if best_adj[i] & (1 << j):
                            edges.append([i, j])
                return edges
        
        return None

    # Try different strategies for n=47, 46, 45
    for target_n in [47, 46, 45]:
        if time.time() > deadline - 15:
            break
        
        # Strategy 1: Finite field / greedy with local search
        edges = try_finite_field_construction(target_n, 2, 64, deadline - 10)
        if edges is not None:
            return {'n': target_n, 'edges': edges}
        
        # Strategy 2: Simulated annealing
        edges = try_simulated_annealing(target_n, deadline - 5)
        if edges is not None:
            return {'n': target_n, 'edges': edges}
    
    # Fallback: return n=39 (trivially valid)
    return {'n': 39, 'edges': []}

import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety

    def is_c4_free_fast(adj):
        """
        Check if a graph is C4-free.
        Returns True if no C4 exists.
        Uses the property: C4 exists iff there exist u != v with |N(u) ∩ N(v)| >= 2.
        """
        n = adj.shape[0]
        # Common neighbors count
        # A @ A gives counts of common neighbors
        # But we need to be careful with self-loops (diagonal)
        # A[i,j] = 1 if edge.
        # (A @ A)[i,j] = sum_k A[i,k] A[k,j]
        # This counts paths of length 2.
        # A C4 is a-u-b-v-a. This implies u and v have at least 2 common neighbors (a and b).
        # So if any off-diagonal entry of A@A is >= 2, there is a C4.
        A = adj.astype(np.int32)
        A2 = A @ A
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        # If any off-diagonal element is >= 2, C4 exists
        return not np.any(A2[mask] >= 2)

    def count_c4s(adj):
        """
        Count the number of C4s (roughly). 
        Number of C4s = sum_{u<v} C(common_neighbors(u,v), 2)
        """
        n = adj.shape[0]
        A = adj.astype(np.int32)
        A2 = A @ A
        # For each pair u,v, if common neighbors is k, it contributes k*(k-1)/2 to C4 count?
        # No, a specific set of 4 vertices {u,v,a,b} forms a C4 if u~a, a~v, v~b, b~u.
        # Actually, the number of C4s is 1/2 * sum_{u != v} C(|N(u) ∩ N(v)|, 2).
        # Let's just compute the sum of squares of common neighbors off-diagonal.
        # This is a proxy for "badness".
        mask = ~np.eye(n, dtype=bool)
        common_counts = A2[mask]
        # Sum of k*(k-1)/2
        return int(np.sum(common_counts * (common_counts - 1) // 2))

    def get_min_degree(adj):
        if adj.size == 0:
            return 0
        return int(np.min(adj.sum(axis=1)))

    def random_c4_free_init(n):
        """
        Generate a random C4-free graph.
        We use a greedy approach but with random restarts and shuffling.
        """
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Random order of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates C4
            # u and v must have <= 1 common neighbor
            common = np.sum(adj[u] & adj[v])
            if common <= 1:
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
        
        return adj

    def simulated_annealing(n, target_deg, time_limit=10.0):
        """
        Simulated Annealing to maximize min degree while keeping C4-free.
        State: adjacency matrix.
        Moves: 
          1. Add edge (u,v) if C4-free.
          2. Remove edge (u,v).
          3. Swap: Remove (u,v), Add (u',v') if C4-free.
        Energy: -min_degree + penalty * (num_c4s)
        We want to minimize energy (maximize min degree, keep c4s=0).
        """
        # Initialize with a random C4-free graph
        adj = random_c4_free_init(n)
        current_min_deg = get_min_degree(adj)
        
        # If we already hit target, we are done? 
        # But we want to maximize n, so we try to hit target for n=46.
        # If we hit target, we can return immediately? 
        # The task says "Maximize n". 
        # If we find a valid graph for n, we return it.
        # The loop in construct() tries n=47, 46, etc.
        # So SA should try to reach target_deg.
        
        if current_min_deg >= target_deg:
            return adj

        T = 1.0
        T_min = 0.01
        alpha = 0.999
        best_adj = adj.copy()
        best_min_deg = current_min_deg
        
        n_vertices = n
        all_edges = [(i, j) for i in range(n_vertices) for j in range(i + 1, n_vertices)]
        random.shuffle(all_edges)
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_limit:
            # Generate a move
            move_type = random.random()
            
            new_adj = adj.copy()
            valid_move = False
            delta_energy = 0
            
            if move_type < 0.4:
                # Try to add an edge
                # Pick random u, v not connected
                attempts = 0
                while attempts < 10:
                    u, v = random.sample(range(n_vertices), 2)
                    if not adj[u, v]:
                        # Check C4 constraint
                        common = np.sum(adj[u] & adj[v])
                        if common <= 1:
                            new_adj[u, v] = True
                            new_adj[v, u] = True
                            valid_move = True
                            # Energy change: min degree might increase
                            break
                    attempts += 1
                    
            elif move_type < 0.8:
                # Try to remove an edge
                # Pick random edge that exists
                # To avoid getting stuck at low degree, only remove if min degree is not threatened?
                # Or just remove randomly.
                existing_edges = np.argwhere(adj)
                if len(existing_edges) > 0:
                    u, v = existing_edges[random.randint(0, len(existing_edges)-1)]
                    if u > v: u, v = v, u
                    new_adj[u, v] = False
                    new_adj[v, u] = False
                    valid_move = True
            else:
                # Swap: Remove (u,v), Add (u',v')
                # This is complex to check C4-free for the new state quickly without full matrix mult.
                # Let's skip swaps for now and rely on add/remove.
                pass
            
            if not valid_move:
                continue
            
            # Calculate new energy
            # We need to check if new_adj is C4-free.
            # If it is, energy = -min_degree(new_adj)
            # If not, energy = large_penalty
            
            # Optimization: Only check C4-free if we added an edge.
            # If we removed an edge, it's still C4-free (subgraph of C4-free is C4-free).
            # So:
            if move_type < 0.4:
                # Added edge
                if is_c4_free_fast(new_adj):
                    new_min_deg = get_min_degree(new_adj)
                    new_energy = -new_min_deg
                    old_energy = -current_min_deg
                    delta_energy = new_energy - old_energy
                else:
                    # C4 formed, reject
                    continue
            else:
                # Removed edge, still C4-free
                new_min_deg = get_min_degree(new_adj)
                new_energy = -new_min_deg
                old_energy = -current_min_deg
                delta_energy = new_energy - old_energy
            
            # Metropolis criterion
            if delta_energy < 0:
                # Improvement
                adj = new_adj
                current_min_deg = -new_energy
                if current_min_deg > best_min_deg:
                    best_min_deg = current_min_deg
                    best_adj = adj.copy()
                    if best_min_deg >= target_deg:
                        break
            else:
                # Worse, accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_energy / T):
                    adj = new_adj
                    current_min_deg = -new_energy
            
            T = max(T * alpha, T_min)
            
            # Optional: If we hit target, break
            if current_min_deg >= target_deg:
                best_adj = adj.copy()
                best_min_deg = current_min_deg
                break
        
        return best_adj

    # Main loop
    best_result = None
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44]:
        if time.time() - start_time > deadline - 15:
            break
        
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        # Try multiple restarts of SA
        found = False
        for restart in range(5):
            if time.time() - start_time > deadline - 15:
                break
            
            # Allocate remaining time for this restart
            time_left = deadline - 15 - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            adj = simulated_annealing(n, target_deg, time_limit=time_left / (5 - restart))
            
            if adj is not None:
                min_deg = get_min_degree(adj)
                if min_deg >= target_deg and is_c4_free_fast(adj):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: Return a valid small graph
    if best_result is None:
        # Try a simple greedy for n=46 one last time if time permits
        if time.time() - start_time < deadline - 5:
            n = 46
            target_deg = n - 39
            adj = random_c4_free_init(n)
            if get_min_degree(adj) >= target_deg and is_c4_free_fast(adj):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                best_result = {'n': n, 'edges': edges}
        
        if best_result is None:
            # n=39, empty graph
            best_result = {'n': 39, 'edges': []}
    
    return best_result

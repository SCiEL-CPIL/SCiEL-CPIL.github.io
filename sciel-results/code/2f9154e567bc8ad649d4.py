import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def is_safe(adj, u, v):
        """Check if adding edge (u,v) creates a C4.
        Edge (u,v) is safe iff u and v have at most 1 common neighbor."""
        common = np.sum(adj[u] & adj[v])
        return common < 2

    def verify(n, edges):
        """
        Verify the graph is C4-free and has min degree >= n-39.
        Returns True if valid.
        """
        if n < 1 or n > 47:
            return False
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Check min degree
        degrees = adj.sum(axis=1)
        if np.min(degrees) < target_deg:
            return False
            
        # Check C4-free via matrix multiplication
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
            
        return True

    def simulated_annealing(n, target_deg, time_budget):
        """
        Simulated Annealing with incremental C4 counting.
        State: adjacency matrix.
        Move: Flip a random edge.
        Energy: C4_count * 1000 + degree_deficit.
        Uses incremental updates to avoid full matrix mult each step.
        """
        if time.time() - start_time > deadline - 5:
            return None
            
        # Initialize with a random graph of moderate density
        p = max(0.05, target_deg / (n - 1) * 1.5)
        
        adj = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    
        # Precompute common-neighbor counts for all pairs
        # common[u,v] = number of common neighbors of u and v
        # We maintain this incrementally.
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        common = A2.copy()  # common[u,v] = # common neighbors
        
        def compute_energy(adj, common):
            c4_count = np.sum(common >= 2)
            degrees = adj.sum(axis=1)
            deficit = int(np.sum(np.maximum(0, target_deg - degrees)))
            energy = c4_count * 1000 + deficit
            return energy, c4_count, deficit

        current_energy, cur_c4, cur_def = compute_energy(adj, common)
        best_energy = current_energy
        best_adj = adj.copy()
        
        T = 100.0
        cooling_rate = 0.995
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > deadline - 5:
                break
                
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_val = adj[i, j]
            
            # Compute delta in common-neighbor counts
            # When we flip edge (i,j), for any vertex k:
            #   if adding: common[i,k] += adj[j,k], common[j,k] += adj[i,k]
            #   if removing: common[i,k] -= adj[j,k], common[j,k] -= adj[i,k]
            # Also common[i,j] changes.
            
            delta_common = 0
            if old_val == 0:
                # Adding edge (i,j)
                # common[i,k] increases by adj[j,k] for all k != i,j
                # common[j,k] increases by adj[i,k] for all k != i,j
                # common[i,j] increases by 0 (i and j are now adjacent, but common counts neighbors)
                # Actually common[i,j] = # common neighbors of i and j = sum over k of adj[i,k]*adj[j,k]
                # This doesn't change when we add edge (i,j) itself.
                
                # For k in N(j) \ {i}: common[i,k] += 1
                # For k in N(i) \ {j}: common[j,k] += 1
                
                # Track how many pairs cross the threshold from 1 to 2 or 2 to 3 etc.
                # We need to count how many (u,v) pairs have common[u,v] >= 2 before and after.
                
                # Before: c4_count = sum(common >= 2)
                # After adding: 
                #   For k in N(j)\{i}: common[i,k] += 1. If common[i,k] was 1, it becomes 2 (new C4 pair).
                #   For k in N(i)\{j}: common[j,k] += 1. If common[j,k] was 1, it becomes 2 (new C4 pair).
                
                new_c4_delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if adj[j, k] == 1:
                        if common[i, k] == 1:
                            new_c4_delta += 1
                        common[i, k] += 1
                    if adj[i, k] == 1:
                        if common[j, k] == 1:
                            new_c4_delta += 1
                        common[j, k] += 1
                # common[i,j] doesn't change
                adj[i, j] = 1
                adj[j, i] = 1
                
                deg_i_change = 1
                deg_j_change = 1
                delta_energy = new_c4_delta * 1000 - (deg_i_change if adj[i].sum() - 1 < target_deg else 0) - (deg_j_change if adj[j].sum() - 1 < target_deg else 0)
            else:
                # Removing edge (i,j)
                new_c4_delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if adj[j, k] == 1:
                        if common[i, k] == 2:
                            new_c4_delta -= 1
                        common[i, k] -= 1
                    if adj[i, k] == 1:
                        if common[j, k] == 2:
                            new_c4_delta -= 1
                        common[j, k] -= 1
                adj[i, j] = 0
                adj[j, i] = 0
                
                deg_i_change = -1
                deg_j_change = -1
                delta_energy = new_c4_delta * 1000 + (1 if adj[i].sum() + 1 < target_deg else 0) + (1 if adj[j].sum() + 1 < target_deg else 0)
            
            current_energy += delta_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj.copy()
            
            if current_energy < 0 or random.random() < np.exp(-max(0, delta_energy) / T):
                pass  # accept
            else:
                # Revert
                if old_val == 0:
                    # We added, now remove
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[j, k] == 1:
                            common[i, k] -= 1
                        if adj[i, k] == 1:
                            common[j, k] -= 1
                    adj[i, j] = 0
                    adj[j, i] = 0
                else:
                    # We removed, now add back
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[j, k] == 1:
                            common[i, k] += 1
                        if adj[i, k] == 1:
                            common[j, k] += 1
                    adj[i, j] = 1
                    adj[j, i] = 1
                current_energy -= delta_energy
            
            T *= cooling_rate
            
        # Final check on best found
        if verify(n, [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]):
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]
            return {'n': n, 'edges': edges}
            
        return None

    def greedy_with_backtracking(n, target_deg, time_budget):
        """
        Balanced greedy: repeatedly pick the vertex with lowest degree and
        connect it to the safe vertex with the lowest degree (minimax strategy).
        """
        if time.time() - start_time > deadline - 5:
            return None
            
        for attempt in range(50):
            if time.time() - start_time > deadline - 5:
                break
                
            adj = np.zeros((n, n), dtype=np.int32)
            degrees = np.zeros(n, dtype=int)
            
            # Build common-neighbor matrix incrementally
            common = np.zeros((n, n), dtype=np.int32)
            
            # Multiple passes to keep adding edges
            for _pass in range(20):
                if time.time() - start_time > deadline - 5:
                    break
                
                added_any = False
                for _ in range(n * n):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    # Find vertex with minimum degree
                    u = int(np.argmin(degrees))
                    if degrees[u] >= target_deg:
                        break  # All good
                    
                    # Find safe neighbors of u, preferring those with lowest degree
                    best_v = -1
                    best_v_deg = n + 1
                    for v in range(n):
                        if v == u or adj[u, v] == 1:
                            continue
                        if common[u, v] < 2:  # Safe
                            if degrees[v] < best_v_deg:
                                best_v_deg = degrees[v]
                                best_v = v
                    
                    if best_v == -1:
                        break  # No safe neighbor for u
                    
                    # Add edge (u, best_v)
                    adj[u, best_v] = 1
                    adj[best_v, u] = 1
                    degrees[u] += 1
                    degrees[best_v] += 1
                    added_any = True
                    
                    # Update common-neighbor counts
                    for k in range(n):
                        if k == u or k == best_v:
                            continue
                        if adj[best_v, k] == 1:
                            common[u, k] += 1
                            common[k, u] += 1
                        if adj[u, k] == 1:
                            common[best_v, k] += 1
                            common[k, best_v] += 1
                    # common[u, best_v] stays the same (it's the count of common neighbors,
                    # and u,best_v are now adjacent but that doesn't affect their common neighbor count)
                    
                if not added_any:
                    break  # Stuck
            
            if verify(n, [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]):
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
                return {'n': n, 'edges': edges}
                
        return None

    # Main logic
    # Try n=47, then 46, etc.
    # Give greedy more time since it's fast and structured.
    
    time_alloc = {
        47: 30,
        46: 30,
        45: 15,
        44: 10,
        43: 5,
        42: 3,
        41: 2,
        40: 1,
        39: 1
    }
    
    best_result = None
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() - start_time > deadline - 2:
            break
            
        target_deg = max(0, n - 39)
        
        # Method 1: Greedy (fast, structured)
        res = greedy_with_backtracking(n, target_deg, time_alloc.get(n, 5))
        if res:
            return res
            
        # Method 2: Simulated Annealing
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            res = simulated_annealing(n, target_deg, remaining)
            if res:
                return res
            
    # Fallback: n=39, empty graph
    return {'n': 39, 'edges': []}

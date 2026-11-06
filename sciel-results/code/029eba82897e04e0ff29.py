import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible_edges = len(all_edges)
    
    def spectral_init(rng_seed):
        """Initialize a dense graph using spectral method.
        
        Create a random symmetric matrix, compute top eigenvector,
        threshold at mean to create a dense random-like starting point.
        Then greedily remove edges that violate K3,3-free condition.
        """
        rng = np.random.RandomState(rng_seed)
        
        # Generate random symmetric matrix
        M = rng.randn(n, n)
        M = (M + M.T) / 2.0
        np.fill_diagonal(M, 0)
        
        # Get top eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # Top eigenvector (last column since eigh returns in ascending order)
        v = eigenvectors[:, -1]
        
        # Threshold at mean to create dense graph
        threshold = np.mean(v)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if v[i] * v[j] > threshold * threshold:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # If too sparse, use a different threshold
        edge_count = sum(x.bit_count() for x in adj) // 2
        if edge_count < 80:
            # Try lower threshold
            threshold = np.mean(v) - np.std(v) * 0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if v[i] * v[j] > threshold * threshold:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Now remove edges that violate K3,3-free condition
        # For each edge, check if it creates a K3,3
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check if removing this edge is necessary
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    if not can_add(adj, i, j):
                        # Edge must be removed
                        pass
                    else:
                        # Keep the edge
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        return adj
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) violates K3,3-free condition.
        
        After adding (u,v), for any triple {a,b,c}, the number of common neighbors
        must be <= 2. The only new common neighbor relationships involve u or v
        being adjacent to the new edge endpoint.
        
        Specifically:
        - v becomes a common neighbor of {u,a,b} if a,b are both adjacent to v.
          We need |N(u) ∩ N(a) ∩ N(b) ∩ N(v)| <= 2 after adding, but since v is 
          already adjacent to a,b, and we're checking if v is common neighbor of {u,a,b},
          we need existing common neighbors of {u,a,b} (excluding v) to be <= 1.
        - Similarly for u as common neighbor of {v,a,b}.
        """
        if adj[u] & (1 << v):
            return False
            
        # Get neighbors of u and v (excluding each other since edge not yet added)
        neigh_u = adj[u]
        neigh_v = adj[v]
        
        # For each pair (a,b) where both are adjacent to v:
        # v will be a common neighbor of {u,a,b} after adding (u,v)
        # Need: count of vertices w adjacent to u,a,b (w != v) <= 1
        # i.e., (neigh_u & adj[a] & adj[b] & ~neigh_v).bit_count() <= 1
        # But we also need a,b to be adjacent to v, which is already checked.
        
        # Optimize: only iterate over neighbors of v
        mask_v = neigh_v
        a_list = []
        for a in range(n):
            if a != u and a != v and (mask_v & (1 << a)):
                a_list.append(a)
        
        for i in range(len(a_list)):
            a = a_list[i]
            adj_a = adj[a]
            for j in range(i + 1, len(a_list)):
                b = a_list[j]
                adj_b = adj[b]
                # Common neighbors of {u, a, b} in current graph
                common = neigh_u & adj_a & adj_b
                # After adding (u,v), v becomes a common neighbor too
                # Total = common.bit_count() + 1 (for v)
                # Need <= 2, so common.bit_count() <= 1
                if common.bit_count() > 1:
                    return False
        
        # Similarly for u as common neighbor of {v,a,b}
        mask_u = neigh_u
        a_list2 = []
        for a in range(n):
            if a != u and a != v and (mask_u & (1 << a)):
                a_list2.append(a)
        
        for i in range(len(a_list2)):
            a = a_list2[i]
            adj_a = adj[a]
            for j in range(i + 1, len(a_list2)):
                b = a_list2[j]
                adj_b = adj[b]
                # Common neighbors of {v, a, b} in current graph
                common = neigh_v & adj_a & adj_b
                if common.bit_count() > 1:
                    return False
                        
        return True

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def greedy_fill(adj, max_time=None):
        """Greedily add as many edges as possible. Returns updated adj and edge count."""
        if max_time is None:
            max_time = deadline - 2
        current_edges = sum(x.bit_count() for x in adj) // 2
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # Try multiple random orders for greedy fill
        best_adj = adj[:]
        best_edges = current_edges
        
        num_attempts = min(8, max(2, int((max_time - time.time()) / 0.5) if max_time > time.time() else 2))
        
        for attempt in range(num_attempts):
            if time.time() > max_time:
                break
            temp_adj = adj[:]
            temp_missing = missing[:]
            random.shuffle(temp_missing)
            
            for u, v in temp_missing:
                if can_add(temp_adj, u, v):
                    temp_adj[u] |= (1 << v)
                    temp_adj[v] |= (1 << u)
            
            temp_edges = sum(x.bit_count() for x in temp_adj) // 2
            if temp_edges > best_edges:
                best_edges = temp_edges
                best_adj = temp_adj[:]
                
        return best_adj, best_edges

    def local_search(adj, max_time=2.0):
        """Local search with kick mechanism."""
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # First, greedy fill
        adj, current_edges = greedy_fill(adj, max_time=max(0.1, max_time * 0.3))
        best_adj = adj[:]
        best_edges = current_edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if not missing:
            return best_adj, best_edges
        
        no_improve_count = 0
        max_no_improve = 80
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 3:
                break
                
            if not missing:
                break
                
            # Try to add a random missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove 1-4 random edges, then greedy fill
                kick_size = random.choice([1, 1, 2, 2, 3, 4])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                
                # Greedy fill from current state
                adj, current_edges = greedy_fill(adj, max_time=max(0.05, (max_time - (time.time() - ls_start)) * 0.2))
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Revert to best if we're stuck
                    if no_improve_count >= max_no_improve:
                        adj = best_adj[:]
                        current_edges = best_edges
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                        no_improve_count = 0
                        if not missing:
                            break
                        continue
                
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts with spectral initialization
    restart_count = 0
    while restart_count < 10000:
        if time.time() > deadline - 10:
            break
        restart_count += 1
        
        random.seed(restart_count * 7 + 13)
        
        # Strategy 1: Spectral initialization (50% of restarts)
        if restart_count % 2 == 0:
            adj = spectral_init(restart_count * 31 + 17)
        
        # Strategy 2: Pure greedy with random order
        elif restart_count % 4 == 0:
            adj = [0] * n
            pairs = all_edges[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Strategy 3: Start from best solution with perturbation
        elif best_edges > 0:
            adj = best_adj[:]
            existing = get_edge_list(adj)
            num_remove = random.randint(4, 12)
            random.shuffle(existing)
            for a, b in existing[:num_remove]:
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
        
        # Strategy 4: Random dense initialization
        else:
            adj = [0] * n
            # Start with a random dense graph (60-80% of edges)
            density = random.uniform(0.6, 0.8)
            for u, v in all_edges:
                if random.random() < density:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            # Remove violating edges
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        if not can_add(adj, i, j):
                            pass  # Must remove
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        edges = sum(x.bit_count() for x in adj) // 2
        
        # Greedy fill first
        remaining = deadline - time.time() - 5
        gf_time = min(2.0, max(0.2, remaining * 0.3))
        adj, edges = greedy_fill(adj, max_time=gf_time)
        
        # Local search with time budget
        remaining = deadline - time.time() - 5
        ls_time = min(3.0, max(0.3, remaining * 0.4))
        adj, edges = local_search(adj, max_time=ls_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish: try to add any remaining edges
    if time.time() < deadline - 2:
        best_adj, final_edges = greedy_fill(best_adj, max_time=deadline - 1)
        if final_edges > best_edges:
            best_edges = final_edges
            best_adj = best_adj
    
    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")

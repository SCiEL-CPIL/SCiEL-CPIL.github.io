import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    max_time = 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    def run_sa(seed):
        random.seed(seed)
        
        # State
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        # Helper to update common counts after adding edge (u, v)
        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
            
            # Update common neighbors
            # For w in N(v): common[u][w] += 1
            mask_v = adj[v]
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if w != u:
                    common[u][w] += 1
                    common[w][u] += 1
                m -= lsb
                
            # For w in N(u): common[v][w] += 1
            mask_u = adj[u]
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if w != v:
                    common[v][w] += 1
                    common[w][v] += 1
                    m -= lsb

        def remove_edge(u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
            
            # Update common neighbors (decrement)
            # For w in N(v) (current, which excludes u): common[u][w] -= 1
            mask_v = adj[v]
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[u][w] -= 1
                common[w][u] -= 1
                m -= lsb
                
            # For w in N(u) (current, which excludes v): common[v][w] -= 1
            mask_u = adj[u]
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[v][w] -= 1
                common[w][v] -= 1
                m -= lsb

        def can_add(u, v):
            # u and v must not be connected
            if (adj[u] >> v) & 1:
                return False
            
            # Check common neighbors of u and v
            if common[u][v] > 2:
                return False
            
            # Check neighbors of v
            mask_v = adj[v]
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if w != u:
                    if common[u][w] >= 2:
                        return False
                m -= lsb
                
            # Check neighbors of u
            mask_u = adj[u]
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if w != v:
                    if common[v][w] >= 2:
                        return False
                m -= lsb
                
            return True

        # Initialize with greedy construction
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if can_add(u, v):
                add_edge(u, v)
        
        current_count = len(edges)
        local_best_count = current_count
        local_best_edges = list(edges)
        
        # Simulated Annealing Parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        steps_per_temp = 1000
        
        while T > min_T:
            for _ in range(steps_per_temp):
                if time.time() - start_time > max_time:
                    return local_best_edges
                    
                move_type = random.random()
                
                if move_type < 0.3:
                    # Try Add
                    # Find a valid non-edge
                    # To speed up, we can pick random non-edges
                    # Since graph is dense, most are edges.
                    # Let's pick random u, v
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if (adj[u] >> v) & 1:
                        continue # Already an edge
                    
                    if can_add(u, v):
                        add_edge(u, v)
                        current_count += 1
                        if current_count > local_best_count:
                            local_best_count = current_count
                            local_best_edges = list(edges)
                    else:
                        # Rejection in SA?
                        # For hard constraints, we just don't do the move.
                        # Or we can treat it as a penalty?
                        # No, it must be valid. So just skip.
                        pass
                elif move_type < 0.6:
                    # Try Remove
                    if not edges:
                        continue
                    u, v = random.choice(edges)
                    remove_edge(u, v)
                    current_count -= 1
                else:
                    # Try Swap
                    # Remove a random edge, then try to add a random non-edge
                    if not edges:
                        continue
                    u_out, v_out = random.choice(edges)
                    remove_edge(u_out, v_out)
                    
                    # Try to add a random non-edge
                    swapped = False
                    attempts = 0
                    while attempts < 10:
                        u_in = random.randint(0, n-1)
                        v_in = random.randint(0, n-1)
                        if u_in == v_in:
                            attempts += 1
                            continue
                        if u_in > v_in: u_in, v_in = v_in, u_in
                        if (adj[u_in] >> v_in) & 1:
                            attempts += 1
                            continue
                            
                        if can_add(u_in, v_in):
                            add_edge(u_in, v_in)
                            swapped = True
                            break
                        attempts += 1
                    
                    if not swapped:
                        # Revert the removal
                        add_edge(u_out, v_out)
                    else:
                        # Count stays same, but structure changed
                        # Check if this led to a better "potential"?
                        # For now, just accept if valid.
                        pass
            
            T *= cooling

        return local_best_edges

    # Run multiple restarts
    seed = 42
    while time.time() - start_time < max_time:
        edges = run_sa(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")

import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for evaluation overhead

    def fast_check_all_subsets(n, adj_np):
        """
        Check if graph has no regular induced subgraph of size >= 6.
        Uses bitset-based approach for speed.
        """
        # Precompute adjacency as bitmasks for fast subset operations
        adj_mask = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj_np[i, j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
        
        def subset_degrees(indices):
            """Compute degrees within a subset using bitmask operations."""
            k = len(indices)
            if k < 2:
                return [0] * k
            idx_set = set(indices)
            degrees = []
            for idx in indices:
                # Count neighbors of idx within the subset
                neighbors_in_subset = adj_mask[idx] & 0  # start with 0
                for other in indices:
                    if other != idx and (adj_mask[idx] >> other) & 1:
                        neighbors_in_subset += 1
                degrees.append(neighbors_in_subset)
            return degrees
        
        # Check all subsets of size 6 to n
        # For efficiency, only check size 6 (if no regular 6-set, larger sets 
        # might still be regular, but we need to check all)
        # Actually, we must check all sizes >= 6
        
        # Precompute all 6-combinations as a list for speed
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                # Fast regularity check
                deg = 0
                first_deg = -1
                for i in range(k):
                    v = comb[i]
                    d = 0
                    for j in range(k):
                        if i != j and adj_np[v, comb[j]]:
                            d += 1
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        break
                else:
                    # All degrees equal - regular subgraph found
                    return False
        return True

    def count_violations_fast(n, adj_np):
        """
        Count number of 6-vertex subsets that induce regular subgraphs.
        This is the key metric for local search.
        """
        count = 0
        for comb in itertools.combinations(range(n), 6):
            # Check if regular
            first_deg = -1
            regular = True
            for i in range(6):
                v = comb[i]
                d = 0
                for j in range(6):
                    if i != j and adj_np[v, comb[j]]:
                        d += 1
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    regular = False
                    break
            if regular:
                count += 1
        return count

    def get_violating_edges(n, adj_np):
        """Return set of edges that appear in at least one violating 6-set."""
        violating_edges = set()
        for comb in itertools.combinations(range(n), 6):
            first_deg = -1
            regular = True
            for i in range(6):
                v = comb[i]
                d = 0
                for j in range(6):
                    if i != j and adj_np[v, comb[j]]:
                        d += 1
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    regular = False
                    break
            if regular:
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = comb[i], comb[j]
                        violating_edges.add((min(u, v), max(u, v)))
        return violating_edges

    def local_search(n, max_iters=2000, seed=None):
        """
        Local search with simulated annealing-like acceptance.
        Uses violation count as the objective to minimize.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Initialize random graph
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        current_viol = count_violations_fast(n, adj)
        if current_viol == 0:
            if fast_check_all_subsets(n, adj):
                return adj
        
        best_adj = adj.copy()
        best_viol = current_viol
        
        # Temperature for simulated annealing
        temp = 1.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose edge to flip - bias toward violating edges
            if current_viol > 0 and it % 20 == 0:
                viol_edges = get_violating_edges(n, adj)
                if viol_edges and rng.random() < 0.7:
                    u, v = rng.choice(list(viol_edges))
                else:
                    u, v = rng.choice(edges)
            else:
                u, v = rng.choice(edges)
            
            # Flip
            adj[u, v] ^= 1
            adj[v, u] ^= 1
            
            new_viol = count_violations_fast(n, adj)
            
            # Acceptance criterion
            if new_viol <= current_viol:
                current_viol = new_viol
                if current_viol == 0:
                    if fast_check_all_subsets(n, adj):
                        return adj
            else:
                # Simulated annealing acceptance
                delta = new_viol - current_viol
                accept_prob = np.exp(-delta / temp)
                if rng.random() < accept_prob:
                    current_viol = new_viol
                else:
                    # Revert
                    adj[u, v] ^= 1
                    adj[v, u] ^= 1
            
            if current_viol < best_viol:
                best_viol = current_viol
                best_adj = adj.copy()
            
            temp *= cooling
        
        if best_viol == 0:
            if fast_check_all_subsets(n, best_adj):
                return best_adj
        
        return None

    def try_find(target_n, time_budget, num_restarts=5):
        """Try to find a valid graph on target_n vertices."""
        t_start = time.time()
        for r in range(num_restarts):
            if time.time() > t_start + time_budget:
                break
            adj = local_search(target_n, max_iters=1500, seed=r * 1000 + 42)
            if adj is not None:
                return {'n': target_n, 'edges': [[u, v] for u in range(target_n) 
                        for v in range(u + 1, target_n) if adj[u, v] == 1]}
        return None

    # Strategy: Try n=20 first (known achievable), then n=21 if time permits
    # Focus most time on n=20 to ensure we get a good result
    
    # First, try n=20 with substantial time
    result = try_find(20, 40, num_restarts=4)
    if result is not None:
        # Check if we have time to try n=21
        if time.time() < start_time + 95:
            result21 = try_find(21, 35, num_restarts=2)
            if result21 is not None:
                return result21
        return result

    # Fallback to n=18
    if time.time() < start_time + 100:
        result = try_find(18, 20, num_restarts=3)
        if result is not None:
            return result

    # Fallback to n=15
    if time.time() < start_time + 115:
        result = try_find(15, 15, num_restarts=5)
        if result is not None:
            return result

    # Very small fallback: construct a known small valid graph
    # Use a small graph that is likely valid
    n = 12
    adj = np.zeros((n, n), dtype=np.uint8)
    # Use a cyclic pattern with some randomness
    for i in range(n):
        for j in range(i + 1, n):
            if ((i - j) % 5) in (1, 2) or ((i + j) % 7) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Verify and fix
    comb_indices = list(itertools.combinations(range(n), 6))
    for _ in range(100):
        viol = count_violations_fast(n, adj)
        if viol == 0:
            if fast_check_all_subsets(n, adj):
                return {'n': n, 'edges': [[u, v] for u in range(n) 
                        for v in range(u + 1, n) if adj[u, v] == 1]}
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            adj[u, v] ^= 1
            adj[v, u] ^= 1

    # Ultimate fallback: empty graph on 6 vertices (K6 is not regular-free, 
    # but we need a valid one)
    # Actually, let's use a small valid construction
    n = 7
    adj = np.zeros((n, n), dtype=np.uint8)
    # Random small graph and verify
    for _ in range(1000):
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i, j] = 1
                    adj[j, i] = 1
        if fast_check_all_subsets(n, adj):
            return {'n': n, 'edges': [[u, v] for u in range(n) 
                    for v in range(u + 1, n) if adj[u, v] == 1]}
    
    # Return minimal valid: single vertex
    return {'n': 1, 'edges': []}


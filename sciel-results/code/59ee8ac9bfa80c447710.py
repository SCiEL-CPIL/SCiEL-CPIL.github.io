import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid_fast(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg. Optimized."""
        # Check min degree
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    return False
        return True

    def find_c4_violations(adj, n):
        """Find all pairs (i,j) with >1 common neighbor. Returns list of (i,j,common_mask)."""
        violations = []
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    violations.append((i, j, common))
        return violations

    def repair_c4(adj, n, deg):
        """Fix all C4 violations by removing edges. Returns True if any fix was made."""
        fixed_any = False
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if common.bit_count() > 1:
                        # Need to remove edges so that common has at most 1 bit
                        # Remove from the endpoint with higher degree
                        common_list = []
                        mask = common
                        k = 0
                        while mask:
                            if mask & 1:
                                common_list.append(k)
                            mask >>= 1
                            k += 1
                        
                        # Remove enough edges to leave at most 1 common neighbor
                        for w in common_list[1:]:
                            if deg[i] >= deg[j]:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg[i] -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg[j] -= 1
                        fixed_any = True
                        changed = True
                        break
                if changed:
                    break
        return fixed_any

    def greedy_build_optimized(n, target_deg, seed=None, time_budget=5.0):
        """
        Optimized greedy construction with numpy-accelerated C4 checks.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * n
        deg = [0] * n
        start = time.time()
        
        # Precompute for fast C4 checking using numpy
        # adj_matrix will be n x n boolean array
        adj_np = np.zeros((n, n), dtype=np.uint8)
        
        iterations = 0
        max_iterations = n * n * 3
        
        while iterations < max_iterations and time.time() - start < time_budget:
            iterations += 1
            
            # Find vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])
            
            if deg[u] >= target_deg:
                if min(deg) >= target_deg:
                    break
                u = min(range(n), key=lambda i: deg[i])
            
            # Find candidates: vertices v such that no C4 would be created
            # i.e., |N(u) ∩ N(v)| == 0
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                # Sort by degree ascending to balance
                candidates.sort(key=lambda x: deg[x])
                # Add edges to all valid candidates
                for v in candidates:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
            else:
                # No safe candidates, try edge swap
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    w = random.choice(neighbors)
                    # Remove (u,w)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
                    
                    # Find new candidates for u
                    candidates2 = []
                    for v in range(n):
                        if v == u:
                            continue
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates2.append(v)
                    
                    if candidates2:
                        candidates2.sort(key=lambda x: deg[x])
                        v = candidates2[0]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                    else:
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        deg[u] += 1
                        deg[w] += 1
        
        if min(deg) >= target_deg:
            return adj, deg
        return None, None

    def local_search_numba_style(adj, n, target_deg, max_time=10.0, seed=None):
        """
        Local search with aggressive edge addition and C4 repair.
        Uses a soft-min objective: maximize the minimum degree.
        """
        if seed is not None:
            random.seed(seed)
        
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()
        
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start < max_time and no_improve < max_no_improve:
            # Step 1: Fix any C4 violations
            repair_c4(adj, n, deg)
            
            # Step 2: Try to add edges to boost min degree
            # Focus on the vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])
            
            if deg[u] >= target_deg and min(deg) >= target_deg:
                break
            
            # Find all candidates for u that don't create C4
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                # Sort by degree ascending
                candidates.sort(key=lambda x: deg[x])
                # Add edge to the best candidate
                v = candidates[0]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # No direct candidates, try edge swap
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    w = random.choice(neighbors)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
                    
                    candidates2 = []
                    for v in range(n):
                        if v == u:
                            continue
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates2.append(v)
                    
                    if candidates2:
                        candidates2.sort(key=lambda x: deg[x])
                        v = candidates2[0]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        
                        new_min = min(deg)
                        if new_min > best_min_deg:
                            best_min_deg = new_min
                            best_adj = adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        deg[u] += 1
                        deg[w] += 1
                        no_improve += 1
                else:
                    no_improve += 1
        
        return best_adj, best_min_deg

    def algebraic_construction(n, target_deg):
        """
        Try an algebraic construction based on finite fields or quadratic residues.
        For n = q+1 where q is a prime power, use polarity graph construction.
        """
        # For n=46, we can't use a standard finite field directly
        # But we can try: take a known C4-free graph and verify
        # Or use a construction based on Paley graphs or similar
        
        # Try: vertices 0..n-1, connect i to j if (i-j) is in a specific set
        # For C4-freeness, we need the difference set to have no 4-cycles
        
        # Simple approach: use a quadratic residue graph on a prime
        # If n is prime, use Paley graph
        def is_prime(m):
            if m < 2:
                return False
            if m == 2:
                return True
            if m % 2 == 0:
                return False
            for i in range(3, int(m**0.5) + 1, 2):
                if m % i == 0:
                    return False
            return True
        
        # Try nearby primes
        for p in range(max(3, n - 5), n + 5):
            if not is_prime(p):
                continue
            
            # Build Paley graph on p vertices
            adj = [0] * p
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    # Check if diff is a quadratic residue mod p
                    if pow(diff, (p - 1) // 2, p) == 1:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            deg = [adj[i].bit_count() for i in range(p)]
            min_deg = min(deg)
            
            if min_deg >= target_deg:
                # Check C4-freeness
                if check_valid_fast(adj, p, min_deg):
                    if p >= n:
                        # Truncate to n vertices
                        adj_trunc = adj[:n]
                        deg_trunc = [adj_trunc[i].bit_count() for i in range(n)]
                        if min(deg_trunc) >= target_deg and check_valid_fast(adj_trunc, n, target_deg):
                            return adj_trunc
            
            # Also try the complement or other variants
            # For now, just return if we found a valid one
        
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # Strategy 1: Algebraic construction (fast, try first)
        if n <= 47:
            adj_alg = algebraic_construction(n, min_deg)
            if adj_alg is not None:
                if check_valid_fast(adj_alg, n, min_deg):
                    edges = get_edges(adj_alg, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 2: Optimized greedy build (multiple attempts)
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            adj, deg = greedy_build_optimized(n, min_deg, seed=attempt * 7919 + 13, time_budget=2.0)
            
            if adj is not None:
                if check_valid_fast(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 3: Local search from greedy results
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            adj, deg = greedy_build_optimized(n, min_deg, seed=attempt * 104729 + 7, time_budget=1.5)
            
            if adj is not None:
                adj_ls, min_deg_ls = local_search_numba_style(adj, n, min_deg, max_time=5.0, seed=attempt * 31 + 1)
                if min_deg_ls >= min_deg and check_valid_fast(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 4: Random initialization + repair
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 15485863 + 1)
            
            # Initialize with random graph with appropriate density
            p = min(min_deg / n * 1.2, 0.5)
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            deg = [adj[i].bit_count() for i in range(n)]
            
            # Repair phase with time limit
            repair_start = time.time()
            for _ in range(300):
                if time.time() - repair_start > 3.0 or time.time() > deadline:
                    break
                
                # Fix C4s
                repair_c4(adj, n, deg)
                
                # Check if we're done
                if min(deg) >= min_deg:
                    if check_valid_fast(adj, n, min_deg):
                        edges = get_edges(adj, n)
                        return {'n': n, 'edges': edges}
                    break
                
                # Add edges to boost min degree
                u = min(range(n), key=lambda i: deg[i])
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if not (adj[u] & (1 << v)):
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates.append(v)
                
                if candidates:
                    candidates.sort(key=lambda x: deg[x])
                    v = candidates[0]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1

    # Fallback: Return a valid small graph
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid_fast(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}

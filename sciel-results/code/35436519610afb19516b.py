import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Strategy:
    # 1. Use a fast, bitmask-based randomized search for n=40 down to 27.
    # 2. If a valid graph is found, return it immediately.
    # 3. If not, fall back to a known valid graph (And(9) on n=26).
    # 4. The key improvement over the current program is the use of bitmasks
    #    for both graph representation and clique detection, which is significantly
    #    faster than numpy arrays for these operations.
    
    # Fallback: And(9) on n=26, alpha=9 (valid)
    # This is guaranteed to be valid and fast to construct.
    def _and9():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # We'll try to find a better graph first.
    # If the search takes too long, we return the fallback.
    
    # Bitmask-based random maximal triangle-free graph generation
    def _random_maximal_tf(n, deadline):
        adj_mask = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                return None
            if adj_mask[u] & adj_mask[v]:
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Fast clique detection in complement graph using bitmasks
    # Returns True if complement has a clique of size k (i.e., original has independent set of size k)
    def _has_clique_size_k(comp_mask, n, k, deadline):
        # Quick prune: if max degree < k-1, no clique
        max_deg = max(bin(comp_mask[i]).count('1') for i in range(n))
        if max_deg < k - 1:
            return False
        
        # Order vertices by degree in comp (descending)
        deg_order = sorted(range(n), key=lambda i: bin(comp_mask[i]).count('1'), reverse=True)
        
        # Rebuild comp_mask in new order for better pruning
        comp_mask_new = [0] * n
        for i in range(n):
            v = deg_order[i]
            mask = 0
            for j in range(n):
                if j != i and (comp_mask[v] >> deg_order[j]) & 1:
                    mask |= (1 << j)
            comp_mask_new[i] = mask
        
        def bnb(candidates, need):
            if time.time() > deadline:
                return True  # Conservative: assume invalid on timeout
            if need == 0:
                return True
            if candidates == 0:
                return False
            if bin(candidates).count('1') < need:
                return False
            
            # Pick vertex with highest degree within candidates
            # For simplicity, iterate through set bits
            c = candidates
            while c:
                lb = c & (-c)
                idx = lb.bit_length() - 1
                c ^= lb
                
                # Remaining candidates: bits above idx
                remaining = candidates & (~((1 << (idx + 1)) - 1))
                new_candidates = remaining & comp_mask_new[idx]
                
                if bnb(new_candidates, need - 1):
                    return True
            return False
        
        all_candidates = (1 << n) - 1
        return bnb(all_candidates, k)
    
    def _alpha_less_than_10(adj_mask, n, deadline):
        all_mask = (1 << n) - 1
        comp_mask = [all_mask ^ (1 << i) ^ adj_mask[i] for i in range(n)]
        return not _has_clique_size_k(comp_mask, n, 10, deadline)
    
    def _mask_to_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break
        
        # More attempts for smaller n (easier to find)
        max_attempts = 300 if n <= 30 else 150
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            adj_mask = _random_maximal_tf(n, deadline - 10)
            if adj_mask is None:
                break
            
            if _alpha_less_than_10(adj_mask, n, deadline - 5):
                return {'n': n, 'edges': _mask_to_edges(adj_mask, n)}
    
    # Fallback
    return _and9()

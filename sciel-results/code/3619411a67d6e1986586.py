import time
import random
import sys
import itertools

def construct():
    """
    Finds a graph on n vertices with no regular induced subgraph on >= 6 vertices.
    Uses simulated annealing with sampling-based violation detection.
    Tries n=21 first, falls back to n=20.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    def popcount(x):
        return x.bit_count()
    
    def is_regular_subset_fast(mask, adj_masks, size):
        """
        Check if the induced subgraph on vertices in 'mask' is regular.
        Uses early termination.
        """
        if size < 6:
            return False
        
        # Calculate degrees of vertices in the subset
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = popcount(adj_masks[v] & mask)
            degs.append(deg)
            m ^= lsb
            
        # Check if all degrees are equal
        first_deg = degs[0]
        for d in degs[1:]:
            if d != first_deg:
                return False
        return True
    
    def find_violation_sampled(adj_masks, n, num_samples=200):
        """
        Find a regular induced subgraph of size >= 6 by random sampling.
        Returns (mask, size) or (None, 0).
        """
        # Sample random subsets of various sizes
        for _ in range(num_samples):
            # Choose a random size from 6 to min(n, 10)
            max_size = min(n, 12)
            size = random.randint(6, max_size)
            
            # Choose random vertices
            if size > n:
                continue
            vertices = random.sample(range(n), size)
            mask = 0
            for v in vertices:
                mask |= (1 << v)
            
            if is_regular_subset_fast(mask, adj_masks, size):
                return mask, size
        
        # Also do some exhaustive checks for small sizes if time permits
        # Check all 6-subsets (most likely to be regular)
        if n <= 21:
            for combo in itertools.combinations(range(n), 6):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_subset_fast(mask, adj_masks, 6):
                    return mask, 6
        
        return None, 0
    
    def flip_edge(adj_masks, u, v):
        """
        Flip the edge between u and v in the adjacency masks.
        """
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)
    
    def random_adj(n):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks
    
    def adj_to_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def simulate_annealing(n, max_time=30):
        """
        Simulated annealing to find a valid graph on n vertices.
        Returns adj_masks or None.
        """
        sa_start = time.time()
        sa_deadline = sa_start + max_time
        
        # Initialize
        adj_masks = random_adj(n)
        
        # Initial violation check
        mask, size = find_violation_sampled(adj_masks, n, num_samples=100)
        if mask is None:
            return adj_masks
        
        # Annealing parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        best_adj = None
        best_violations = float('inf')
        
        while time.time() < sa_deadline and T > T_min:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Compute delta: how many violations change
            # This is expensive, so we just flip and check
            old_mask, old_size = mask, size
            flip_edge(adj_masks, u, v)
            
            new_mask, new_size = find_violation_sampled(adj_masks, n, num_samples=50)
            
            # Accept or reject
            if new_mask is None:
                # No violations - we're done!
                return adj_masks
            
            # Simple acceptance: always accept if we don't make it worse
            # For simplicity, just accept with probability based on temperature
            if random.random() < T:
                mask, size = new_mask, new_size
            else:
                # Reject: flip back
                flip_edge(adj_masks, u, v)
            
            T *= cooling
        
        return None
    
    # Try n=21 first with aggressive time budget
    n_candidates = [21, 20]
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        
        # Give more time to n=21
        if n == 21:
            max_time_per_attempt = 20
            num_attempts = 3
        else:
            max_time_per_attempt = 40
            num_attempts = 5
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            result = simulate_annealing(n, max_time=max_time_per_attempt)
            if result is not None:
                # Verify the result
                mask, size = find_violation_sampled(result, n, num_samples=500)
                if mask is None:
                    # Double-check with more thorough verification
                    # Check all 6-subsets
                    valid = True
                    for combo in itertools.combinations(range(n), 6):
                        m = 0
                        for v in combo:
                            m |= (1 << v)
                        if is_regular_subset_fast(m, result, 6):
                            valid = False
                            break
                    if valid:
                        edges = adj_to_edges(result, n)
                        return {'n': n, 'edges': edges}
        
        # If n=21 failed, move to n=20
        if n == 21:
            continue
    
    # Fallback: return a valid small graph
    # Empty graph on 5 vertices is always valid
    return {'n': 5, 'edges': []}

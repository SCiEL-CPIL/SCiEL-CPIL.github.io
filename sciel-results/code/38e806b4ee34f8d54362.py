import itertools
import random
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 145

    # Known valid 20-vertex graph
    known_20_edges = [
        [0, 1], [0, 2], [0, 5], [0, 6], [0, 9], [0, 10], [0, 13], [0, 14],
        [1, 3], [1, 4], [1, 7], [1, 8], [1, 11], [1, 12], [1, 15], [1, 16],
        [2, 3], [2, 6], [2, 8], [2, 10], [2, 13], [2, 16], [2, 18],
        [3, 5], [3, 9], [3, 12], [3, 14], [3, 17],
        [4, 6], [4, 7], [4, 10], [4, 15], [4, 18],
        [5, 7], [5, 11], [5, 14], [5, 17],
        [6, 8], [6, 12], [6, 15],
        [7, 9], [7, 13], [7, 16],
        [8, 11], [8, 14], [8, 19],
        [9, 12], [9, 15], [9, 18],
        [10, 13], [10, 16], [10, 19],
        [11, 14], [11, 17],
        [12, 15], [12, 18],
        [13, 16], [13, 19],
        [14, 17],
        [15, 18],
        [16, 19],
    ]

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_regular_subset(adj_masks, subset, n):
        """Check if the induced subgraph on 'subset' is regular."""
        mask = 0
        for v in subset:
            mask |= (1 << v)
        d0 = (adj_masks[subset[0]] & mask).bit_count()
        for v in subset[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                return False
        return True

    def find_violation_fast(n, adj_masks, max_k=None):
        """Find a violating subset. Returns (k, subset) or (0, None)."""
        if max_k is None:
            max_k = n
        for k in range(6, max_k + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular_subset(adj_masks, subset, n):
                    return (k, subset)
        return (0, None)

    def count_violations(n, adj_masks, sample_size=5000):
        """Estimate number of violations by sampling."""
        count = 0
        # Check some random subsets
        for _ in range(sample_size):
            k = random.randint(6, n)
            subset = random.sample(range(n), k)
            if is_regular_subset(adj_masks, subset, n):
                count += 1
        return count

    def simulated_annealing(n, adj_masks, time_budget, seed=None):
        """
        Simulated annealing to minimize the number of regular induced subgraphs.
        Energy = number of regular induced subsets of size >= 6 (estimated).
        """
        if seed is not None:
            random.seed(seed)
        
        deadline = time.time() + time_budget
        current = list(adj_masks)
        current_energy = count_violations(n, current, 2000)
        best = list(current)
        best_energy = current_energy
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            if (current[i] >> j) & 1:
                current[i] &= ~(1 << j)
                current[j] &= ~(1 << i)
            else:
                current[i] |= (1 << j)
                current[j] |= (1 << i)
            
            new_energy = count_violations(n, current, 2000)
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                # Revert
                if (current[i] >> j) & 1:
                    current[i] &= ~(1 << j)
                    current[j] &= ~(1 << i)
                else:
                    current[i] |= (1 << j)
                    current[j] |= (1 << i)
            
            T *= cooling
        
        # Check if best is valid
        if best_energy == 0:
            k, subset = find_violation_fast(n, best)
            if k == 0:
                return best
        return None

    def targeted_repair(n, adj_masks, time_budget):
        """
        Find violations and directly repair them by flipping edges within the violating subset.
        """
        deadline = time.time() + time_budget
        current = list(adj_masks)
        
        max_iters = 100000
        iters = 0
        
        while time.time() < deadline and iters < max_iters:
            iters += 1
            k, subset = find_violation_fast(n, current, max_k=min(8, n))
            if k == 0:
                # No violation found in sizes 6-8, check all
                k, subset = find_violation_fast(n, current)
                if k == 0:
                    return current
                # Found a larger violation, fall through to repair
            
            # Flip a random edge within the violating subset
            u, v = random.sample(subset, 2)
            if (current[u] >> v) & 1:
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
            else:
                current[u] |= (1 << v)
                current[v] |= (1 << u)
        
        return None

    # Step 1: Verify n=20
    adj_20 = build_adj_masks(20, known_20_edges)
    if find_violation_fast(20, adj_20)[0] != 0:
        return {'n': 5, 'edges': []}

    # Step 2: Try to find n=21 with multiple strategies
    n = 21
    
    # Strategy 1: Extend the known 20-vertex graph with vertex 20
    # Try different connection patterns for vertex 20
    best_result = None
    
    # Try structured extensions: vertex 20 connected to vertices with similar degrees
    degrees_20 = [(adj_20[i] & ((1 << 20) - 1)).bit_count() for i in range(20)]
    
    # Try several random masks for vertex 20
    for attempt in range(30):
        if time.time() > time_limit - 15:
            break
        
        adj_21 = list(adj_20) + [0]
        
        # Random mask for vertex 20
        mask_20 = random.getrandbits(20)
        adj_21[20] = mask_20
        for i in range(20):
            if (mask_20 >> i) & 1:
                adj_21[i] |= (1 << 20)
        
        # Try targeted repair
        result = targeted_repair(21, adj_21, time_budget=2.0)
        if result is not None:
            k, _ = find_violation_fast(21, result)
            if k == 0:
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}
    
    # Strategy 2: Random graphs with simulated annealing
    for attempt in range(10):
        if time.time() > time_limit - 10:
            break
        
        adj_21 = [0] * 21
        # Random graph with p=0.5
        for i in range(21):
            for j in range(i + 1, 21):
                if random.random() < 0.5:
                    adj_21[i] |= (1 << j)
                    adj_21[j] |= (1 << i)
        
        result = simulated_annealing(21, adj_21, time_budget=5.0)
        if result is not None:
            k, _ = find_violation_fast(21, result)
            if k == 0:
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}
    
    # Strategy 3: Try complement of random graphs
    for attempt in range(5):
        if time.time() > time_limit - 5:
            break
        
        adj_21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if random.random() < 0.5:
                    adj_21[i] |= (1 << j)
                    adj_21[j] |= (1 << i)
        
        # Complement
        for i in range(21):
            for j in range(i + 1, 21):
                if not ((adj_21[i] >> j) & 1):
                    adj_21[i] |= (1 << j)
                    adj_21[j] |= (1 << i)
        
        result = simulated_annealing(21, adj_21, time_budget=5.0)
        if result is not None:
            k, _ = find_violation_fast(21, result)
            if k == 0:
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}
    
    # Step 3: Return n=20
    return {'n': 20, 'edges': known_20_edges}

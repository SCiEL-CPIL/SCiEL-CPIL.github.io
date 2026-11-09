import math
import time
import random

def construct():
    N = 512
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute LCMs and conflict structure
    # We only need to know if lcm(a, b) > N
    # conflicts[a] is a set of b such that lcm(a, b) > N
    conflicts = [set() for _ in range(N + 1)]
    
    # Optimization: Only compute for a <= b
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if a * b // math.gcd(a, b) > N:
                conflicts[a].add(b)
                if a != b:
                    conflicts[b].add(a)
    
    # Convert to lists for faster iteration if needed, but sets are good for removal
    # For greedy, we maintain a set of available nodes
    # Initial available nodes: 1..N
    
    best_set = set(range(1, 23)) # Fallback to seed
    best_size = len(best_set)
    
    # Greedy MIS heuristic with randomization
    # Strategy: Pick node with min degree in current induced subgraph
    # To speed up, we can use a priority queue or just scan if N is small (512 is small)
    
    def greedy_mis(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        available = set(range(1, N + 1))
        solution = set()
        
        while available:
            # Find node with minimum number of conflicts within 'available'
            # degree(u) = len(conflicts[u] & available)
            # Since N=512, scanning all available nodes is O(N^2) per step, total O(N^3) = 512^3 ~ 134M ops.
            # This might be too slow for many restarts in 55s.
            # Let's optimize:
            # We can maintain degrees dynamically.
            
            # Simpler approach for N=512:
            # Just pick the node with min conflicts.
            # If we do 512 steps, and each step scans 512 nodes, and each node scan is O(1) if we have bitmasks or precomputed intersection sizes?
            # Python set intersection is fast in C.
            
            min_deg = float('inf')
            candidates = []
            
            # Optimization: We can just pick one. To add randomness, pick from top 5 min-degree nodes.
            # But calculating degree for all nodes is expensive.
            
            # Alternative: Use a list of available nodes and pick randomly, then check if it's "good"?
            # No, Min-Degree is a strong heuristic.
            
            # Let's try a faster heuristic:
            # Pick a random node from available, but bias towards those with fewer conflicts.
            # Or, just calculate degrees for a subset?
            
            # Given N=512, let's just calculate degrees for all available nodes.
            # 512 * 512 operations per restart?
            # If we do 1000 restarts, 512M ops. Might be tight in Python.
            
            # Let's try to be smarter.
            # We can precompute a bitmask for conflicts?
            # 512 bits fits in 8 64-bit ints.
            # Python ints are arbitrary precision, so we can use a single int as a bitmask!
            
            # Re-encode conflicts as bitmasks
            conflict_masks = [0] * (N + 1)
            for a in range(1, N + 1):
                mask = 0
                for b in conflicts[a]:
                    mask |= (1 << b)
                conflict_masks[a] = mask
            
            # Reset available to bitmask
            avail_mask = (1 << (N + 1)) - 2 # Bits 1..N set
            # Bit i represents node i
            
            solution = set()
            
            while avail_mask:
                # Find node with min degree in available subgraph
                # degree(u) = popcount(conflict_masks[u] & avail_mask)
                
                # To avoid O(N) scan every time, we can just scan.
                # N=512 is small.
                min_deg = float('inf')
                best_node = -1
                # Collect top few to allow randomization
                top_nodes = []
                
                # Iterate through set bits in avail_mask?
                # Or just 1..N?
                # 1..N is 512 iterations.
                for u in range(1, N + 1):
                    if (avail_mask >> u) & 1:
                        deg = bin(conflict_masks[u] & avail_mask).count("1")
                        if deg < min_deg:
                            min_deg = deg
                            best_node = u
                            top_nodes = [u]
                        elif deg == min_deg:
                            top_nodes.append(u)
                
                if not top_nodes:
                    break
                    
                # Pick randomly from top_nodes to diversify
                chosen = random.choice(top_nodes)
                
                solution.add(chosen)
                
                # Remove chosen and its neighbors from avail_mask
                # Neighbors are conflict_masks[chosen]
                # We want to remove chosen AND its neighbors.
                # So we keep only nodes NOT in (neighbors | {chosen})
                # mask_to_remove = conflict_masks[chosen] | (1 << chosen)
                avail_mask &= ~(conflict_masks[chosen] | (1 << chosen))
                
            return solution

    # Run multiple restarts
    start_time = time.time()
    restart_count = 0
    
    # Initial deterministic run
    best_set = greedy_mis(0)
    best_size = len(best_set)
    
    while time.time() < deadline:
        restart_count += 1
        # Use different seeds
        s = greedy_mis(restart_count)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # If we found a very large set, we might be close to optimal.
            # But keep searching.
            
        # If we hit a time limit inside greedy, break.
        if time.time() > deadline:
            break
            
    return list(best_set)

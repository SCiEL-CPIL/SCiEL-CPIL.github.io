import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s safety margin for 60s limit
    
    N = 256
    # Nodes are 2..256
    # Map value to index: idx = value - 2, range 0..254
    num_nodes = N - 1
    
    # Precompute conflicts
    # conflict[i][j] is True if lcm(i+2, j+2) <= 256
    # We use a list of lists of indices that conflict with i
    conflicts = [[] for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        val_i = i + 2
        for j in range(i + 1, num_nodes):
            val_j = j + 2
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(val_i, val_j)
            lcm_val = (val_i // g) * val_j
            if lcm_val <= 256:
                conflicts[i].append(j)
                conflicts[j].append(i)
    
    # Helper to check if a node can be added to a set represented by a bitmask or boolean array
    # For speed, we'll use a boolean array `in_set`
    
    def get_score(in_set):
        s = 0.0
        for i in range(num_nodes):
            if in_set[i]:
                s += 1.0 / (i + 2)
        return s

    def can_add(in_set, node_idx):
        if in_set[node_idx]:
            return False
        for neighbor in conflicts[node_idx]:
            if in_set[neighbor]:
                return False
        return True

    def try_add_all(in_set):
        changed = True
        while changed:
            changed = False
            for i in range(num_nodes):
                if not in_set[i] and can_add(in_set, i):
                    in_set[i] = True
                    changed = True
                    # Check time
                    if time.time() > deadline:
                        return False
        return True

    def hill_climb(in_set):
        improved = True
        while improved:
            improved = False
            current_score = get_score(in_set)
            
            # Try single swaps: remove u, add v
            # To be efficient, iterate u in set, then v not in set
            # But checking all v for each u is O(N^2). N=255, so ~65k checks.
            # Each check is O(degree). Average degree might be moderate.
            # 65k * 10 ops = 650k ops per pass. Very fast.
            
            for u in range(num_nodes):
                if not in_set[u]:
                    continue
                for v in range(num_nodes):
                    if in_set[v]:
                        continue
                    # Check if v can be added if u is removed
                    # v must not conflict with any w in set \ {u}
                    valid = True
                    for neighbor in conflicts[v]:
                        if in_set[neighbor] and neighbor != u:
                            valid = False
                            break
                    if valid:
                        # Check if swap improves score
                        if (1.0 / (v + 2)) > (1.0 / (u + 2)):
                            in_set[u] = False
                            in_set[v] = True
                            improved = True
                            if time.time() > deadline:
                                return
            # After swaps, try to add any remaining nodes
            try_add_all(in_set)
            if time.time() > deadline:
                return

    best_score = 0.0
    best_set = [False] * num_nodes
    
    # Strategy 1: Greedy from scratch
    # Strategy 2: Greedy excluding small numbers
    # Strategy 3: Random restarts
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        in_set = [False] * num_nodes
        
        # Initialize with a greedy heuristic
        # Randomly decide to exclude some small numbers to explore different basins
        # The knowledge suggests excluding 2 is good.
        
        # Let's try a few specific initializations
        if num_restarts == 1:
            # Standard greedy
            pass
        elif num_restarts == 2:
            # Force exclude 2
            in_set[0] = False # 2 is index 0
            # We need a way to mark "excluded" vs "not yet added"
            # For simplicity, just don't add it in the greedy phase?
            # Or add it to a "forbidden" list?
            # Let's just use the hill climber to remove it if it's bad.
            # To bias, we can start with 2 NOT in set, and run greedy add.
            # But greedy add might add 2 back if it's beneficial.
            # To prevent 2, we can temporarily set its weight to 0 or mark it forbidden.
            
        # Let's implement a parameterized greedy init
        # Exclude a random subset of small numbers
        exclude_small = []
        if num_restarts > 1:
            # Randomly exclude 1 to 3 small numbers from {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
            candidates = list(range(0, 15)) # indices for 2..16
            k = random.randint(1, 3)
            exclude_small = random.sample(candidates, k)
        
        # Greedy Add
        # Sort nodes by weight (1/x) descending, i.e., x ascending
        # But we want to add nodes that are compatible.
        # Simple greedy: iterate x from 2 to 256, if compatible and not excluded, add.
        # This is the seed's strategy.
        
        # Better greedy: Iterate nodes sorted by weight.
        # If a node is excluded, skip it.
        for i in range(num_nodes):
            if i in exclude_small:
                continue
            if can_add(in_set, i):
                in_set[i] = True
        
        # Local Search
        hill_climb(in_set)
        
        current_score = get_score(in_set)
        if current_score > best_score:
            best_score = current_score
            best_set = in_set[:]
            
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break

    # Convert best_set to list of integers
    result = []
    for i in range(num_nodes):
        if best_set[i]:
            result.append(i + 2)
            
    # If no result found (shouldn't happen), return a valid default
    if not result:
        return [256]
        
    return result
# EVOLVE-BLOCK-END

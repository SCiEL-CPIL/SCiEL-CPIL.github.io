import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # valid_mask: bit v is set if 2*v is not a square
    valid_mask = 0
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_mask |= (1 << v)
    
    # conflict_mask[v]: bit u is set if u+v is a square
    # We only care about u in valid_mask
    conflict_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if u == v:
                continue
            if u + v in squares:
                mask |= (1 << u)
        # Only keep conflicts with valid nodes
        conflict_mask[v] = mask & valid_mask
        
    # List of valid nodes
    valid_nodes = [v for v in range(1, N + 1) if (valid_mask >> v) & 1]
    
    def local_add(chosen_mask, order):
        """
        Repeatedly add any node from 'order' that has no conflicts with chosen_mask.
        """
        improved = True
        while improved:
            improved = False
            for v in order:
                if (chosen_mask >> v) & 1:
                    continue
                if (conflict_mask[v] & chosen_mask) == 0:
                    chosen_mask |= (1 << v)
                    improved = True
        return chosen_mask

    best_mask = 0
    best_size = 0
    
    # Initial deterministic greedy: Order by degree (number of conflicts) ascending
    # Nodes with fewer conflicts are "easier" to place and leave more room for others
    degrees = [bin(conflict_mask[v]).count('1') for v in range(N + 1)]
    order_sorted = sorted(valid_nodes, key=lambda x: degrees[x])
    
    curr = local_add(0, order_sorted)
    size = bin(curr).count('1')
    if size > best_size:
        best_size = size
        best_mask = curr

    # Randomized restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Strategy 1: Random Greedy
        order = valid_nodes[:]
        random.shuffle(order)
        curr = local_add(0, order)
        size = bin(curr).count('1')
        if size > best_size:
            best_size = size
            best_mask = curr
            
        # Strategy 2: Perturbation from current best
        # Remove 1 or 2 random nodes from best_mask, then local_add with random order
        if best_mask != 0:
            bits_in_best = [i for i in valid_nodes if (best_mask >> i) & 1]
            
            # Remove 1 node
            if bits_in_best:
                r1 = random.choice(bits_in_best)
                curr = best_mask & ~(1 << r1)
                order = valid_nodes[:]
                random.shuffle(order)
                curr = local_add(curr, order)
                size = bin(curr).count('1')
                if size > best_size:
                    best_size = size
                    best_mask = curr

            # Remove 2 nodes
            if len(bits_in_best) >= 2:
                r1, r2 = random.sample(bits_in_best, 2)
                curr = best_mask & ~(1 << r1) & ~(1 << r2)
                order = valid_nodes[:]
                random.shuffle(order)
                curr = local_add(curr, order)
                size = bin(curr).count('1')
                if size > best_size:
                    best_size = size
                    best_mask = curr

    # Convert best_mask to list
    result = []
    for v in range(1, N + 1):
        if (best_mask >> v) & 1:
            result.append(v)
            
    if not result:
        result = [1]
        
    return result

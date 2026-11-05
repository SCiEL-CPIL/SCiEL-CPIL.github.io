import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 256
    # Precompute squares
    squares = set()
    s = 2
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # Determine which numbers are "diagonal-safe" (2*x is not a perfect square)
    diagonal_safe = [False] * (N + 1)
    for x in range(1, N + 1):
        diagonal_safe[x] = (2 * x) not in squares
    
    # Build conflict graph: conflict[i] = set of j != i such that i+j is a perfect square
    # Only consider j where both i and j are diagonal_safe
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        if not diagonal_safe[i]:
            continue
        for sq in squares:
            j = sq - i
            if 1 <= j <= N and j != i and diagonal_safe[j]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Get list of valid nodes
    valid_nodes = [x for x in range(1, N + 1) if diagonal_safe[x]]
    n_valid = len(valid_nodes)
    
    # Map from value to index
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build adjacency as bitmasks for fast operations
    # Use Python integers as bitsets
    adj = [0] * n_valid
    for i in range(n_valid):
        v = valid_nodes[i]
        for j in conflict[v]:
            adj[i] |= (1 << val_to_idx[j])
    
    def is_independent(mask):
        """Check if the set of nodes indicated by mask is independent."""
        # For each node in the set, check no other node in the set is adjacent
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            m ^= lb
            if adj[i] & m:
                return False
        return True
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def greedy_construct(order, adj_list):
        """Greedy: process nodes in given order, add if no conflict with already selected."""
        selected = set()
        selected_mask = 0
        for i in order:
            if adj_list[i] & selected_mask == 0:
                selected.add(i)
                selected_mask |= (1 << i)
        return selected_mask, selected
    
    def local_search(mask, adj_list, max_iters=500):
        """Try to improve by removing one node and adding a different one."""
        current_mask = mask
        current_size = count_bits(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try removing each node and see if we can add a better one
            m = current_mask
            improved = False
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                m ^= lb
                
                new_mask = current_mask ^ lb  # remove i
                # Find all nodes not in new_mask that can be added
                candidates = 0
                for j in range(n_valid):
                    if not (new_mask & (1 << j)) and (adj_list[j] & new_mask) == 0:
                        candidates |= (1 << j)
                
                if candidates:
                    # Try adding the best candidate (highest degree reduction? or just first)
                    # Try all candidates
                    cm = candidates
                    while cm:
                        lb2 = cm & (-cm)
                        j = lb2.bit_length() - 1
                        cm ^= lb2
                        test_mask = new_mask | lb2
                        if count_bits(test_mask) > best_size:
                            best_mask = test_mask
                            best_size = count_bits(test_mask)
                            current_mask = test_mask
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_mask, best_size
    
    best_mask = 0
    best_size = 0
    best_solution = None
    
    # Strategy 1: Degree-based greedy ordering
    # Compute degrees
    degrees = [count_bits(adj[i]) for i in range(n_valid)]
    
    # Try multiple random orderings biased by degree
    n_attempts = 0
    while time.time() < deadline - 10:
        n_attempts += 1
        
        if n_attempts % 50 == 1:
            # Pure degree ordering (ascending degree)
            order = sorted(range(n_valid), key=lambda i: degrees[i])
            mask, selected = greedy_construct(order, adj)
            size = count_bits(mask)
            if size > best_size:
                best_size = size
                best_mask = mask
                best_solution = [valid_nodes[i] for i in selected]
            
            # Local search
            mask2, size2 = local_search(mask, adj, max_iters=200)
            if size2 > best_size:
                best_size = size2
                best_mask = mask2
                best_solution = [valid_nodes[i] for i in range(n_valid) if best_mask & (1 << i)]
        
        elif n_attempts % 50 == 2:
            # Descending degree
            order = sorted(range(n_valid), key=lambda i: -degrees[i])
            mask, selected = greedy_construct(order, adj)
            size = count_bits(mask)
            if size > best_size:
                best_size = size
                best_mask = mask
                best_solution = [valid_nodes[i] for i in selected]
            
            mask2, size2 = local_search(mask, adj, max_iters=200)
            if size2 > best_size:
                best_size = size2
                best_mask = mask2
                best_solution = [valid_nodes[i] for i in range(n_valid) if best_mask & (1 << i)]
        
        else:
            # Randomized ordering with degree bias
            # Weight by inverse degree
            weights = [1.0 / (degrees[i] + 1) for i in range(n_valid)]
            total_w = sum(weights)
            order = list(range(n_valid))
            remaining = list(range(n_valid))
            for pos in range(n_valid):
                # Weighted random pick
                r = random.random() * total_w
                cum = 0
                chosen = remaining[0]
                for idx, node in enumerate(remaining):
                    cum += weights[node]
                    if r <= cum:
                        chosen = node
                        break
                order[pos] = chosen
                remaining.remove(chosen)
                total_w -= weights[chosen]
            
            mask, selected = greedy_construct(order, adj)
            size = count_bits(mask)
            if size > best_size:
                best_size = size
                best_mask = mask
                best_solution = [valid_nodes[i] for i in selected]
            
            # Quick local search
            mask2, size2 = local_search(mask, adj, max_iters=100)
            if size2 > best_size:
                best_size = size2
                best_mask = mask2
                best_solution = [valid_nodes[i] for i in range(n_valid) if best_mask & (1 << i)]
    
    # Final local search on best solution
    if time.time() < deadline - 3:
        mask2, size2 = local_search(best_mask, adj, max_iters=500)
        if size2 > best_size:
            best_size = size2
            best_mask = mask2
            best_solution = [valid_nodes[i] for i in range(n_valid) if best_mask & (1 << i)]
    
    if best_solution is None or len(best_solution) == 0:
        # Fallback: just return a single diagonal-safe number
        for x in range(1, N + 1):
            if diagonal_safe[x]:
                return [x]
        return [1]
    
    return best_solution

import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency masks
    adj_mask = [0] * N
    
    # Identify vertices with self-loops (2*v is a square)
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden.add(i)
        
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                if j != i:
                    adj_mask[i] |= (1 << j)

    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    candidate_set = set(candidates)
    
    # Precompute degrees for sorting
    degrees = [bin(adj_mask[i]).count('1') for i in range(N)]

    def get_size(mask):
        return bin(mask).count('1')

    def greedy_solve(order):
        """
        Greedy construction with a given order.
        """
        current_mask = 0
        
        for v in order:
            if v not in candidate_set:
                continue
            # Check if v conflicts with any u in current_mask
            if not (adj_mask[v] & current_mask):
                current_mask |= (1 << v)
                
        return current_mask

    def local_improve(mask):
        """
        Add vertices that don't conflict. Repeat until no more can be added.
        """
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if (mask >> v) & 1:
                    continue
                if not (adj_mask[v] & mask):
                    mask |= (1 << v)
                    improved = True
        return mask

    def local_search(mask, max_iterations=200):
        """
        Local search with 1-removal, 1-1 swap, and 2-removal moves.
        """
        mask = local_improve(mask)
        best_size = get_size(mask)
        best_mask = mask
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            if time.time() - start_time > DEADLINE - 3:
                return best_mask
            improved = False
            iter_count += 1
            
            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            verts_not_in_set = [v for v in candidates if not ((mask >> v) & 1)]
            
            # Try 1-removal + refill
            random.shuffle(verts_in_set)
            for v in verts_in_set:
                if time.time() - start_time > DEADLINE - 3:
                    return best_mask
                new_mask = mask & ~(1 << v)
                new_mask = local_improve(new_mask)
                new_size = get_size(new_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                    mask = new_mask
                    improved = True
                    break
            
            if improved:
                continue

            # Try 1-1 swap: remove one from set, add one not in set
            random.shuffle(verts_in_set)
            random.shuffle(verts_not_in_set)
            swap_found = False
            for v_out in verts_in_set[:30]:
                if time.time() - start_time > DEADLINE - 3:
                    return best_mask
                for v_in in verts_not_in_set[:30]:
                    if (adj_mask[v_in] & mask) & ~(1 << v_out):
                        continue  # v_in still conflicts with remaining set
                    new_mask = (mask & ~(1 << v_out)) | (1 << v_in)
                    new_mask = local_improve(new_mask)
                    new_size = get_size(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if improved:
                continue

            # Try 2-removal + refill
            if len(verts_in_set) >= 2:
                random.shuffle(verts_in_set)
                pairs_tried = 0
                max_pairs = 40
                for i in range(len(verts_in_set)):
                    if pairs_tried >= max_pairs:
                        break
                    for j in range(i + 1, len(verts_in_set)):
                        if pairs_tried >= max_pairs:
                            break
                        pairs_tried += 1
                        v1 = verts_in_set[i]
                        v2 = verts_in_set[j]
                        new_mask = mask & ~(1 << v1) & ~(1 << v2)
                        new_mask = local_improve(new_mask)
                        new_size = get_size(new_mask)
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break

        return best_mask

    def simulated_annealing(mask, steps=500):
        """
        Simple simulated annealing to escape local optima.
        """
        current_mask = local_improve(mask)
        current_size = get_size(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        verts_in_set = [v for v in range(N) if (current_mask >> v) & 1]
        verts_not_in_set = [v for v in candidates if not ((current_mask >> v) & 1)]
        
        temp = 1.0
        cooling = 0.995
        
        for step in range(steps):
            if time.time() - start_time > DEADLINE - 3:
                break
            temp *= cooling
            
            verts_in_set = [v for v in range(N) if (current_mask >> v) & 1]
            verts_not_in_set = [v for v in candidates if not ((current_mask >> v) & 1)]
            
            if not verts_in_set or not verts_not_in_set:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and verts_not_in_set:
                # Add a random vertex (may create conflict, we'll fix)
                v = random.choice(verts_not_in_set)
                new_mask = current_mask | (1 << v)
                # Check validity
                if (adj_mask[v] & new_mask) & ~(1 << v):
                    continue  # conflict, skip
                delta = 1
            elif move_type < 0.6 and verts_in_set:
                # Remove a random vertex
                v = random.choice(verts_in_set)
                new_mask = current_mask & ~(1 << v)
                delta = -1
            else:
                # Swap
                if not verts_in_set or not verts_not_in_set:
                    continue
                v_out = random.choice(verts_in_set)
                v_in = random.choice(verts_not_in_set)
                # Check if v_in conflicts with (current_mask - v_out)
                temp_mask = current_mask & ~(1 << v_out)
                if (adj_mask[v_in] & temp_mask):
                    continue
                new_mask = temp_mask | (1 << v_in)
                delta = 0  # size unchanged, but structure changes
            
            new_size = get_size(new_mask)
            delta_size = new_size - current_size
            
            if delta_size > 0 or random.random() < math.exp(delta_size / max(temp, 0.01)):
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        
        return best_mask

    best_size = 0
    best_mask = 0
    
    # Phase 1: Multiple greedy initializations
    strategies = []
    
    # Low degree first
    strategies.append(sorted(candidates, key=lambda v: degrees[v]))
    # High degree first
    strategies.append(sorted(candidates, key=lambda v: -degrees[v]))
    
    # Modular classes
    for r in range(8):
        class_verts = [v for v in candidates if (v + 1) % 8 == r]
        if class_verts:
            strategies.append(sorted(class_verts, key=lambda v: degrees[v]))
    
    # Random orderings
    for _ in range(5):
        s = candidates[:]
        random.shuffle(s)
        strategies.append(s)
    
    for order in strategies:
        if time.time() - start_time > DEADLINE - 10:
            break
        s_mask = greedy_solve(order)
        s_mask = local_improve(s_mask)
        s_size = get_size(s_mask)
        if s_size > best_size:
            best_size = s_size
            best_mask = s_mask
    
    # Phase 2: Iterated local search with restarts
    while time.time() - start_time < DEADLINE - 5:
        
        # Choose perturbation strategy
        r = random.random()
        if r < 0.3:
            # Fresh random start
            order = candidates[:]
            random.shuffle(order)
            s_mask = greedy_solve(order)
            s_mask = local_improve(s_mask)
        elif r < 0.6:
            # Perturb best: remove random subset
            s_mask = best_mask
            verts_in_set = [v for v in range(N) if (s_mask >> v) & 1]
            random.shuffle(verts_in_set)
            remove_count = max(1, len(verts_in_set) // 8)
            for v in verts_in_set[:remove_count]:
                s_mask &= ~(1 << v)
            s_mask = local_improve(s_mask)
        elif r < 0.8:
            # Perturb best: remove more aggressively
            s_mask = best_mask
            verts_in_set = [v for v in range(N) if (s_mask >> v) & 1]
            random.shuffle(verts_in_set)
            remove_count = max(2, len(verts_in_set) // 5)
            for v in verts_in_set[:remove_count]:
                s_mask &= ~(1 << v)
            s_mask = local_improve(s_mask)
        else:
            # Simulated annealing from best
            s_mask = simulated_annealing(best_mask, steps=300)
        
        # Local search
        if time.time() - start_time < DEADLINE - 3:
            s_mask = local_search(s_mask, max_iterations=100)
        
        s_size = get_size(s_mask)
        if s_size > best_size:
            best_size = s_size
            best_mask = s_mask

    # Final polish: try to add any remaining vertices
    best_mask = local_improve(best_mask)

    # Convert mask back to list of 1-based values
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

import time
import random

def construct():
        N = 512
        start_time = time.time()
        time_limit = 55.0

        # Precompute APs
        ap_by_elem = [[] for _ in range(N)]
        for a in range(N):
            for d in range(1, N // 3 + 1):
                a3 = a + 3 * d
                if a3 >= N:
                    break
                p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
                ap_by_elem[p0].append((p1, p2, p3))
                ap_by_elem[p1].append((p0, p2, p3))
                ap_by_elem[p2].append((p0, p1, p3))
                ap_by_elem[p3].append((p0, p1, p2))

        def can_add(mask, x):
            for (o1, o2, o3) in ap_by_elem[x]:
                if (mask >> o1) & 1 and (mask >> o2) & 1 and (mask >> o3) & 1:
                    return False
            return True

        def greedy_fill(mask, order=None):
            if order is None:
                order = list(range(N))
            for x in order:
                if (mask >> x) & 1:
                    continue
                if can_add(mask, x):
                    mask |= (1 << x)
            return mask

        def count_bits(mask):
            return bin(mask).count('1')

        best_mask = 0
        best_size = 0

        # 1. Initialization with multiple random orders
        random.seed(42)
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            order = list(range(N))
            random.shuffle(order)
            m = greedy_fill(0, order)
            sz = count_bits(m)
            if sz > best_size:
                best_size = sz
                best_mask = m

        current_mask = best_mask
        current_size = best_size

        # 2. Local Search with K-Swap
        # We try to remove K elements and greedily refill.
        # If the new size is larger, we accept.
        # If the new size is equal, we might accept to escape local optima (with low prob).
        # If smaller, reject (unless doing annealing, but let's keep it simple first).
        
        stuck_count = 0
        max_stuck = 50
        
        while time.time() - start_time < time_limit:
            # Try to add any element (polish)
            # This is fast, do it every iteration
            improved = False
            # Shuffle order for adding to be non-deterministic if multiple available?
            # Or just iterate.
            for x in range(N):
                if not (current_mask >> x) & 1:
                    if can_add(current_mask, x):
                        current_mask |= (1 << x)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
            
            if improved:
                stuck_count = 0
                continue

            # If stuck, try K-Swap
            # K=1, 2, 3
            swap_improved = False
            for K in [1, 2, 3]:
                if time.time() - start_time > time_limit:
                    break
                
                # Try a few random K-swaps
                for attempt in range(5):
                    if time.time() - start_time > time_limit:
                        break
                    
                    in_set = [x for x in range(N) if (current_mask >> x) & 1]
                    if len(in_set) < K:
                        break
                        
                    # Pick K random elements to remove
                    removals = random.sample(in_set, K)
                    new_mask = current_mask
                    for x in removals:
                        new_mask &= ~(1 << x)
                    
                    # Greedy refill
                    order = list(range(N))
                    random.shuffle(order)
                    new_mask = greedy_fill(new_mask, order)
                    new_size = count_bits(new_mask)
                    
                    if new_size > current_size:
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        swap_improved = True
                        break
                    elif new_size == current_size and random.random() < 0.1:
                        # Accept equal size with small probability to change configuration
                        current_mask = new_mask
                        # current_size unchanged
                        swap_improved = True # Count as progress to reset stuck counter?
                        break

                if swap_improved:
                    break

            if swap_improved:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    # Restart from best? Or just continue?
                    # Let's restart current from best to explore different basin
                    current_mask = best_mask
                    current_size = best_size
                    stuck_count = 0
                    max_stuck = 50 # Reset counter

        # Final polish
        for x in range(N):
            if not (best_mask >> x) & 1:
                if can_add(best_mask, x):
                    best_mask |= (1 << x)
                    best_size += 1

        result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
        return sorted(result)

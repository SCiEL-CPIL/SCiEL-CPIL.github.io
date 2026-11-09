import time
import random

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        # x is 1st: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1) and (mask >> (x + 3*d) & 1):
                return False
        # x is 2nd: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (mask >> (x - d) & 1) and (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1):
                return False
        # x is 3rd: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1) and (mask >> (x + d) & 1):
                return False
        # x is 4th: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (mask >> (x - 3*d) & 1) and (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1):
                return False
        return True

    def fast_greedy(rng=None):
        """Fast greedy construction with optional randomization."""
        mask = 0
        if rng is None:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        
        return mask

    def local_search(mask, rng, deadline):
        """Local search with add and swap moves."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try adding any blocked element
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            for x in blocked_list:
                if time.time() >= deadline:
                    return mask, S_size
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
                    blocked.discard(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try swaps: remove r, add a
            if not blocked:
                break
                
            # Try multiple random blocked elements
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            
            swap_found = False
            for a in blocked_list[:10]:  # Try up to 10 random blocked elements
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                # Try up to 50 random removals
                for r in S_list[:50]:
                    if time.time() >= deadline:
                        return mask, S_size
                    
                    mask_without_r = mask & ~(1 << r)
                    
                    if can_add(mask_without_r, a):
                        # Perform swap
                        mask = mask_without_r | (1 << a)
                        blocked.discard(a)
                        blocked.add(r)
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return mask, S_size

    # Main
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different random seeds
    for trial in range(20):
        if time.time() >= deadline - 10:
            break
        
        rng = random.Random(42 + trial)
        
        # Initial seed with randomization
        mask = fast_greedy(rng)
        
        # Run local search
        mask, size = local_search(mask, rng, deadline - 5)
        
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")

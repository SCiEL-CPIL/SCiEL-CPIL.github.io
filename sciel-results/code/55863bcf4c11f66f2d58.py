import time
import random

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Cache for can_add results to speed up repeated checks
    # Note: Mask changes frequently, so caching (mask, x) -> bool is risky for memory/invalidation.
    # We rely on the speed of the bitmask check instead.
    
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

    def polish_pass(mask, deadline):
        """Final polish: try adding elements in multiple orders."""
        S_size = bin(mask).count('1')
        blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
        
        # Try natural order
        for x in blocked:
            if time.time() >= deadline:
                break
            if can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try reverse order
        for x in reversed(blocked):
            if time.time() >= deadline:
                break
            if not (mask >> x & 1) and can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try a few more random passes
        rng = random.Random(int(time.time() * 1000) % 10000)
        for _ in range(5):
            blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
            if not blocked:
                break
            rng.shuffle(blocked)
            for x in blocked:
                if time.time() >= deadline:
                    break
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
        
        return mask, S_size

    def local_search(mask, rng, deadline):
        """Local search with add and swap moves."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        no_improve_count = 0
        max_no_improve = 100 # Increased slightly
        
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
            # Increased attempts to find a valid swap
            for a in blocked_list[:30]:  
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                # Try up to 100 random removals
                for r in S_list[:100]:
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
    
    # Use ILS: Start with greedy, then iterate perturb+search
    # This is often better than independent restarts for structured problems
    
    rng = random.Random(42)
    
    # Initial Greedy
    current_mask = fast_greedy(rng)
    current_mask, current_size = local_search(current_mask, rng, deadline - 15)
    
    if current_size > best_size:
        best_size = current_size
        best_mask = current_mask
    
    # ILS Loop
    # We will perturb the current best (or current state) and search again.
    # To avoid getting stuck in one basin, we occasionally restart from scratch if stuck.
    
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        
        # Perturbation: Remove 3-5 random elements to break the local optimum structure
        # This creates "holes" that might allow different elements to be added
        perturb_mask = best_mask
        S_list = [x for x in range(1, N + 1) if (perturb_mask >> x & 1)]
        
        # Randomly remove k elements
        k = 3 + rng.randint(0, 2) # 3 to 5 elements
        if len(S_list) > k:
            to_remove = rng.sample(S_list, k)
            for r in to_remove:
                perturb_mask &= ~(1 << r)
        
        # Run local search from this perturbed state
        # Use a tighter deadline for this iteration to allow more iterations
        iter_deadline = min(deadline - 10, time.time() + 5.0) # 5s per iteration max
        new_mask, new_size = local_search(perturb_mask, rng, iter_deadline)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
        else:
            # If we didn't improve, we might want to restart from scratch occasionally
            # to avoid exploring a bad basin for too long.
            if iterations % 5 == 0:
                # Full restart
                current_mask = fast_greedy(rng)
                current_mask, current_size = local_search(current_mask, rng, deadline - 10)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final polish on the best solution
    if best_size > 0 and time.time() < deadline - 2:
        best_mask, best_size = polish_pass(best_mask, deadline - 1)
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")

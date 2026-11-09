import time
import random

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial seed: Greedy with bitmask
    # Or better: Base-4 style heuristic
    # Let's stick to a fast greedy first, then local search.
    
    def fast_greedy():
        mask = 0
        for x in range(1, N + 1):
            # Check if adding x creates 4-AP
            ok = True
            # x is 1st: x, x+d, x+2d, x+3d
            # Need x+d, x+2d, x+3d in mask
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1) and (mask >> (x + 3*d) & 1):
                    ok = False
                    break
            if not ok:
                continue
                
            # x is 2nd: x-d, x, x+d, x+2d
            max_d2 = min(x - 1, (N - x) // 2)
            for d in range(1, max_d2 + 1):
                if (mask >> (x - d) & 1) and (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1):
                    ok = False
                    break
            if not ok:
                continue
                
            # x is 3rd: x-2d, x-d, x, x+d
            max_d3 = min((x - 1) // 2, N - x)
            for d in range(1, max_d3 + 1):
                if (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1) and (mask >> (x + d) & 1):
                    ok = False
                    break
            if not ok:
                continue
                
            # x is 4th: x-3d, x-2d, x-d, x
            max_d4 = (x - 1) // 3
            for d in range(1, max_d4 + 1):
                if (mask >> (x - 3*d) & 1) and (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1):
                    ok = False
                    break
            if not ok:
                continue
                
            if ok:
                mask |= (1 << x)
                
        return mask

    # Local Search
    def local_search(mask, rng, deadline):
        S_size = bin(mask).count('1')
        blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try adding any blocked element
            # Shuffle blocked to avoid bias
            rng.shuffle(blocked)
            for x in blocked:
                if time.time() >= deadline:
                    return mask, S_size
                # Check if x can be added
                ok = True
                max_d = (N - x) // 3
                for d in range(1, max_d + 1):
                    if (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1) and (mask >> (x + 3*d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d2 = min(x - 1, (N - x) // 2)
                for d in range(1, max_d2 + 1):
                    if (mask >> (x - d) & 1) and (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d3 = min((x - 1) // 2, N - x)
                for d in range(1, max_d3 + 1):
                    if (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1) and (mask >> (x + d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d4 = (x - 1) // 3
                for d in range(1, max_d4 + 1):
                    if (mask >> (x - 3*d) & 1) and (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1):
                        ok = False
                        break
                if not ok: continue
                
                if ok:
                    mask |= (1 << x)
                    S_size += 1
                    blocked.remove(x)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
            
            # 2. Try swaps: remove r, add a
            # To be fast, we only try a few swaps per iteration
            # Pick a random blocked element 'a'
            if not blocked:
                break
                
            a = rng.choice(blocked)
            
            # Check which 'r' in S might be blocking 'a'
            # An 'r' blocks 'a' if 'r' is part of a 4-AP with 'a' and other elements in S.
            # Instead of calculating this, we just try swapping 'a' with random 'r's.
            # But checking if 'a' fits in S\{r} is expensive if done for all r.
            
            # Optimization: Only try 'r' that are "close" to 'a' in the AP sense?
            # Or just try a limited number of random 'r's.
            
            S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
            rng.shuffle(S_list)
            
            swap_found = False
            # Try up to 20 random removals
            for r in S_list[:20]: 
                if time.time() >= deadline:
                    return mask, S_size
                
                # Temporarily remove r
                mask_without_r = mask & ~(1 << r)
                
                # Check if a fits in mask_without_r
                ok = True
                max_d = (N - a) // 3
                for d in range(1, max_d + 1):
                    if (mask_without_r >> (a + d) & 1) and (mask_without_r >> (a + 2*d) & 1) and (mask_without_r >> (a + 3*d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d2 = min(a - 1, (N - a) // 2)
                for d in range(1, max_d2 + 1):
                    if (mask_without_r >> (a - d) & 1) and (mask_without_r >> (a + d) & 1) and (mask_without_r >> (a + 2*d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d3 = min((a - 1) // 2, N - a)
                for d in range(1, max_d3 + 1):
                    if (mask_without_r >> (a - 2*d) & 1) and (mask_without_r >> (a - d) & 1) and (mask_without_r >> (a + d) & 1):
                        ok = False
                        break
                if not ok: continue
                max_d4 = (a - 1) // 3
                for d in range(1, max_d4 + 1):
                    if (mask_without_r >> (a - 3*d) & 1) and (mask_without_r >> (a - 2*d) & 1) and (mask_without_r >> (a - d) & 1):
                        ok = False
                        break
                if not ok: continue
                
                if ok:
                    # Perform swap
                    mask = mask_without_r | (1 << a)
                    # Update blocked list
                    if r in blocked: pass # r was in S, so not in blocked
                    # r is now blocked, a is now in S
                    if a in blocked:
                        blocked.remove(a)
                    blocked.append(r)
                    swap_found = True
                    break
            
            if not swap_found:
                # Try a different 'a' or break?
                # Let's try another 'a'
                # To avoid infinite loop, we limit the number of failed 'a' attempts
                # But the outer loop will just pick another 'a'.
                # We should ensure we don't get stuck in a tight loop of failing swaps.
                # If no swap found for this 'a', try next 'a'.
                pass

        return mask, S_size

    # Main
    rng = random.Random(42)
    
    # Initial seed
    mask = fast_greedy()
    
    # Run local search
    mask, size = local_search(mask, rng, deadline - 5)
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")

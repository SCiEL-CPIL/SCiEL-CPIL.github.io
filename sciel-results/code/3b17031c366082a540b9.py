import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit (evaluator kills at 90s)
    
    # 1. Precompute 4-AP conflicts using bitmasks
    # Map 1..256 to bits 0..255.
    # bit i corresponds to element i+1.
    
    def val_to_bit(x):
        return 1 << (x - 1)
    
    # forbidden_masks[v] = list of masks. If S contains any mask, v is forbidden.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    for v in range(1, n + 1):
        masks = []
        for d in range(1, n // 3 + 1):
            # Case 1: v is 1st term. v, v+d, v+2d, v+3d
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
            # Case 2: v is 2nd term. v-d, v, v+d, v+2d
            if v - d >= 1 and v + 2 * d <= n:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
            # Case 3: v is 3rd term. v-2d, v-d, v, v+d
            if v - 2 * d >= 1 and v + d <= n:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
            # Case 4: v is 4th term. v-3d, v-2d, v-d, v
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP contributions for delta
    # ap3_masks[u] = list of masks M. M represents the other two elements of a 3-AP containing u.
    # If (S & M) == M, then that 3-AP exists in S (assuming u is in S).
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks
        
    # Helper: Check if v can be added to S (S_mask)
    def can_add(S_mask, v):
        masks = forbidden_masks[v]
        for m in masks:
            if (S_mask & m) == m:
                return False
        return True
        
    # Helper: Calculate delta 3-APs for swap u -> v
    def calc_delta(S_mask, u, v):
        # S_mask contains u.
        S_mask_without_u = S_mask & ~val_to_bit(u)
        
        # Lost: count masks in ap3_masks[u] that are subset of S_mask.
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        # Gained: count masks in ap3_masks[v] that are subset of S_mask_without_u.
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # Helper: Count 3-APs in S (represented as int bitmask)
    def count_3aps_from_bitmask(S_mask):
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    # Strategy: Multi-Start Greedy Construction + Deterministic Local Search
    
    best_count = 0
    best_mask = 0
    
    # We will run several "starts" with different greedy orders
    # Start 1: Standard ascending order
    # Start 2: Descending order
    # Start 3: Random orders (multiple)
    
    def run_greedy_and_refine(order, max_refine_steps=5000):
        nonlocal best_count, best_mask
        
        S_mask = 0
        S_set = set()
        
        # Greedy phase
        for x in order:
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
                S_set.add(x)
                
        current_count = count_3aps_from_bitmask(S_mask)
        
        if current_count > best_count:
            best_count = current_count
            best_mask = S_mask
            
        # Local Search Phase: Deterministic Hill Climbing
        # We look for the best single swap (remove u, add v) that increases count.
        # If none, stop.
        
        improved = True
        steps = 0
        while improved and steps < max_refine_steps:
            if time.time() - start_time > time_limit:
                return
                
            improved = False
            steps += 1
            
            # List of elements in S
            elements_in_S = [i for i in range(1, n+1) if (S_mask >> (i-1)) & 1]
            # List of elements not in S
            elements_not_in_S = [i for i in range(1, n+1) if not ((S_mask >> (i-1)) & 1)]
            
            best_delta = 0
            best_u = -1
            best_v = -1
            
            # To speed up, we can limit the search or use heuristics.
            # With n=256, |S| is likely ~50-100, |not S| ~150-200.
            # Pairs ~ 20k. Delta calc is ~100-200 ops. 20k * 200 = 4M ops.
            # This might be slow in Python if done repeatedly.
            # Let's optimize: Only check swaps where u is "low contribution" and v is "high potential".
            # Or just do a limited random sample of swaps if the full search is too slow.
            
            # Given the time limit, let's try a full scan but break early if time is tight.
            # Actually, 4M ops in Python might take ~0.5s. If we do 5000 steps, that's 2500s. Too slow.
            # We need a faster local search or fewer steps.
            
            # Let's use a randomized local search instead of full deterministic scan.
            # Pick random u in S, random v not in S.
            # If valid and delta > 0, accept.
            # Repeat.
            
            # Wait, the prompt asks for a STRUCTURAL change.
            # The current program uses SA.
            # A "Greedy Construction" is the initial phase.
            # The refinement can be a "First-Fit Improving" local search.
            # To make it fast, we can precompute "potential" of v given S.
            # Potential(v) = number of 3-APs v would form if added to S (ignoring 4-AP constraint for a moment, or checking it).
            
            # Let's stick to a fast randomized local search with acceptance only if delta > 0 (greedy local search).
            # This is faster than SA because we don't need to calculate exp(delta/T) and we accept immediately.
            
            u = random.choice(elements_in_S)
            v = random.choice(elements_not_in_S)
            
            S_mask_without_u = S_mask & ~val_to_bit(u)
            
            valid = True
            for m in forbidden_masks[v]:
                if (S_mask_without_u & m) == m:
                    valid = False
                    break
            
            if not valid:
                # Try a few more random swaps in this iteration to find an improvement
                # If we can't find one quickly, we might be stuck.
                # For now, just continue to next outer loop iteration (which is expensive).
                # Better: try a small batch of random swaps.
                found = False
                for _ in range(10):
                    u = random.choice(elements_in_S)
                    v = random.choice(elements_not_in_S)
                    S_mask_without_u = S_mask & ~val_to_bit(u)
                    
                    valid = True
                    for m in forbidden_masks[v]:
                        if (S_mask_without_u & m) == m:
                            valid = False
                            break
                    
                    if valid:
                        delta = calc_delta(S_mask, u, v)
                        if delta > 0:
                            S_mask = S_mask_without_u | val_to_bit(v)
                            current_count += delta
                            improved = True
                            if current_count > best_count:
                                best_count = current_count
                                best_mask = S_mask
                            found = True
                            break
                if not found:
                    # No improvement found in this batch. 
                    # To avoid infinite loop, we can break or continue.
                    # If we continue, we might loop forever if no moves exist.
                    # Let's break if we've done many steps without improvement.
                    # For simplicity, break if no improvement in this batch.
                    pass
            else:
                delta = calc_delta(S_mask, u, v)
                if delta > 0:
                    S_mask = S_mask_without_u | val_to_bit(v)
                    current_count += delta
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_mask = S_mask
                else:
                    # Try a few more
                    for _ in range(10):
                        u = random.choice(elements_in_S)
                        v = random.choice(elements_not_in_S)
                        S_mask_without_u = S_mask & ~val_to_bit(u)
                        
                        valid = True
                        for m in forbidden_masks[v]:
                            if (S_mask_without_u & m) == m:
                                valid = False
                                break
                        
                        if valid:
                            delta = calc_delta(S_mask, u, v)
                            if delta > 0:
                                S_mask = S_mask_without_u | val_to_bit(v)
                                current_count += delta
                                improved = True
                                if current_count > best_count:
                                    best_count = current_count
                                    best_mask = S_mask
                                break

    # Start 1: Ascending
    run_greedy_and_refine(list(range(1, n+1)))
    
    # Start 2: Descending
    if time.time() - start_time < time_limit:
        run_greedy_and_refine(list(range(n, 0, -1)))
        
    # Start 3-10: Random orders
    while time.time() - start_time < time_limit:
        random_order = list(range(1, n+1))
        random.shuffle(random_order)
        run_greedy_and_refine(random_order)
        
    # Final fallback: if best_mask is 0 (shouldn't happen), return a valid set
    if best_mask == 0:
        # Simple greedy
        S_mask = 0
        for x in range(1, n+1):
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
        best_mask = S_mask
        
    return sorted([i+1 for i in range(n) if (best_mask >> i) & 1])

# Test run to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])

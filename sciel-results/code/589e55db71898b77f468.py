import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp[i] is an integer where bit j is 1 if mask i and mask j are comparable (i != j)
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            # Check if i is subset of j or j is subset of i
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp[i] = ci
    
    # Helper to calculate score of a selection given a selection mask
    # sel_mask: integer with bits set for selected masks
    def get_score(sel, sel_mask):
        s = 0
        for x in sel:
            s += (comp[x] & sel_mask).bit_count()
        return s // 2

    # Greedy initialization
    def greedy_init():
        selected = []
        used_mask = 0
        # Start with empty set (mask 0)
        selected.append(0)
        used_mask |= (1 << 0)
        
        for _ in range(m - 1):
            best_mask = -1
            best_score = -1
            for mask in range(total_masks):
                if (used_mask >> mask) & 1:
                    continue
                # Score is number of comparable pairs with existing selected sets
                # This is popcount(comp[mask] & used_mask)
                score = (comp[mask] & used_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = mask
            selected.append(best_mask)
            used_mask |= (1 << best_mask)
        return selected, used_mask

    best_family, best_score = greedy_init()
    
    # Local search
    def local_search(family, sel_mask):
        current_score = get_score(family, sel_mask)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try swapping each element with any other
            for i in range(m):
                if time.time() > deadline - 2.0:
                    break
                old_mask = family[i]
                # Mask of selected elements excluding old_mask
                sel_wo_old = sel_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_wo_old).bit_count()
                
                best_new_mask = -1
                best_gain = loss
                # Iterate through all possible new masks
                # Optimization: We can iterate through all masks, but skip those in sel_mask
                # This is O(4096) per swap attempt. 40 * 4096 * (bit_count ops)
                # bit_count is fast.
                for new_mask in range(total_masks):
                    if (sel_mask >> new_mask) & 1:
                        continue
                    gain = (comp[new_mask] & sel_wo_old).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_new_mask = new_mask
                
                if best_new_mask != -1:
                    # Accept swap
                    family[i] = best_new_mask
                    sel_mask = (sel_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                    current_score += best_gain - loss
                    improved = True
                    # If we made a swap, we can break and restart the outer loop
                    # or continue. Restarting is safer to ensure we don't miss better swaps
                    # caused by the change. But for speed, we can just continue.
                    # Given the structure, a single pass of improvements is usually enough for a step.
                    break # Break inner loop to restart outer loop with new state
        
        return family, sel_mask, current_score

    # Run local search
    family, sel_mask, score = local_search(best_family, best_score if False else (1 << 0)) # Wait, need correct sel_mask
    # Re-calculate sel_mask from greedy result
    sel_mask = 0
    for x in best_family:
        sel_mask |= (1 << x)
    
    family, sel_mask, score = local_search(best_family, sel_mask)
    best_score = score
    best_family = list(family)
    
    # If time remains, do restarts with perturbation
    while time.time() < deadline - 5.0:
        # Perturb the best solution
        perturbed_family = list(best_family)
        perturbed_mask = 0
        for x in perturbed_family:
            perturbed_mask |= (1 << x)
        
        # Randomly swap 5-10 elements
        num_swaps = random.randint(5, 10)
        for _ in range(num_swaps):
            i = random.randint(0, m - 1)
            old = perturbed_family[i]
            # Pick a random new mask not in perturbed_family
            while True:
                new = random.randint(0, total_masks - 1)
                if not ((perturbed_mask >> new) & 1):
                    break
            perturbed_family[i] = new
            perturbed_mask = (perturbed_mask & ~(1 << old)) | (1 << new)
        
        # Run local search on perturbed solution
        p_family, p_mask, p_score = local_search(perturbed_family, perturbed_mask)
        
        if p_score > best_score:
            best_score = p_score
            best_family = p_family
            best_sel_mask = p_mask
        else:
            # If no improvement, try a different perturbation strength or break if stuck too long
            # For now, just continue loop
            continue

    # Convert best_family masks to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure types are correct
    final_result = []
    for s in result:
        final_result.append([int(x) for x in s])
        
    return final_result

# Self-test / sanity check
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:5])
    # Verify distinctness
    assert len(set(tuple(sorted(s)) for s in res)) == 40

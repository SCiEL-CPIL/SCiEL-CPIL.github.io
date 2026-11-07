import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # Safety margin

    # Precompute subset check? 
    # a is subset of b: (a & ~b) == 0
    # comparable: (a & ~b) == 0 or (b & ~a) == 0
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_comparable(config[i], config[j]):
                    cnt += 1
        return cnt

    # Greedy Initialization
    # Start with a random set of 40? Or start with empty and add best?
    # Starting with empty and adding best is O(m * N * m) which is fine.
    
    selected = []
    available = set(range(total_masks))
    
    # To break symmetry and potentially find different local optima, 
    # we can force the first few elements or just use random tie-breaking.
    # However, a deterministic greedy is reproducible.
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in available:
            score = 0
            for s in selected:
                if is_comparable(mask, s):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
        available.remove(best_mask)
        
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    
    # Local Search (Hill Climbing)
    # We will try to swap each element with any other available mask.
    # To speed up, we maintain a set of currently selected masks for O(1) membership check.
    
    selected_set = set(selected)
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices
        for i in range(m):
            old_mask = selected[i]
            # Try all possible new masks
            # Optimization: We can skip masks that are already in selected_set (except old_mask)
            # But we have to check membership.
            
            best_new_mask = -1
            best_new_score = current_score
            
            # We can restrict candidates to those that are "close" in size or similar bits?
            # No, let's try all. 4096 is small.
            
            for new_mask in range(total_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in selected_set:
                    continue
                
                # Calculate delta score
                # Remove old_mask: lose edges with all other j != i
                # Add new_mask: gain edges with all other j != i
                
                # To avoid O(m) calculation for every candidate, we can precompute 
                # "compatibility" of new_mask with the rest of the set?
                # But the rest changes as we swap.
                # However, in a single pass, we only swap one at a time.
                # So for a fixed i, the "other" set is constant.
                
                # Let's compute the gain/loss.
                # loss = sum(is_comparable(old_mask, selected[j]) for j != i)
                # gain = sum(is_comparable(new_mask, selected[j]) for j != i)
                # delta = gain - loss
                
                # We can precompute the loss for old_mask once per i.
                # But gain depends on new_mask.
                
                # Let's just do the O(m) check. 4096 * 40 = 163k ops per i.
                # 40 i's = 6.5M ops per pass.
                # This is fast enough.
                
                # Optimization: 
                # Precompute a list of "others" = selected excluding selected[i]
                # But this is O(m) to create.
                
                # Let's just inline the loop.
                
                gain = 0
                # Iterate through all other selected sets
                # We can break early if gain + (remaining_max_possible) < loss? 
                # Max possible gain is m-1.
                
                # Simple loop
                for j in range(m):
                    if j == i:
                        continue
                    if is_comparable(new_mask, selected[j]):
                        gain += 1
                            
                # We need loss. 
                # We can compute loss once outside the new_mask loop.
                
                # Refactor:
                # loss = 0
                # for j in range(m):
                #    if j != i and is_comparable(old_mask, selected[j]):
                #        loss += 1
                
                # So:
                # if gain - loss > 0:
                #    ...
                
                # Let's restructure to compute loss once.
                
            # Restructured Loop for i
            loss = 0
            for j in range(m):
                if j != i and is_comparable(old_mask, selected[j]):
                    loss += 1
            
            for new_mask in range(total_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in selected_set:
                    continue
                
                gain = 0
                for j in range(m):
                    if j == i:
                        continue
                    if is_comparable(new_mask, selected[j]):
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    # Accept swap
                    current_score += delta
                    selected[i] = new_mask
                    selected_set.remove(old_mask)
                    selected_set.add(new_mask)
                    old_mask = new_mask # update for next iterations? No, i loop moves on.
                    improved = True
                    # After a swap, the 'loss' for this i is no longer valid if we continue swapping i?
                    # But we break out of new_mask loop?
                    # If we take first improvement, we should break.
                    break
                    
        if not improved:
            # Try to escape local minimum with a small random perturbation?
            # Or just break if we've been stuck for a while.
            # Let's try a random restart of 2 elements.
            if random.random() < 0.5:
                # Swap two random elements with random new masks
                i1 = random.randint(0, m-1)
                i2 = random.randint(0, m-1)
                if i1 == i2: continue
                
                m1 = random.randint(0, total_masks-1)
                m2 = random.randint(0, total_masks-1)
                
                # Ensure distinctness and validity
                # This is complex. Let's just stick to hill climbing.
                # If hill climbing stalls, it's likely a strong local optimum.
                pass
            else:
                break

    # Convert to lists of ints
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result

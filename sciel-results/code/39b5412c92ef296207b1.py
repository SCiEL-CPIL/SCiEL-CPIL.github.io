import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit

    # Precomputation
    # 1. Find all 4-APs
    four_aps = []
    for a in range(1, N - 2):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # Map each element to the witness masks for 4-APs it belongs to
    # witness_masks[x] is a list of bitmasks. Each bitmask represents the other 3 elements in a 4-AP containing x.
    # If (current_set & mask) == mask, then adding x would complete a 4-AP.
    witness_masks = {i: [] for i in range(1, N + 1)}
    for ap in four_aps:
        for i in range(4):
            others = ap[:i] + ap[i+1:]
            mask = 0
            for o in others:
                mask |= (1 << o)
            witness_masks[ap[i]].append(mask)

    # 2. Find all 3-APs and precompute gain masks
    # gain_masks[x] is a list of bitmasks. Each bitmask represents a pair (u, v) such that {x, u, v} is a 3-AP.
    # gain(x, S) = number of masks m in gain_masks[x] such that (S & m) == m
    gain_masks = {i: [] for i in range(1, N + 1)}
    
    # Iterate through all 3-APs (a, b, c)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = b + d
            if c <= N:
                # AP is (a, b, c)
                # For x=a, pair is (b, c)
                mask_bc = (1 << b) | (1 << c)
                gain_masks[a].append(mask_bc)
                
                # For x=b, pair is (a, c)
                mask_ac = (1 << a) | (1 << c)
                gain_masks[b].append(mask_ac)
                
                # For x=c, pair is (a, b)
                mask_ab = (1 << a) | (1 << b)
                gain_masks[c].append(mask_ab)

    def is_valid_addition(x, S_mask):
        for m in witness_masks[x]:
            if (S_mask & m) == m:
                return False
        return True

    def calc_gain_add(x, S_mask):
        gain = 0
        for m in gain_masks[x]:
            if (S_mask & m) == m:
                gain += 1
        return gain

    def calc_loss_remove(x, S_mask):
        # Loss is the number of 3-APs that are destroyed by removing x.
        # This is the number of pairs (u, v) in gain_masks[x] such that u, v are in S.
        # Note: S_mask includes x, but the pairs do not include x, so this is correct.
        loss = 0
        for m in gain_masks[x]:
            if (S_mask & m) == m:
                loss += 1
        return loss

    def count_3aps(S_mask):
        count = 0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                count += calc_gain_add(x, S_mask & ~(1 << x)) # Exclude x from S for gain calc? 
                # Wait, calc_gain_add checks if pairs are in S. The pairs don't contain x. 
                # So S_mask can be the full mask.
                # Let's re-verify. 
                # gain_masks[x] contains masks of pairs (u,v). 
                # If u,v in S, then (x,u,v) is a 3-AP. 
                # So yes, just use S_mask.
                # But we must be careful not to double count. 
                # Each 3-AP (a,b,c) is counted once for a, once for b, once for c? 
                # No, my gain_masks construction adds the pair to the mask of each element.
                # So summing gain for all x in S counts each 3-AP exactly 3 times? 
                # Let's check. 
                # AP (a,b,c). 
                # gain_masks[a] has (b,c). If a,b,c in S, gain(a) includes this.
                # gain_masks[b] has (a,c). If a,b,c in S, gain(b) includes this.
                # gain_masks[c] has (a,b). If a,b,c in S, gain(c) includes this.
                # So total sum is 3 * (# 3-APs).
                # So count_3aps should return sum // 3.
                pass
        # Better to just calculate directly for final score to be safe, 
        # or trust the delta logic. 
        # For the final score, let's do a direct count to be 100% sure.
        direct_count = 0
        for a in range(1, N + 1):
            if not (S_mask & (1 << a)): continue
            for d in range(1, (N - a) // 2 + 1):
                b = a + d
                c = b + d
                if c > N: break
                if (S_mask & (1 << b)) and (S_mask & (1 << c)):
                    direct_count += 1
        return direct_count

    # Initial Solution: Greedy
    # We want a good starting point. 
    # Random order greedy.
    A_mask = 0
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        if is_valid_addition(x, A_mask):
            A_mask |= (1 << x)
    
    best_mask = A_mask
    best_score = count_3aps(best_mask)
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        current_mask = best_mask
        current_score = best_score
        
        # Hill Climbing Phase
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            best_move_gain = -1
            best_move_x = -1
            best_move_type = None # 'add' or 'remove'
            
            # Check all possible additions
            for x in range(1, N + 1):
                if not (current_mask & (1 << x)):
                    if is_valid_addition(x, current_mask):
                        gain = calc_gain_add(x, current_mask)
                        if gain > best_move_gain:
                            best_move_gain = gain
                            best_move_x = x
                            best_move_type = 'add'
            
            # Check all possible removals
            for x in range(1, N + 1):
                if current_mask & (1 << x):
                    # Loss is negative gain
                    loss = calc_loss_remove(x, current_mask)
                    gain = -loss
                    if gain > best_move_gain:
                        best_move_gain = gain
                        best_move_x = x
                        best_move_type = 'remove'
            
            if best_move_gain > 0:
                # Apply best move
                if best_move_type == 'add':
                    current_mask |= (1 << best_move_x)
                else:
                    current_mask &= ~(1 << best_move_x)
                
                current_score += best_move_gain
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

        # Perturbation Phase
        # If we are stuck, perturb the current best (or current local opt)
        # Strategy: Remove k random elements
        # k should be small enough to keep validity likely, but large enough to escape basin
        # Let's try k between 3 and 10
        if time.time() - start_time < time_limit:
            perturbed_mask = best_mask
            elements_in_set = [x for x in range(1, N + 1) if perturbed_mask & (1 << x)]
            if elements_in_set:
                k = random.randint(3, 10)
                if k < len(elements_in_set):
                    to_remove = random.sample(elements_in_set, k)
                    for x in to_remove:
                        perturbed_mask &= ~(1 << x)
                
                # Try to hill climb from perturbed state
                p_score = count_3aps(perturbed_mask)
                
                # Quick hill climb on perturbed state
                p_improved = True
                while p_improved and time.time() - start_time < time_limit:
                    p_improved = False
                    p_best_gain = -1
                    p_best_x = -1
                    p_best_type = None
                    
                    for x in range(1, N + 1):
                        if not (perturbed_mask & (1 << x)):
                            if is_valid_addition(x, perturbed_mask):
                                gain = calc_gain_add(x, perturbed_mask)
                                if gain > p_best_gain:
                                    p_best_gain = gain
                                    p_best_x = x
                                    p_best_type = 'add'
                    
                    for x in range(1, N + 1):
                        if perturbed_mask & (1 << x):
                            loss = calc_loss_remove(x, perturbed_mask)
                            gain = -loss
                            if gain > p_best_gain:
                                p_best_gain = gain
                                p_best_x = x
                                p_best_type = 'remove'
                    
                    if p_best_gain > 0:
                        if p_best_type == 'add':
                            perturbed_mask |= (1 << p_best_x)
                        else:
                            perturbed_mask &= ~(1 << p_best_x)
                        p_score += p_best_gain
                        p_improved = True
                        
                        if p_score > best_score:
                            best_score = p_score
                            best_mask = perturbed_mask

    # Convert best_mask to list
    result = []
    for x in range(1, N + 1):
        if best_mask & (1 << x):
            result.append(x)
    return sorted(result)

# EVOLVE-BLOCK-END

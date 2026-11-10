import time
import random
import numpy as np

N = 256
DEADLINE = 85.0

def _precompute_4ap_triples():
    """
    Precompute all 4-APs and for each position, the list of (other1, other2, other3)
    such that {pos, other1, other2, other3} is a 4-AP.
    Actually, for efficient checking:
    For each pos, store list of tuples (a, b, c) such that {pos, a, b, c} is a 4-AP.
    When checking if adding pos creates a 4-AP, we check if any (a,b,c) are all in the set.
    """
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each pos, list of the other 3 elements in 4-APs containing it
    pos_to_others = [[] for _ in range(N)]
    for ap in all_4aps:
        for i in range(4):
            pos = ap[i]
            others = [ap[j] for j in range(4) if j != i]
            pos_to_others[pos].append(others)
            
    return all_4aps, pos_to_others

def _count_3ap_fast(mask):
    """Count 3-APs using difference-based vectorization."""
    n = len(mask)
    count = 0
    for d in range(1, n // 2 + 1):
        a_part = mask[:n - 2 * d]
        b_part = mask[d:n - d]
        c_part = mask[2 * d:]
        count += np.sum(a_part & b_part & c_part)
    return int(count)

def _count_4ap_violations(mask, pos_to_others):
    """Count the number of 4-APs fully contained in the set."""
    count = 0
    # To avoid double counting, we can just iterate through all 4-APs.
    # Or use the structure. 
    # Since N=256, total 4-APs is ~ (256^2)/6 ~ 10k. 
    # Iterating 10k APs and checking 4 bits is fast.
    # But we didn't store all_4aps in this function signature. 
    # Let's restructure to pass all_4aps or compute count differently.
    # For now, let's assume we have a global or pass it.
    # I will modify the function to take all_4aps.
    pass

def _count_4ap_violations_mask(mask, all_4aps):
    count = 0
    for ap in all_4aps:
        if mask[ap[0]] and mask[ap[1]] and mask[ap[2]] and mask[ap[3]]:
            count += 1
    return count

def _marginal_3ap_gain(mask, pos):
    """Compute how many new 3-APs are created by adding pos."""
    n = len(mask)
    gain = 0
    # pos as first (a): b = pos+d, c = pos+2d
    max_d = (n - 1 - pos) // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos + d] & mask[pos + 2 * d])
    # pos as middle (b): a = pos-d, c = pos+d
    max_d = min(pos, n - 1 - pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos - d] & mask[pos + d])
    # pos as last (c): b = pos-d, a = pos-2d
    max_d = pos // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos - d] & mask[pos - 2 * d])
    return int(gain)

def _lagrangian_annealing(all_4aps, pos_to_others, start_time, deadline, seed=0):
    """
    Simulated Annealing with Lagrangian penalty for 4-APs.
    Objective: Maximize (3AP_count - lambda * 4AP_count)
    Lambda adapts: if 4AP_count > 0, increase lambda; if 0, decrease lambda.
    """
    rng = random.Random(seed)
    mask = np.zeros(N, dtype=bool)
    
    # Initialize with a random dense set? Or empty?
    # Let's start with a random set of density 0.5
    for i in range(N):
        if rng.random() < 0.5:
            mask[i] = True
    
    current_score_3ap = _count_3ap_fast(mask)
    current_violations = _count_4ap_violations_mask(mask, all_4aps)
    
    lambda_val = 1.0
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    best_mask = mask.copy()
    best_feasible_score = -1
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Pick a random position to flip
        pos = rng.randint(0, N - 1)
        
        # Calculate change in 3AP score
        # If mask[pos] is True, we remove it. Loss = marginal gain of adding it to (mask without pos).
        # If mask[pos] is False, we add it. Gain = marginal gain of adding it to mask.
        
        if mask[pos]:
            # Removing pos
            # Temporarily remove
            mask[pos] = False
            new_3ap = _count_3ap_fast(mask)
            new_violations = _count_4ap_violations_mask(mask, all_4aps)
            delta_3ap = new_3ap - current_score_3ap
            delta_viol = new_violations - current_violations
            
            # Check if this flip is feasible (doesn't create new violations? No, removing can't create violations)
            # Actually, removing an element can only reduce or keep violations same.
            
            # Evaluate Lagrangian
            obj_old = current_score_3ap - lambda_val * current_violations
            obj_new = new_3ap - lambda_val * new_violations
            delta_obj = obj_new - obj_old
            
            # Accept if better, or with probability exp(delta_obj / T)
            if delta_obj > 0 or (delta_obj <= 0 and rng.random() < np.exp(delta_obj / T)):
                # Accept
                current_score_3ap = new_3ap
                current_violations = new_violations
                # mask[pos] is already False
            else:
                # Reject, restore
                mask[pos] = True
        else:
            # Adding pos
            # Check if adding pos creates a 4-AP
            # If it does, delta_viol will be positive.
            mask[pos] = True
            new_3ap = _count_3ap_fast(mask)
            new_violations = _count_4ap_violations_mask(mask, all_4aps)
            delta_3ap = new_3ap - current_score_3ap
            delta_viol = new_violations - current_violations
            
            obj_old = current_score_3ap - lambda_val * current_violations
            obj_new = new_3ap - lambda_val * new_violations
            delta_obj = obj_new - obj_old
            
            if delta_obj > 0 or (delta_obj <= 0 and rng.random() < np.exp(delta_obj / T)):
                # Accept
                current_score_3ap = new_3ap
                current_violations = new_violations
                # mask[pos] is already True
            else:
                # Reject, restore
                mask[pos] = False
                
        # Track best feasible
        if current_violations == 0:
            if current_score_3ap > best_feasible_score:
                best_feasible_score = current_score_3ap
                best_mask = mask.copy()
                
        # Adapt lambda
        if current_violations > 0:
            lambda_val *= 1.5
            if lambda_val > 1000:
                lambda_val = 1000
        else:
            lambda_val *= 0.9
            if lambda_val < 0.1:
                lambda_val = 0.1
                
        # Cool down
        T *= cooling
        
    return best_mask, best_feasible_score

def _strict_projection(mask, all_4aps, pos_to_others):
    """
    Given a mask that might have 4-APs, remove elements to make it 4-AP-free.
    Greedy removal: while any 4-AP exists, remove one of its elements that minimizes loss in 3APs.
    """
    mask = mask.copy()
    
    # Find all 4-APs in the set
    # We can iterate through all_4aps and check if all 4 are present.
    # If yes, we have a violation.
    
    # To be efficient, we can maintain a list of active violations.
    # But for N=256, just scanning is fine for a few iterations.
    
    max_iters = 1000
    for _ in range(max_iters):
        found_violation = False
        for ap in all_4aps:
            if mask[ap[0]] and mask[ap[1]] and mask[ap[2]] and mask[ap[3]]:
                found_violation = True
                # Decide which element to remove.
                # Calculate marginal 3AP loss for each of the 4.
                min_loss = float('inf')
                min_pos = -1
                for p in ap:
                    # Temporarily remove p
                    mask[p] = False
                    # Calculate loss: difference in 3AP count
                    # This is expensive if done naively.
                    # Approximation: count 3APs involving p.
                    # Let's just use the marginal gain function on the current mask (with p removed)
                    # Wait, marginal_gain(mask_without_p, p) is the gain of adding p back.
                    # So loss = marginal_gain(mask_without_p, p).
                    gain_if_added_back = _marginal_3ap_gain(mask, p)
                    if gain_if_added_back < min_loss:
                        min_loss = gain_if_added_back
                        min_pos = p
                    # Restore p for next check? No, we want to remove one.
                    # But we need to compare against the original mask state.
                    # Let's restore p first.
                    mask[p] = True
                    
                # Remove the one with min loss
                mask[min_pos] = False
                break # Restart scan from beginning as mask changed
                
        if not found_violation:
            break
            
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute
    all_4aps, pos_to_others = _precompute_4ap_triples()
    
    best_mask = None
    best_score = -1
    
    # Run multiple annealing runs
    # Each run should be short enough to fit in budget
    # Let's say 5 runs of ~15s each? Or 1 long run?
    # Annealing benefits from long runs. Let's do 3 runs of ~20s.
    
    num_runs = 3
    run_duration = 20.0
    
    for i in range(num_runs):
        run_deadline = start_time + (i + 1) * run_duration
        if run_deadline > deadline - 5:
            run_deadline = deadline - 5
            
        if time.time() > run_deadline:
            break
            
        mask, score = _lagrangian_annealing(all_4aps, pos_to_others, start_time, run_deadline, seed=i)
        
        # The annealing might end with a feasible set (if lambda pushed it there)
        # But it's not guaranteed. We should project if necessary.
        # However, the annealing tracks best_feasible. 
        # If best_feasible_score is -1, it means no feasible set was found during annealing?
        # Unlikely, as it starts random and cools.
        
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
            
        # Also, let's try to project the final mask of this run if it's infeasible?
        # The annealing returns best_mask which is guaranteed feasible (violations==0).
        
    if best_mask is None:
        # Fallback: simple greedy
        mask = np.zeros(N, dtype=bool)
        mask[0] = True
        return [1]
        
    # Final polish: Local search on the best feasible mask
    if time.time() < deadline - 2:
        # Simple hill climbing: try to add any element that doesn't create 4-AP and increases 3-AP
        # We need a fast 4-AP check for addition.
        # blocked[p] logic from original code is good for this.
        
        # Recompute blocked for current best_mask
        blocked = np.zeros(N, dtype=np.int32)
        for p in np.where(best_mask)[0]:
            for others in pos_to_others[p]:
                if best_mask[others[0]] and best_mask[others[1]] and best_mask[others[2]]:
                    # Wait, pos_to_others[p] contains the OTHER 3.
                    # If all 3 others are in the set, then adding p creates a 4-AP?
                    # No, p is ALREADY in the set.
                    # We want to know if adding a NEW element q creates a 4-AP.
                    # If q is added, and there is a 4-AP (q, a, b, c) where a,b,c are in set.
                    # So for each q not in set, check if any 4-AP containing q has the other 3 in set.
                    pass
        
        # Let's just do a simple scan for additions
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for q in range(N):
                if best_mask[q]:
                    continue
                # Check if adding q creates 4-AP
                creates_4ap = False
                for others in pos_to_others[q]:
                    if best_mask[others[0]] and best_mask[others[1]] and best_mask[others[2]]:
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # Check 3AP gain
                gain = _marginal_3ap_gain(best_mask, q)
                if gain > 0:
                    best_mask[q] = True
                    best_score += gain
                    improved = True
                    # Restart scan? Or continue? 
                    # Continuing might miss interactions, but it's fast.
                    # Let's break to restart to be safe? 
                    # No, just continue, it's a heuristic polish.
                    
    result = sorted([int(i) + 1 for i in np.where(best_mask)[0]])
    
    if len(result) == 0:
        return [1]
        
    return result

import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _precompute_4ap_structure():
    """
    Precompute all 4-APs and for each position, the list of 4-APs it belongs to.
    Also precompute for each 4-AP, the indices of the other 3 elements.
    """
    all_4aps = []
    pos_to_4aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_4aps)
            all_4aps.append(ap)
            for p in ap:
                pos_to_4aps[p].append(idx)
                
    return all_4aps, pos_to_4aps


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


def _marginal_3ap_gain(mask, pos):
    """Compute how many new 3-APs are created by adding pos (or lost by removing)."""
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


def _count_4aps_in_mask(mask):
    """Count the number of 4-APs fully contained in mask."""
    count = 0
    # This is slow if done naively, but we can use the precomputed structure
    # However, for SA, we need fast evaluation. 
    # Let's maintain a 'blocked' count or similar.
    # Actually, for the objective function in SA, we need:
    # Score = (# 3-APs) - M * (# 4-APs)
    # Where M is large.
    # We need a fast way to count 4-APs or at least update the count incrementally.
    
    # Let's use the precomputed 4-APs.
    # For SA, we will maintain the state and the 4-AP count incrementally.
    # This function is for initial evaluation or verification.
    pass # Placeholder, we will use incremental updates in SA


def _initial_greedy(all_4aps, pos_to_4aps, seed=42):
    """
    Create a good initial solution using a fast greedy approach.
    This provides a warm start for the SA.
    """
    rng = random.Random(seed)
    mask = np.zeros(N, dtype=bool)
    blocked = np.zeros(N, dtype=np.int32)
    
    # We want to maximize 3-APs while avoiding 4-APs.
    # Greedy: add elements that give high 3-AP gain and don't create 4-APs.
    # To speed up, we can prioritize elements with high potential.
    
    # A simple heuristic: start with a random subset of low-density, or use a specific pattern.
    # Let's try a random shuffle and greedy add.
    remaining = list(range(N))
    rng.shuffle(remaining)
    
    for pos in remaining:
        if blocked[pos] > 0:
            continue
        # Check if adding pos creates a 4-AP
        # This is already handled by 'blocked' if we update it correctly.
        # But we need to update 'blocked' when we add an element.
        
        # Let's verify 'blocked' logic:
        # blocked[p] > 0 means there exists a 4-AP containing p where the other 3 are in the set.
        # If we add q, we must update blocked for all p in 4-APs containing q.
        
        mask[pos] = True
        # Update blocked for other elements in 4-APs containing pos
        for ap_idx in pos_to_4aps[pos]:
            ap = all_4aps[ap_idx]
            for other in ap:
                if other == pos:
                    continue
                # Check if the other two elements in this 4-AP (besides other and pos) are in mask
                # The 4-AP is (a, b, c, d). If we added pos, and 'other' is one of the others.
                # We need to check if the remaining two are in mask.
                # Let's identify the other two.
                others = [x for x in ap if x != other and x != pos]
                if len(others) == 2:
                    if mask[others[0]] and mask[others[1]]:
                        blocked[other] += 1
                        
    return mask


def _simulated_annealing(initial_mask, all_4aps, pos_to_4aps, start_time, deadline):
    """
    Simulated Annealing to improve the 3-AP count while keeping 4-AP count at 0.
    Objective: Maximize 3-APs.
    Constraint: 4-APs == 0.
    
    We use a penalty method: Score = 3-AP_count - P * 4-AP_count.
    Start with high P, anneal P down? Or keep P high enough to always avoid 4-APs?
    If we keep P very high, we effectively constrain 4-APs to 0.
    But if we get stuck, we might need to allow temporary 4-APs to escape?
    Given the strict constraint, it's safer to use a hard constraint check:
    Only accept moves that keep 4-AP count at 0, OR accept worse 4-AP states with probability exp(-delta/P).
    
    Let's use the penalty approach with a large constant P.
    P = 1000 (since max 3-APs is ~ few thousand, 1000 is large enough to prioritize 4-AP removal).
    
    Moves: Flip a random bit (add or remove).
    Delta Score = Delta 3-APs - P * Delta 4-APs.
    
    We need to efficiently calculate Delta 3-APs and Delta 4-APs.
    Delta 3-APs can be computed by _marginal_3ap_gain (if adding) or -_marginal_3ap_gain (if removing).
    Delta 4-APs:
    If we add pos:
      New 4-APs formed are those containing pos where the other 3 are already in the set.
      This is exactly blocked[pos] (if blocked was maintained correctly).
      Wait, blocked[pos] counts 4-APs where the *other* 3 are in the set.
      So if we add pos, we create blocked[pos] new 4-APs.
      So Delta 4-APs = blocked[pos].
      
    If we remove pos:
      We destroy 4-APs that contain pos and have the other 3 in the set.
      This is also blocked[pos] (before removal).
      So Delta 4-APs = -blocked[pos].
      
    So the penalty change is simple if we maintain 'blocked' correctly.
    
    However, 'blocked' changes when we add/remove elements.
    When we add pos:
      For each 4-AP containing pos, for each other element 'other' in that 4-AP:
        If the other two elements (besides pos and other) are in the set, then blocked[other] increases by 1.
        
    When we remove pos:
      For each 4-AP containing pos, for each other element 'other' in that 4-AP:
        If the other two elements (besides pos and other) are in the set, then blocked[other] decreases by 1.
        
    This update is O(#4APs containing pos) per move.
    """
    
    mask = initial_mask.copy()
    current_3ap = _count_3ap_fast(mask)
    
    # Initialize blocked
    blocked = np.zeros(N, dtype=np.int32)
    for p in np.where(mask)[0]:
        for ap_idx in pos_to_4aps[p]:
            ap = all_4aps[ap_idx]
            for other in ap:
                if other == p:
                    continue
                others = [x for x in ap if x != other and x != p]
                if len(others) == 2:
                    if mask[others[0]] and mask[others[1]]:
                        blocked[other] += 1
                        
    current_4ap = 0 # Should be 0 if initial mask is valid
    # Verify initial mask has no 4-APs
    # If not, we might need to fix it first, but _initial_greedy should ensure it.
    
    best_mask = mask.copy()
    best_score = current_3ap
    
    T = 10.0 # Initial temperature
    T_end = 0.1
    alpha = 0.9995 # Cooling rate
    
    # Estimate number of steps possible
    # Each step: O(1) for gain calc (vectorized), O(k) for blocked update.
    # k is small (~10-20).
    # Let's run until deadline.
    
    rng = random.Random(42)
    
    while time.time() < deadline - 2:
        if T < T_end:
            T = T_end
            
        # Pick a random position to flip
        pos = rng.randint(0, N - 1)
        
        if mask[pos]:
            # Remove pos
            # Delta 3-APs = - gain
            gain = _marginal_3ap_gain(mask, pos)
            delta_3ap = -gain
            
            # Delta 4-APs = - blocked[pos]
            delta_4ap = -blocked[pos]
            
            # Update blocked
            # Removing pos: for each 4-AP containing pos, update blocked for others
            for ap_idx in pos_to_4aps[pos]:
                ap = all_4aps[ap_idx]
                for other in ap:
                    if other == pos:
                        continue
                    others = [x for x in ap if x != other and x != pos]
                    if len(others) == 2:
                        if mask[others[0]] and mask[others[1]]:
                            blocked[other] -= 1
                            
            mask[pos] = False
            current_3ap += delta_3ap
            current_4ap += delta_4ap
            
        else:
            # Add pos
            gain = _marginal_3ap_gain(mask, pos)
            delta_3ap = gain
            
            # Delta 4-APs = blocked[pos]
            delta_4ap = blocked[pos]
            
            mask[pos] = True
            current_3ap += delta_3ap
            current_4ap += delta_4ap
            
            # Update blocked
            # Adding pos: for each 4-AP containing pos, update blocked for others
            for ap_idx in pos_to_4aps[pos]:
                ap = all_4aps[ap_idx]
                for other in ap:
                    if other == pos:
                        continue
                    others = [x for x in ap if x != other and x != pos]
                    if len(others) == 2:
                        if mask[others[0]] and mask[others[1]]:
                            blocked[other] += 1
            
        # Calculate objective change
        # We want to maximize 3-APs and minimize 4-APs.
        # Score = current_3ap - P * current_4ap
        # But we track delta.
        P = 10000.0 # Large penalty
        delta_obj = delta_3ap - P * delta_4ap
        
        # Acceptance criterion
        if delta_obj >= 0:
            # Always accept improvement
            pass
        else:
            # Accept with probability exp(delta_obj / T)
            # Note: delta_obj is negative
            prob = np.exp(delta_obj / T)
            if rng.random() > prob:
                # Reject move: Undo
                if mask[pos]:
                    # We added, so remove
                    mask[pos] = False
                    # Revert 3ap
                    current_3ap -= delta_3ap
                    current_4ap -= delta_4ap
                    # Revert blocked
                    for ap_idx in pos_to_4aps[pos]:
                        ap = all_4aps[ap_idx]
                        for other in ap:
                            if other == pos:
                                continue
                            others = [x for x in ap if x != other and x != pos]
                            if len(others) == 2:
                                if mask[others[0]] and mask[others[1]]:
                                    blocked[other] -= 1
                else:
                    # We removed, so add back
                    mask[pos] = True
                    # Revert 3ap
                    current_3ap -= delta_3ap
                    current_4ap -= delta_4ap
                    # Revert blocked
                    for ap_idx in pos_to_4aps[pos]:
                        ap = all_4aps[ap_idx]
                        for other in ap:
                            if other == pos:
                                continue
                            others = [x for x in ap if x != other and x != pos]
                            if len(others) == 2:
                                if mask[others[0]] and mask[others[1]]:
                                    blocked[other] += 1
                # Continue to next iteration without updating best
                T *= alpha
                continue
        
        # If we are here, the move was accepted (either improvement or accepted worse)
        # Check if it's a new best (must be valid, i.e., 4-APs == 0)
        if current_4ap == 0 and current_3ap > best_score:
            best_score = current_3ap
            best_mask = mask.copy()
            
        T *= alpha
        
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute 4-AP structure
    all_4aps, pos_to_4aps = _precompute_4ap_structure()
    
    best_mask = None
    best_score = -1
    
    # Multi-start SA
    # We have ~80 seconds.
    # Let's do a few restarts with different initial seeds.
    
    # First, generate a good initial solution
    # The greedy one is fast. Let's use it.
    
    for seed in range(5):
        if time.time() > deadline - 10:
            break
            
        init_mask = _initial_greedy(all_4aps, pos_to_4aps, seed=seed + 100)
        
        # Run SA
        sa_mask, sa_score = _simulated_annealing(init_mask, all_4aps, pos_to_4aps, start_time, deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask.copy()
            
    if best_mask is None:
        return [1]
        
    # Final verification and cleanup
    # Ensure no 4-APs
    # We trust the SA logic, but let's do a quick check if time permits.
    # Actually, the penalty P is large, so it's very likely 4-AP free.
    # If not, we might need a repair step.
    
    # Let's add a simple repair step if 4-APs exist
    final_mask = best_mask.copy()
    final_score = best_score
    
    # Check for 4-APs
    has_4ap = False
    for ap in all_4aps:
        if final_mask[ap[0]] and final_mask[ap[1]] and final_mask[ap[2]] and final_mask[ap[3]]:
            has_4ap = True
            break
            
    if has_4ap:
        # Remove elements to break 4-APs
        # Simple greedy removal
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for ap in all_4aps:
                if final_mask[ap[0]] and final_mask[ap[1]] and final_mask[ap[2]] and final_mask[ap[3]]:
                    # Remove one with lowest marginal gain
                    min_gain = float('inf')
                    min_pos = -1
                    for p in ap:
                        g = _marginal_3ap_gain(final_mask, p)
                        if g < min_gain:
                            min_gain = g
                            min_pos = p
                    final_mask[min_pos] = False
                    changed = True
                    break
        final_score = _count_3ap_fast(final_mask)
        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask.copy()
            
    result = sorted([int(i) + 1 for i in np.where(best_mask)[0]])
    
    if len(result) == 0:
        return [1]
        
    return result


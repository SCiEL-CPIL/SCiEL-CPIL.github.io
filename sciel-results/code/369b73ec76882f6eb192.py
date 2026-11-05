import time
import random
import numpy as np

N = 256
TIME_LIMIT = 55.0  # Safety margin from 60s

def _precompute():
    # All 3-APs and 4-APs as bitmasks
    m3 = []
    m4 = []
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            m3.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            m4.append(mask)
            
    # Incidence lists: which APs contain each bit
    bit_to_3aps = [[] for _ in range(N)]
    for i, m in enumerate(m3):
        for bit in range(N):
            if m & (1 << bit):
                bit_to_3aps[bit].append(i)
                
    bit_to_4aps = [[] for _ in range(N)]
    for i, m in enumerate(m4):
        for bit in range(N):
            if m & (1 << bit):
                bit_to_4aps[bit].append(i)
                
    return m3, m4, bit_to_3aps, bit_to_4aps

M3, M4, BIT_TO_3APS, BIT_TO_4APS = _precompute()

def _count_3aps(mask):
    count = 0
    for m in M3:
        if (mask & m) == m:
            count += 1
    return count

def _is_4ap_free(mask):
    for m in M4:
        if (mask & m) == m:
            return False
    return True

def _greedy_init():
    """Generate a 4-AP-free set using greedy method."""
    order = list(range(N))
    random.shuffle(order)
    mask = 0
    # Precompute 4-AP masks for fast check
    # We will check if adding a bit creates a 4-AP
    for idx in order:
        bit = 1 << idx
        new_mask = mask | bit
        valid = True
        # Check only 4-APs containing this bit
        for j in BIT_TO_4APS[idx]:
            if (new_mask & M4[j]) == M4[j]:
                valid = False
                break
        if valid:
            mask = new_mask
    return mask

def _local_search(mask, score, time_limit):
    """
    Hill Climbing with 1-bit and 2-bit moves.
    Returns (best_mask, best_score).
    """
    best_mask = mask
    best_score = score
    current_mask = mask
    current_score = score
    
    # State for incremental updates
    # is_3ap_present[j] = True if M3[j] is in current_mask
    # is_4ap_present[j] = True if M4[j] is in current_mask
    is_3ap_present = np.zeros(len(M3), dtype=bool)
    is_4ap_present = np.zeros(len(M4), dtype=bool)
    
    for j, m in enumerate(M3):
        if (current_mask & m) == m:
            is_3ap_present[j] = True
    for j, m in enumerate(M4):
        if (current_mask & m) == m:
            is_4ap_present[j] = True
            
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try 1-bit moves
        for bit in range(N):
            bit_val = 1 << bit
            if current_mask & bit_val:
                # Remove bit
                # Check if any 3-AP is lost
                delta = 0
                for j in BIT_TO_3APS[bit]:
                    if is_3ap_present[j]:
                        delta -= 1
                new_score = current_score + delta
                if new_score > current_score:
                    # Apply
                    current_mask &= ~bit_val
                    for j in BIT_TO_3APS[bit]:
                        if is_3ap_present[j]:
                            is_3ap_present[j] = False
                    for j in BIT_TO_4APS[bit]:
                        if is_4ap_present[j]:
                            is_4ap_present[j] = False
                    current_score = new_score
                    best_mask = current_mask
                    best_score = current_score
                    improved = True
                    no_improve_count = 0
                    break # Restart loop as state changed
            else:
                # Add bit
                # Check if creates 4-AP
                creates_4ap = False
                for j in BIT_TO_4APS[bit]:
                    if not is_4ap_present[j]:
                        others = M4[j] & ~bit_val
                        if (current_mask & others) == others:
                            creates_4ap = True
                            break
                if creates_4ap:
                    continue
                
                # Calculate delta
                delta = 0
                for j in BIT_TO_3APS[bit]:
                    if not is_3ap_present[j]:
                        others = M3[j] & ~bit_val
                        if (current_mask & others) == others:
                            delta += 1
                new_score = current_score + delta
                if new_score > current_score:
                    # Apply
                    current_mask |= bit_val
                    for j in BIT_TO_3APS[bit]:
                        if not is_3ap_present[j]:
                            others = M3[j] & ~bit_val
                            if (current_mask & others) == others:
                                is_3ap_present[j] = True
                    for j in BIT_TO_4APS[bit]:
                        if not is_4ap_present[j]:
                            others = M4[j] & ~bit_val
                            if (current_mask & others) == others:
                                is_4ap_present[j] = True
                    current_score = new_score
                    best_mask = current_mask
                    best_score = current_score
                    improved = True
                    no_improve_count = 0
                    break # Restart loop
        
        if improved:
            continue
            
        # Try 2-bit moves (Add 2, Remove 2)
        # This is more expensive, so we do it less frequently or only if stuck
        if no_improve_count > 10:
            # Random 2-bit move
            bits = random.sample(range(N), 2)
            b1, b2 = bits
            v1, v2 = 1 << b1, 1 << b2
            has1 = bool(current_mask & v1)
            has2 = bool(current_mask & v2)
            
            if has1 and has2:
                # Remove both
                # Check 3-APs lost
                delta = 0
                affected_3aps = set()
                for j in BIT_TO_3APS[b1]:
                    if is_3ap_present[j]:
                        affected_3aps.add(j)
                for j in BIT_TO_3APS[b2]:
                    if is_3ap_present[j]:
                        affected_3aps.add(j)
                for j in affected_3aps:
                    m = M3[j]
                    if (current_mask & m) == m:
                        delta -= 1
                new_score = current_score + delta
                if new_score > current_score:
                    current_mask &= ~(v1 | v2)
                    for j in affected_3aps:
                        is_3ap_present[j] = False
                    for j in BIT_TO_4APS[b1]:
                        if is_4ap_present[j]: is_4ap_present[j] = False
                    for j in BIT_TO_4APS[b2]:
                        if is_4ap_present[j]: is_4ap_present[j] = False
                    current_score = new_score
                    best_mask = current_mask
                    best_score = current_score
                    improved = True
                    no_improve_count = 0
            elif not has1 and not has2:
                # Add both
                # Check 4-APs created
                creates_4ap = False
                # Check 4-APs containing b1 or b2
                for j in BIT_TO_4APS[b1]:
                    if not is_4ap_present[j]:
                        others = M4[j] & ~(v1 | v2)
                        if (current_mask & others) == others:
                            creates_4ap = True
                            break
                if not creates_4ap:
                    for j in BIT_TO_4APS[b2]:
                        if not is_4ap_present[j]:
                            others = M4[j] & ~(v1 | v2)
                            if (current_mask & others) == others:
                                creates_4ap = True
                                break
                if creates_4ap:
                    continue
                
                delta = 0
                affected_3aps = set()
                for j in BIT_TO_3APS[b1]:
                    if not is_3ap_present[j]:
                        affected_3aps.add(j)
                for j in BIT_TO_3APS[b2]:
                    if not is_3ap_present[j]:
                        affected_3aps.add(j)
                for j in affected_3aps:
                    m = M3[j]
                    others = m & ~(v1 | v2)
                    if (current_mask & others) == others:
                        delta += 1
                new_score = current_score + delta
                if new_score > current_score:
                    current_mask |= (v1 | v2)
                    for j in affected_3aps:
                        m = M3[j]
                        others = m & ~(v1 | v2)
                        if (current_mask & others) == others:
                            is_3ap_present[j] = True
                    for j in BIT_TO_4APS[b1]:
                        if not is_4ap_present[j]:
                            others = M4[j] & ~(v1 | v2)
                            if (current_mask & others) == others:
                                is_4ap_present[j] = True
                    for j in BIT_TO_4APS[b2]:
                        if not is_4ap_present[j]:
                            others = M4[j] & ~(v1 | v2)
                            if (current_mask & others) == others:
                                is_4ap_present[j] = True
                    current_score = new_score
                    best_mask = current_mask
                    best_score = current_score
                    improved = True
                    no_improve_count = 0
            else:
                # Swap: remove one, add one
                if has1:
                    b_remove, b_add = b1, b2
                else:
                    b_remove, b_add = b2, b1
                v_remove = 1 << b_remove
                v_add = 1 << b_add
                
                # Check if adding b_add creates 4-AP (with b_remove still in mask? No, we remove it first)
                # We need to check 4-APs that contain b_add and are satisfied by (mask - b_remove + b_add)
                # This is complex. Skip swap for now to keep it simple and fast.
                pass
            
            if improved:
                continue
        
        no_improve_count += 1
        if no_improve_count > 50:
            break
            
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_mask = 0
    best_score = -1
    
    # Multi-start Local Search
    num_restarts = 20
    time_per_restart = (TIME_LIMIT - 2.0) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
            
        # Generate initial mask
        if i == 0:
            mask = _greedy_init()
        else:
            # Perturb the best found so far
            mask = best_mask
            # Randomly flip 5-10 bits
            for _ in range(random.randint(5, 10)):
                bit = random.randint(0, N-1)
                mask ^= (1 << bit)
            # Repair: remove bits that cause 4-APs
            # Simple repair: check all 4-APs, if present, remove one random bit from it
            changed = True
            while changed:
                changed = False
                for m in M4:
                    if (mask & m) == m:
                        # Remove a random bit from this 4-AP
                        bits_in_ap = [b for b in range(N) if m & (1 << b)]
                        b_to_remove = random.choice(bits_in_ap)
                        mask &= ~(1 << b_to_remove)
                        changed = True
                        break
        
        score = _count_3aps(mask)
        
        # Run local search
        ls_mask, ls_score = _local_search(mask, score, time_per_restart)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    print(f"Fitness: {_count_3aps(mask)}")
    print(f"Valid: {_is_4ap_free(mask)}")

import time
import random
import numpy as np

def _score_bitmask(mask):
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _set_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(mask):
    result = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            result.append(idx + 1)
        temp >>= 1
        idx += 1
    return result

def _get_set_bits(mask):
    bits = []
    temp = mask
    idx = 0
    while temp:
        if temp & 1:
            bits.append(idx)
        temp >>= 1
        idx += 1
    return bits

def _greedy_construct(rng):
    mask = 0
    for i in range(32):
        best_score = -1
        best_bit = 0
        # To add diversity, we can shuffle the candidate bits
        candidates = list(range(512))
        rng.shuffle(candidates)
        for bit in candidates:
            if mask & (1 << bit):
                continue
            new_mask = mask | (1 << bit)
            score = _score_bitmask(new_mask)
            if score > best_score:
                best_score = score
                best_bit = bit
        mask |= (1 << best_bit)
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    rng = random.Random(42)
    
    # 1. Greedy Construction
    best_mask = _greedy_construct(rng)
    best_score = _score_bitmask(best_mask)
    
    current_mask = best_mask
    current_score = best_score
    
    # 2. Simulated Annealing
    # We have ~140 seconds left.
    # Each step: 
    # - Generate move (O(1) or O(512) for greedy add)
    # - Score (O(255))
    # Let's estimate steps per second.
    # If we do 1000 steps/sec, we can do 140k steps.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # We will use a mix of random swaps and greedy swaps
    while time.time() < deadline and T > T_min:
        # Decide move type
        if rng.random() < 0.5:
            # Random Swap
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            add_idx = rng.randint(0, 511)
            if add_idx == remove_idx or (current_mask & (1 << add_idx)):
                continue
            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)
        else:
            # Greedy Swap: Remove random, Add best
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            base_mask = current_mask & ~(1 << remove_idx)
            
            best_add_score = -1
            best_add_idx = -1
            # Sample a subset of candidates to keep this fast?
            # Or scan all 512? 512 * 255 ops ~ 130k ops.
            # If we do this 100 times/sec, that's 13M ops/sec. Feasible in Python? Maybe slow.
            # Let's sample 50 candidates.
            candidates = rng.sample(range(512), 50)
            for bit in candidates:
                if base_mask & (1 << bit):
                    continue
                new_mask = base_mask | (1 << bit)
                score = _score_bitmask(new_mask)
                if score > best_add_score:
                    best_add_score = score
                    best_add_idx = bit
            
            if best_add_idx == -1:
                continue
                
            new_mask = base_mask | (1 << best_add_idx)

        new_score = _score_bitmask(new_mask)
        delta_score = new_score - current_score

        if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        T *= cooling

    # 3. Greedy Polish
    # Try all single swaps to improve
    improved = True
    polish_deadline = deadline
    while improved and time.time() < polish_deadline:
        improved = False
        bits_set = _get_set_bits(best_mask)
        rng.shuffle(bits_set)
        for remove_idx in bits_set:
            if time.time() > polish_deadline:
                break
            base_mask = best_mask & ~(1 << remove_idx)
            # Try all 512 adds
            for add_idx in range(512):
                if add_idx == remove_idx:
                    continue
                if best_mask & (1 << add_idx):
                    continue
                
                new_mask = base_mask | (1 << add_idx)
                new_score = _score_bitmask(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    break
            
            if improved:
                break

    result = _mask_to_set(best_mask)
    
    # Ensure exactly 32 elements
    if len(result) != 32:
        remaining = [v for v in range(1, 513) if v not in set(result)]
        result.extend(rng.sample(remaining, 32 - len(result)))
        result = result[:32]

    return sorted(result)

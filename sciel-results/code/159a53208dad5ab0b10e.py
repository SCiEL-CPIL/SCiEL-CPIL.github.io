import time
import random
import numpy as np

def _score_bitmask(mask):
    """
    Fast score using a 512-bit integer bitmask.
    mask: int with bit i set iff (i+1) is in the set (i ranges 0..511).
    Returns number of distinct d>0 such that a, a+d, a+2d all in set.
    """
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

def _get_missing_d(mask):
    """Return a list of d values that are NOT covered."""
    missing = []
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) == 0:
            missing.append(d)
    return missing

def _find_gaps_for_d(mask, d):
    """
    Find all y such that adding y would create a 3-AP with difference d.
    y is a gap if a, a+2d are in mask, but y=a+d is not.
    Returns list of indices (0-based) for y.
    """
    gaps = []
    for a in range(0, 512 - 2 * d):
        if (mask & (1 << a)) and (mask & (1 << (a + 2 * d))):
            y_idx = a + d
            if not (mask & (1 << y_idx)):
                gaps.append(y_idx)
    return gaps

def _density_init(rng, n=32, max_val=512):
    """
    Density-aware initialization: place elements in a somewhat dense cluster
    with some spread to capture multiple difference scales.
    """
    # Strategy: pick a base, then add elements with varying step sizes
    # to create multiple AP structures.
    elements = set()
    
    # Core dense block
    base = rng.randint(1, 50)
    for i in range(12):
        elements.add(base + i * 2)
    
    # Medium-density AP
    base2 = rng.randint(100, 200)
    for i in range(8):
        elements.add(base2 + i * 5)
    
    # Sparse spread elements
    while len(elements) < n:
        v = rng.randint(1, max_val)
        elements.add(v)
    
    return list(elements)[:n]

def _structured_init(rng, n=32, max_val=512):
    """
    Structured initialization combining multiple APs with different differences.
    """
    elements = set()
    
    # AP with d=1 (dense)
    start1 = rng.randint(1, 20)
    for i in range(10):
        elements.add(start1 + i)
    
    # AP with d=3
    start2 = rng.randint(50, 150)
    for i in range(8):
        elements.add(start2 + i * 3)
    
    # AP with d=7
    start3 = rng.randint(200, 350)
    for i in range(7):
        elements.add(start3 + i * 7)
    
    # Fill remaining with random
    while len(elements) < n:
        v = rng.randint(1, max_val)
        elements.add(v)
    
    return list(elements)[:n]

def _triple_swap_mask(mask, rng, bits_set):
    """
    Perform a triple swap: remove 3 elements, add 3 new ones.
    Returns new mask or None if infeasible.
    """
    if len(bits_set) < 3:
        return None
    
    # Pick 3 to remove
    remove_indices = rng.sample(bits_set, 3)
    new_mask = mask
    for idx in remove_indices:
        new_mask &= ~(1 << idx)
    
    # Pick 3 to add (not already in new_mask)
    add_indices = []
    attempts = 0
    while len(add_indices) < 3 and attempts < 100:
        idx = rng.randint(0, 511)
        if not (new_mask & (1 << idx)) and idx not in add_indices:
            add_indices.append(idx)
        attempts += 1
    
    if len(add_indices) < 3:
        return None
    
    for idx in add_indices:
        new_mask |= (1 << idx)
    
    return new_mask

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1

    # Initial diverse starts
    # 1. Random
    s = list(rng.sample(range(1, 513), 32))
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 2. Density-aware init
    s = _density_init(rng)
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 3. Structured init
    s = _structured_init(rng)
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 4. Two dense blocks
    block1 = list(range(1, 17))
    block2 = list(range(300, 316))
    s = block1 + block2
    if len(set(s)) == 32:
        m = _set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # 5. Interleaved APs
    s = set()
    for i in range(16):
        s.add(1 + i * 30)
    for i in range(16):
        s.add(15 + i * 30)
    if len(s) == 32:
        m = _set_to_mask(list(s))
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Multi-restart Simulated Annealing with single-swap and triple-swap moves
    num_restarts = 15
    restart_budget = 8.0  # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbation
        if restart == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            current_mask = best_mask
            # Perturb: remove 3-5 random elements, add new ones
            num_perturb = rng.randint(3, 5)
            for _ in range(num_perturb):
                bits_set = _get_set_bits(current_mask)
                if not bits_set: break
                remove_idx = rng.choice(bits_set)
                current_mask &= ~(1 << remove_idx)
                while True:
                    add_idx = rng.randint(0, 511)
                    if not (current_mask & (1 << add_idx)):
                        current_mask |= (1 << add_idx)
                        break
            current_score = _score_bitmask(current_mask)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        # SA Parameters
        T = 1.5
        T_min = 0.05
        cooling = 0.9995
        restart_deadline = start_time + (time.time() - start_time) + restart_budget
        if restart_deadline > deadline:
            restart_deadline = deadline

        while time.time() < restart_deadline and T > T_min:
            # Generate Move
            # 60% Directed single-swap, 20% triple-swap, 20% Random single-swap
            move_type = rng.random()
            
            if move_type < 0.6:
                # Directed single-swap
                missing_d = _get_missing_d(current_mask)
                if not missing_d:
                    break
                
                target_d = rng.choice(missing_d)
                gaps = _find_gaps_for_d(current_mask, target_d)
                
                if gaps:
                    add_idx = rng.choice(gaps)
                    bits_set = _get_set_bits(current_mask)
                    remove_idx = rng.choice(bits_set)
                else:
                    if len(missing_d) > 1:
                        target_d = rng.choice([d for d in missing_d if d != target_d])
                        gaps = _find_gaps_for_d(current_mask, target_d)
                        if gaps:
                            add_idx = rng.choice(gaps)
                            bits_set = _get_set_bits(current_mask)
                            remove_idx = rng.choice(bits_set)
                        else:
                            add_idx = rng.randint(0, 511)
                            bits_set = _get_set_bits(current_mask)
                            remove_idx = rng.choice(bits_set)
                    else:
                        add_idx = rng.randint(0, 511)
                        bits_set = _get_set_bits(current_mask)
                        remove_idx = rng.choice(bits_set)
            elif move_type < 0.8:
                # Triple swap
                bits_set = _get_set_bits(current_mask)
                new_mask = _triple_swap_mask(current_mask, rng, bits_set)
                if new_mask is None:
                    T *= cooling
                    continue
                # Apply triple swap directly
                current_mask = new_mask
                current_score = _score_bitmask(current_mask)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                T *= cooling
                continue
            else:
                # Random single-swap
                add_idx = rng.randint(0, 511)
                bits_set = _get_set_bits(current_mask)
                remove_idx = rng.choice(bits_set)

            if add_idx == remove_idx:
                T *= cooling
                continue
            if current_mask & (1 << add_idx):
                T *= cooling
                continue

            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)

            new_score = _score_bitmask(new_mask)
            delta_score = new_score - current_score

            # Acceptance
            if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

            T *= cooling

    # Greedy Polish with both single and triple swaps
    polish_deadline = deadline
    improved = True
    polish_rounds = 0
    max_polish_rounds = 50
    
    while improved and time.time() < polish_deadline and polish_rounds < max_polish_rounds:
        improved = False
        polish_rounds += 1
        bits_set = _get_set_bits(best_mask)
        rng.shuffle(bits_set)
        
        # Phase 1: Single-swap polish
        for remove_idx in bits_set:
            if time.time() > polish_deadline:
                break
            
            for add_idx in range(512):
                if add_idx == remove_idx:
                    continue
                if best_mask & (1 << add_idx):
                    continue
                
                new_mask = best_mask
                new_mask &= ~(1 << remove_idx)
                new_mask |= (1 << add_idx)
                
                new_score = _score_bitmask(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    break
            
            if improved:
                break
        
        # Phase 2: Triple-swap polish (if time permits)
        if not improved and time.time() < polish_deadline:
            bits_set = _get_set_bits(best_mask)
            for _ in range(200):  # Try 200 random triple swaps
                if time.time() > polish_deadline:
                    break
                new_mask = _triple_swap_mask(best_mask, rng, bits_set)
                if new_mask is None:
                    continue
                new_score = _score_bitmask(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    bits_set = _get_set_bits(best_mask)
                    break

    result = _mask_to_set(best_mask)
    
    # Ensure exactly 32 elements
    if len(result) != 32:
        remaining = [v for v in range(1, 513) if v not in set(result)]
        result.extend(rng.sample(remaining, 32 - len(result)))
        result = result[:32]

    return sorted(result)

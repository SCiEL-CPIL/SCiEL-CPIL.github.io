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

def _compute_coverage_and_gaps(mask):
    """
    For each d in 1..255, compute:
    - covered[d]: bool, whether d is covered
    - gap_count[d]: number of positions y where a, a+2d are in mask but y=a+d is not
    - gap_positions[d]: list of such y positions (0-based)
    
    Returns (covered_set, gap_counts, gap_positions)
    """
    covered = set()
    gap_counts = {}
    gap_positions = {}
    
    for d in range(1, 256):
        gaps = []
        for a in range(0, 512 - 2 * d):
            if (mask & (1 << a)) and (mask & (1 << (a + 2 * d))):
                y_idx = a + d
                if not (mask & (1 << y_idx)):
                    gaps.append(y_idx)
        
        gap_counts[d] = len(gaps)
        gap_positions[d] = gaps
        
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            covered.add(d)
    
    return covered, gap_counts, gap_positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s wall clock

    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1

    # Initial diverse starts
    # 1. Dense core [1..31] + outlier (based on cell findings)
    s = list(range(1, 32)) + [47]
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 2. Random
    s = list(rng.sample(range(1, 513), 32))
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 3. Structured: Two interleaved APs
    s = set()
    for i in range(16):
        s.add(1 + i * 32)
    for i in range(16):
        s.add(2 + i * 32)
    m = _set_to_mask(list(s))
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # 4. Dense block with variation
    s = list(range(1, 33))
    rng.shuffle(s)
    m = _set_to_mask(s)
    sc = _score_bitmask(m)
    if sc > best_score:
        best_score = sc
        best_mask = m

    # Multi-restart Simulated Annealing with SAT-inspired move generation
    num_restarts = 8
    restart_budget = 14.0  # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbation
        if restart == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            current_mask = best_mask
            # Perturb more aggressively
            num_perturb = 3 + restart
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

        # Compute initial coverage and gaps
        covered, gap_counts, gap_positions = _compute_coverage_and_gaps(current_mask)
        
        # SA Parameters
        T = 2.0
        T_min = 0.05
        cooling = 0.99995
        restart_deadline = min(start_time + (time.time() - start_time) + restart_budget, deadline)

        while time.time() < restart_deadline and T > T_min:
            # Generate Move using SAT-inspired strategy
            # Priority: target uncovered d's with fewest gaps (closest to being covered)
            
            move_made = False
            bits_set = _get_set_bits(current_mask)
            
            if not bits_set:
                break
                
            # 80% directed moves, 20% random
            if rng.random() < 0.8:
                # Find uncovered d's sorted by gap count (ascending)
                uncovered = [d for d in range(1, 256) if d not in covered and gap_counts[d] > 0]
                
                if uncovered:
                    # Sort by gap count, prefer d's with 1 gap (critical)
                    uncovered.sort(key=lambda d: gap_counts[d])
                    
                    # Pick from top candidates (weighted toward fewer gaps)
                    top_k = min(len(uncovered), 10)
                    target_d = rng.choice(uncovered[:top_k])
                    
                    gaps = gap_positions[target_d]
                    if gaps:
                        add_idx = rng.choice(gaps)
                        # Choose remove_idx: prefer elements that don't contribute to many covered d's
                        # For now, random selection
                        remove_idx = rng.choice(bits_set)
                        
                        if add_idx != remove_idx and not (current_mask & (1 << add_idx)):
                            move_made = True
                else:
                    # All d with gaps are covered, try random
                    add_idx = rng.randint(0, 511)
                    remove_idx = rng.choice(bits_set)
                    if add_idx != remove_idx and not (current_mask & (1 << add_idx)):
                        move_made = True
            else:
                add_idx = rng.randint(0, 511)
                remove_idx = rng.choice(bits_set)
                if add_idx != remove_idx and not (current_mask & (1 << add_idx)):
                    move_made = True
            
            if not move_made:
                T *= cooling
                continue

            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)

            new_score = _score_bitmask(new_mask)
            delta_score = new_score - current_score

            if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
                current_mask = new_mask
                current_score = new_score
                # Update coverage and gaps
                covered, gap_counts, gap_positions = _compute_coverage_and_gaps(current_mask)
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

            T *= cooling

    # Greedy Polish with SAT-inspired targeting
    improved = True
    polish_deadline = deadline
    while improved and time.time() < polish_deadline:
        improved = False
        covered, gap_counts, gap_positions = _compute_coverage_and_gaps(best_mask)
        bits_set = _get_set_bits(best_mask)
        rng.shuffle(bits_set)
        
        for remove_idx in bits_set:
            if time.time() > polish_deadline:
                break
            
            # Try to find an add_idx that covers a missing d
            uncovered = [d for d in range(1, 256) if d not in covered and gap_counts[d] > 0]
            uncovered.sort(key=lambda d: gap_counts[d])
            
            # First try targeted moves
            for target_d in uncovered[:20]:
                gaps = gap_positions[target_d]
                for add_idx in gaps:
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
            
            if improved:
                break # Restart the outer loop with new best_mask
            
            # If no targeted move worked, try some random swaps
            for _ in range(5):
                add_idx = rng.randint(0, 511)
                if add_idx == remove_idx or (best_mask & (1 << add_idx)):
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

    result = _mask_to_set(best_mask)
    
    # Ensure exactly 32 elements
    if len(result) != 32:
        remaining = [v for v in range(1, 513) if v not in set(result)]
        result.extend(rng.sample(remaining, 32 - len(result)))
        result = result[:32]

    return sorted(result)

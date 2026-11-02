import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    best_size = total
    best_set = None
    
    while time.time() < deadline - 5.0:
        # 1. Greedy Initialization
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0 and time.time() < deadline - 5.0:
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0: break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        if uncovered_count > 0:
            continue
            
        # 2. Local Search Polish (Removals)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            min_counts = np.min(cover_count[neighbors], axis=1)
            for i in range(len(selected) - 1, -1, -1):
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    # Update uncovered_count?
                    # Removing a codeword might uncover words.
                    # But we only remove if min_count > 1, so no word becomes uncovered.
                    # So uncovered_count remains 0.
                    cover_count[neighbors[cw]] -= 1
        
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_set = selected.copy()
            
        # 3. Simulated Annealing
        T = 50.0
        cooling_rate = 0.95
        penalty = 100.0
        
        # Current Energy
        # Uncovered words are 0 after polish
        current_energy = current_size + penalty * 0.0
        
        for step in range(1000): # More steps per restart
            if time.time() > deadline - 5.0:
                break
                
            move_type = np.random.randint(3)
            
            if move_type == 0 and len(selected) > 0:
                idx = np.random.randint(len(selected))
                cw_out = selected.pop(idx)
                
                # Update cover_count and uncovered_count
                ball = neighbors[cw_out]
                # Words that were covered only by this codeword will become uncovered
                # cover_count[ball] will decrease by 1
                # If any cover_count[ball] was 1, it becomes 0.
                # Count how many were 1.
                num_becoming_uncovered = np.sum(cover_count[ball] == 1)
                
                cover_count[ball] -= 1
                
                new_uncovered_count = uncovered_count + num_becoming_uncovered
                new_size = len(selected)
                new_energy = new_size + penalty * new_uncovered_count
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                    uncovered_count = new_uncovered_count
                    # Update best if valid and smaller
                    if new_uncovered_count == 0 and new_size < best_size:
                        best_size = new_size
                        best_set = selected.copy()
                else:
                    # Revert
                    selected.insert(idx, cw_out)
                    cover_count[ball] += 1
                    
            elif move_type == 1:
                cw_in = np.random.randint(total)
                if cw_in not in selected:
                    selected.append(cw_in)
                    
                    ball = neighbors[cw_in]
                    # Words that were uncovered will become covered
                    # cover_count[ball] will increase by 1
                    # If any cover_count[ball] was 0, it becomes 1.
                    num_becoming_covered = np.sum(cover_count[ball] == 0)
                    
                    cover_count[ball] += 1
                    
                    new_uncovered_count = uncovered_count - num_becoming_covered
                    new_size = len(selected)
                    new_energy = new_size + penalty * new_uncovered_count
                    
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                        current_energy = new_energy
                        uncovered_count = new_uncovered_count
                        if new_uncovered_count == 0 and new_size < best_size:
                            best_size = new_size
                            best_set = selected.copy()
                    else:
                        # Revert
                        selected.pop()
                        cover_count[ball] -= 1
                else:
                    pass
                    
            else:
                if len(selected) == 0:
                    continue
                idx = np.random.randint(len(selected))
                cw_out = selected.pop(idx)
                ball_out = neighbors[cw_out]
                
                # Calculate changes for removal
                num_becoming_uncovered_out = np.sum(cover_count[ball_out] == 1)
                cover_count[ball_out] -= 1
                
                cw_in = np.random.randint(total)
                if cw_in == cw_out:
                    cover_count[ball_out] += 1
                    selected.insert(idx, cw_out)
                    continue
                    
                if cw_in in selected:
                    cover_count[ball_out] += 1
                    selected.insert(idx, cw_out)
                    continue
                    
                ball_in = neighbors[cw_in]
                # Calculate changes for addition
                # Note: cover_count has already been updated for removal
                # So we check the current state
                num_becoming_covered_in = np.sum(cover_count[ball_in] == 0)
                
                cover_count[ball_in] += 1
                selected.append(cw_in)
                
                new_uncovered_count = uncovered_count + num_becoming_uncovered_out - num_becoming_covered_in
                new_size = len(selected)
                new_energy = new_size + penalty * new_uncovered_count
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                    uncovered_count = new_uncovered_count
                    if new_uncovered_count == 0 and new_size < best_size:
                        best_size = new_size
                        best_set = selected.copy()
                else:
                    # Revert
                    selected.pop()
                    cover_count[ball_in] -= 1
                    selected.insert(idx, cw_out)
                    cover_count[ball_out] += 1
            
            T *= cooling_rate

        # Final Polish
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            min_counts = np.min(cover_count[neighbors], axis=1)
            for i in range(len(selected) - 1, -1, -1):
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    # Uncovered count doesn't change as we only remove redundant
                    # But wait, if we removed a codeword, some words might have count 1 -> 0?
                    # No, min_count > 1 means all words in its ball have count >= 2.
                    # So decreasing by 1 leaves them >= 1. No new uncovered words.
        
        if uncovered_count == 0 and len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

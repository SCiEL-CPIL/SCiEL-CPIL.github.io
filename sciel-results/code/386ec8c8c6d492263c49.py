import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """Greedy initialization: each row takes the smallest next mark with no difference collision."""
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_bitset(diff_set, max_diff):
    """Build a bitset representation of a difference set."""
    num_words = (max_diff + 64) // 64 + 1
    bits = [0] * num_words
    for d in diff_set:
        if d <= max_diff:
            idx = d - 1
            bits[idx // 64] |= (1 << (idx % 64))
    return bits

def _bitset_to_set(bits):
    """Convert bitset back to a set."""
    result = set()
    for i, word in enumerate(bits):
        while word:
            bit = (word & -word).bit_length() - 1
            d = i * 64 + bit + 1
            result.add(d)
            word &= word - 1
    return result

def _has_conflict_bitset(new_diffs, old_diffs, used_bits):
    """Check if new_diffs conflicts with (used_bits - old_diffs)."""
    for d in new_diffs:
        if d in old_diffs:
            continue
        idx = d - 1
        if idx >= 0:
            word_idx = idx // 64
            bit_idx = idx % 64
            if word_idx < len(used_bits) and (used_bits[word_idx] & (1 << bit_idx)):
                return True
    return False

def _randomized_greedy_row(forbidden_bits, max_limit, k, max_candidates=8):
    """
    Construct a valid row using randomized greedy against forbidden diffs.
    forbidden_bits: bitset of diffs that cannot be used
    max_limit: maximum allowed value for the last entry (or None for no limit)
    Returns a valid row or None if construction fails.
    """
    row = [0]
    used_in_row_bits = [0] * 4  # Small bitset for in-row diffs (max ~200)
    
    for pos in range(k):
        candidates = []
        x = row[-1] + 1
        
        # Look ahead to find valid candidates
        while len(candidates) < max_candidates:
            if max_limit is not None and x >= max_limit:
                break
            
            # Check if x - a for all a in row are valid
            valid = True
            new_diffs_in_row = set()
            for a in row:
                d = x - a
                if d in new_diffs_in_row:
                    valid = False
                    break
                new_diffs_in_row.add(d)
                # Check forbidden
                idx = d - 1
                if idx >= 0:
                    wi = idx // 64
                    bi = idx % 64
                    if wi < len(forbidden_bits) and (forbidden_bits[wi] & (1 << bi)):
                        valid = False
                        break
                # Check used_in_row
                if wi < len(used_in_row_bits) and (used_in_row_bits[wi] & (1 << bi)):
                    valid = False
                    break
            
            if valid:
                candidates.append((x, new_diffs_in_row))
            
            x += 1
            if x - row[-1] > 300:  # Safety break
                break
        
        if not candidates:
            return None
        
        # Pick randomly, bias toward smaller values
        # Weight by inverse of value for diversity
        weights = [1.0 / (c[0] + 1) for c in candidates]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                chosen_idx = i
                break
        
        chosen_x, chosen_diffs = candidates[chosen_idx]
        row.append(chosen_x)
        
        # Update used_in_row_bits
        for d in chosen_diffs:
            idx = d - 1
            if idx >= 0:
                wi = idx // 64
                bi = idx % 64
                if wi < len(used_in_row_bits):
                    used_in_row_bits[wi] |= (1 << bi)
    
    return row

def _try_replace_row(rows, ri, forbidden_bits, max_limit, k):
    """
    Try to replace row ri with a new valid row.
    Returns (new_row, new_diffs) or (None, None) if failed.
    """
    new_row = _randomized_greedy_row(forbidden_bits, max_limit, k, max_candidates=10)
    if new_row is None:
        return None, None
    new_diffs = _row_diffs_set(new_row)
    return new_row, new_diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain state
    current_rows = [r[:] for r in rows]
    current_val = best_val
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Bitset for fast conflict checking
    max_diff = best_val + 100
    used_bits = _build_bitset(all_used, max_diff)
    
    # 2. Main search: Mix row-replacement and single-position moves
    T = 50.0
    T_min = 0.5
    cool = 0.99997
    no_improve = 0
    max_no_improve = 50000
    
    # Track which rows are "hot" (recently changed) for diversity
    row_age = [0] * n_rows
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Age all rows
        for i in range(n_rows):
            row_age[i] += 1
        
        # Choose move type: 70% row replacement, 30% single position
        if random.random() < 0.70:
            # --- Row replacement move ---
            ri = random.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs[ri]
            
            # Build forbidden bits: all diffs except those in row ri
            forbidden_set = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden_set |= row_diffs[j]
            
            forbidden_bits = _build_bitset(forbidden_set, max_diff)
            
            # Try to find a new row with max <= current_val (or slightly less)
            target_max = current_val
            if random.random() < 0.3:
                target_max = current_val - 1  # Try to reduce
            
            new_row, new_diffs = _try_replace_row(
                current_rows, ri, forbidden_bits, target_max, k)
            
            if new_row is None:
                # Try with no limit (exploration)
                new_row, new_diffs = _try_replace_row(
                    current_rows, ri, forbidden_bits, None, k)
            
            if new_row is not None:
                new_row_max = new_row[-1]
                
                # Compute new global max
                if ri == k or new_row_max > current_rows[ri][k]:
                    cand_row_max = new_row_max
                else:
                    cand_row_max = current_rows[ri][k]
                
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.3:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    all_used -= old_diffs
                    all_used |= new_diffs
                    
                    # Update bitset
                    for d in old_diffs:
                        if d <= max_diff:
                            idx = d - 1
                            used_bits[idx // 64] &= ~(1 << (idx % 64))
                    for d in new_diffs:
                        if d <= max_diff:
                            idx = d - 1
                            used_bits[idx // 64] |= (1 << (idx % 64))
                    
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    row_age[ri] = 0
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # --- Single position move (fine-tuning) ---
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Try a few candidates
            found_valid = False
            for _ in range(5):
                new_val = random.randint(min_allowed, max_allowed)
                if new_val == old_val:
                    continue
                
                # Compute new diffs for this row
                old_row = current_rows[ri]
                old_diffs = row_diffs[ri]
                
                # Temporarily replace and compute
                old_row[pi] = new_val
                new_diffs = _row_diffs_set(old_row)
                old_row[pi] = old_val  # restore
                
                # Check internal validity (all diffs distinct)
                if len(new_diffs) != len(old_diffs):
                    continue
                
                # Check conflicts with other rows
                if _has_conflict_bitset(new_diffs, old_diffs, used_bits):
                    continue
                
                # Valid move - compute new global max
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = current_rows[ri][k]
                
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    all_used -= old_diffs
                    all_used |= new_diffs
                    
                    for d in old_diffs:
                        if d <= max_diff:
                            idx = d - 1
                            used_bits[idx // 64] &= ~(1 << (idx % 64))
                    for d in new_diffs:
                        if d <= max_diff:
                            idx = d - 1
                            used_bits[idx // 64] |= (1 << (idx % 64))
                    
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    row_age[ri] = 0
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    found_valid = True
                    break
            
            if not found_valid:
                no_improve += 1
        
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0
            row_age = [0] * n_rows
            max_diff = best_val + 100
            used_bits = _build_bitset(all_used, max_diff)
    
    # 3. Final Polishing: Try to reduce scope
    current_best = best_val
    for target_val in range(current_best - 1, max(100, current_best - 8), -1):
        if time.time() > deadline - 5:
            break
        
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        max_diff = target_val + 50
        used_bits = _build_bitset(all_used, max_diff)
        
        T = 30.0
        T_min = 0.1
        cool = 0.9999
        no_improve = 0
        max_steps = 10000
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            new_val = random.randint(min_allowed, max_allowed)
            if new_val == old_val:
                continue
            
            old_row = current_rows[ri]
            old_diffs = row_diffs[ri]
            
            old_row[pi] = new_val
            new_diffs = _row_diffs_set(old_row)
            old_row[pi] = old_val
            
            if len(new_diffs) != len(old_diffs):
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            if _has_conflict_bitset(new_diffs, old_diffs, used_bits):
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            if new_global_max > target_val:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if new_global_max <= target_val:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                
                for d in old_diffs:
                    if d <= max_diff:
                        idx = d - 1
                        used_bits[idx // 64] &= ~(1 << (idx % 64))
                for d in new_diffs:
                    if d <= max_diff:
                        idx = d - 1
                        used_bits[idx // 64] |= (1 << (idx % 64))
                
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            if no_improve > 2000:
                break
    
    return best_rows

# EVOLVE-BLOCK-END

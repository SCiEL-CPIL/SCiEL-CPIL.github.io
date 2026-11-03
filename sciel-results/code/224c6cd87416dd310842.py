import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # For the main case: 3^7, radius 1
    if q == 3 and n == 7 and radius == 1:
        return _cover_ternary_7()
    
    # Fallback for other cases
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]
    
    # Bitset-based approach for small spaces
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    balls = [0] * total_words
    
    if radius == 1:
        # Precompute index deltas for each position and value change
        # For a word w, neighbor at position pos with value new_val
        # index = sum(w[k] * q^(n-1-k) for k in range(n))
        # Changing w[pos] to new_val changes index by (new_val - w[pos]) * q^(n-1-pos)
        
        q_powers = [q ** (n - 1 - pos) for pos in range(n)]
        
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = i + (new_val - orig) * q_powers[pos]
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
    
    all_ones_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 110.0
    
    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = all_ones_mask
        selected = []
        
        # Greedy Phase
        while uncovered:
            best_idx = -1
            max_cover = 0
            
            for i in range(total_words):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Removal Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                        
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
                    
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_code = [all_words[i] for i in selected]
            
    return best_code


def _cover_ternary_7():
    """
    Specialized solver for K_3(7,1) using greedy + local search with bitsets.
    """
    total = 2187
    start_time = time.time()
    time_limit = 110.0
    
    # Precompute balls
    q = 3
    n = 7
    q_powers = [q ** (n - 1 - pos) for pos in range(n)]
    
    balls = [0] * total
    for i in range(total):
        w = []
        tmp = i
        for k in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    neighbor_idx = i + (new_val - orig) * q_powers[pos]
                    mask |= (1 << neighbor_idx)
        balls[i] = mask
    
    all_ones_mask = (1 << total) - 1
    
    # Precompute for each word, the list of words at distance 1 (neighbors)
    # This helps with local moves
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        w = []
        tmp = i
        for k in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    neighbor_idx = i + (new_val - orig) * q_powers[pos]
                    neighbors[i].append(neighbor_idx)
    
    best_code = None
    best_size = total + 1
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        uncovered = all_ones_mask
        selected_set = set()
        selected_list = []
        
        # Greedy Phase
        while uncovered:
            best_idx = -1
            max_cover = 0
            
            for i in range(total):
                if i in selected_set:
                    continue
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if random.random() < 0.3:
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            selected_set.add(best_idx)
            selected_list.append(best_idx)
            uncovered &= ~balls[best_idx]
        
        # Local Search / Improvement Phase
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Try to remove a codeword
            removed = False
            for i in range(len(selected_list) - 1, -1, -1):
                if (time.time() - start_time) > time_limit:
                    break
                union_rest = 0
                for j in range(len(selected_list)):
                    if j != i:
                        union_rest |= balls[selected_list[j]]
                if (union_rest & all_ones_mask) == all_ones_mask:
                    removed_idx = selected_list.pop(i)
                    selected_set.remove(removed_idx)
                    removed = True
                    break
            
            if not removed:
                # Try to swap: remove one, add one that covers more uncovered
                # Actually, if we can't remove, try to find a better set
                # Let's try: for each codeword, see if we can replace it with 
                # another that covers at least as much and allows removal of another
                improved = False
                for i in range(len(selected_list)):
                    if (time.time() - start_time) > time_limit:
                        break
                    
                    # What does this codeword uniquely cover?
                    unique_to_i = 0
                    for j in range(len(selected_list)):
                        if j != i:
                            unique_to_i |= balls[selected_list[j]]
                    unique_to_i = balls[selected_list[i]] & ~unique_to_i
                    unique_count = unique_to_i.bit_count()
                    
                    if unique_count == 0:
                        # Can remove, but we already tried that
                        continue
                    
                    # Try to find a replacement
                    # Look for a word that covers all of unique_to_i
                    best_replacement = -1
                    best_gain = -1
                    for k in range(total):
                        if k in selected_set:
                            continue
                        # Does k cover all unique_to_i?
                        if (balls[k] & unique_to_i) == unique_to_i:
                            # What would be the new unique count?
                            # This is complex, let's just try it
                            # Tentatively swap
                            old_idx = selected_list[i]
                            selected_list[i] = k
                            selected_set.remove(old_idx)
                            selected_set.add(k)
                            
                            # Check if we can now remove something
                            can_remove = False
                            for j in range(len(selected_list) - 1, -1, -1):
                                union_rest = 0
                                for m in range(len(selected_list)):
                                    if m != j:
                                        union_rest |= balls[selected_list[m]]
                                if (union_rest & all_ones_mask) == all_ones_mask:
                                    can_remove = True
                                    break
                            
                            if can_remove:
                                improved = True
                                break
                            else:
                                # Revert
                                selected_list[i] = old_idx
                                selected_set.remove(k)
                                selected_set.add(old_idx)
                    
                    if improved:
                        break
            
            if not improved and not removed:
                # Try random swap
                if len(selected_list) > 200:
                    i = random.randint(0, len(selected_list) - 1)
                    k = random.randint(0, total - 1)
                    if k not in selected_set:
                        old_idx = selected_list[i]
                        selected_list[i] = k
                        selected_set.remove(old_idx)
                        selected_set.add(k)
                        
                        # Check coverage
                        union = 0
                        for j in selected_list:
                            union |= balls[j]
                        if (union & all_ones_mask) != all_ones_mask:
                            # Revert
                            selected_list[i] = old_idx
                            selected_set.remove(k)
                            selected_set.add(old_idx)
        
        current_size = len(selected_list)
        if current_size < best_size:
            best_size = current_size
            # Convert to codewords
            best_code = []
            for idx in selected_list:
                w = []
                tmp = idx
                for k in range(n):
                    w.append(tmp % q)
                    tmp //= q
                w.reverse()
                best_code.append(w)
        
        if best_size <= 170:
            break
    
    if best_code is None:
        # Fallback: greedy only
        uncovered = all_ones_mask
        selected_list = []
        while uncovered:
            best_idx = -1
            max_cover = 0
            for i in range(total):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected_list.append(best_idx)
            uncovered &= ~balls[best_idx]
        
        best_code = []
        for idx in selected_list:
            w = []
            tmp = idx
            for k in range(n):
                w.append(tmp % q)
                tmp //= q
            w.reverse()
            best_code.append(w)
    
    return best_code

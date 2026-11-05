import time
import random
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    total_words = q ** n
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
    
    # Precompute balls as lists of neighbor indices
    balls = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neighbors = [i]
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    neighbors.append(i + delta)
        else:
            import itertools
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            delta += (variation[k] - d[pos]) * q_powers[pos]
                        if delta != 0:
                            neighbors.append(i + delta)
        balls[i] = set(neighbors)
    
    # Also precompute ball masks for fast coverage checking
    ball_masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in balls[i]:
            mask |= (1 << j)
        ball_masks[i] = mask
    
    all_mask = (1 << total_words) - 1
    
    # Greedy construction with multiple random restarts
    best_code = None
    best_size = total_words + 1
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = all_mask
        code = []
        while uncovered:
            if time.time() > deadline - 10:
                return None
            best_idx = -1
            max_gain = -1
            best_gain_list = []
            for i in range(total_words):
                if i in code:
                    continue
                gain = (ball_masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
                    best_gain_list = [i]
                elif gain == max_gain:
                    best_gain_list.append(i)
            if best_idx == -1 or max_gain == 0:
                break
            # Random tie-breaking
            chosen = rng.choice(best_gain_list)
            code.append(chosen)
            uncovered &= ~ball_masks[chosen]
        if uncovered:
            # Add remaining uncovered words
            remaining = uncovered
            while remaining:
                lb = remaining & (-remaining)
                idx = lb.bit_length() - 1
                remaining ^= lb
                code.append(idx)
        return code
    
    # Try several greedy restarts with different seeds
    for seed in range(20):
        if time.time() > deadline - 100:
            break
        code = greedy_cover(seed)
        if code is None:
            break
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    if best_code is None:
        # Fallback: just return all words (shouldn't happen)
        return [list(digits_of[i]) for i in range(total_words)]
    
    # Local search: add-swap-remove to minimize code size
    code_set = set(best_code)
    cover_mask = 0
    for idx in best_code:
        cover_mask |= ball_masks[idx]
    
    # Count how many codewords cover each word
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        for j in balls[idx]:
            coverage_count[j] += 1
    
    current_size = len(best_code)
    no_improve_count = 0
    
    while time.time() < deadline - 10:
        improved = False
        
        # Phase 1: Try to remove words
        # A word can be removed if it's the only codeword covering some words? No.
        # A word c can be removed if every word in ball(c) is covered by at least 2 codewords.
        removable_candidates = []
        for c in best_code:
            can_remove = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removable_candidates.append(c)
        
        if removable_candidates:
            # Try removing each candidate
            rng = random.Random()
            rng.shuffle(removable_candidates)
            for c in removable_candidates:
                if time.time() > deadline - 10:
                    break
                # Check if removing c keeps full coverage
                ok = True
                for j in balls[c]:
                    if coverage_count[j] <= 1:
                        ok = False
                        break
                if ok:
                    best_code.remove(c)
                    code_set.discard(c)
                    for j in balls[c]:
                        coverage_count[j] -= 1
                    cover_mask ^= ball_masks[c]  # Actually need to recompute
                    # Recompute cover mask properly
                    cover_mask = 0
                    for idx in best_code:
                        cover_mask |= ball_masks[idx]
                    current_size -= 1
                    improved = True
                    no_improve_count = 0
                    break
        
        # Phase 2: Add-swap-remove
        # Try adding a word, then removing two words, for net -1
        if not improved and current_size > 403:  # Don't go below known lower bound
            # Find words that are "weakly covered" (covered by few codewords)
            # Strategy: pick a word w not in code, add it, then try to remove two words
            # that become redundant.
            
            # For efficiency, try a limited number of swaps per iteration
            rng = random.Random()
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts and time.time() < deadline - 10:
                attempts += 1
                # Pick a random word to add
                add_candidate = rng.randint(0, total_words - 1)
                if add_candidate in code_set:
                    continue
                
                # Add it
                best_code.append(add_candidate)
                code_set.add(add_candidate)
                for j in balls[add_candidate]:
                    coverage_count[j] += 1
                
                new_size = current_size + 1
                
                # Now try to remove two words
                removed_any = False
                for c1 in best_code[:]:
                    if c1 == add_candidate:
                        continue
                    # Check if c1 is removable
                    removable1 = True
                    for j in balls[c1]:
                        if coverage_count[j] <= 1:
                            removable1 = False
                            break
                    if not removable1:
                        continue
                    
                    # Tentatively remove c1
                    for j in balls[c1]:
                        coverage_count[j] -= 1
                    
                    # Now try to remove c2
                    for c2 in best_code[:]:
                        if c2 == add_candidate or c2 == c1:
                            continue
                        removable2 = True
                        for j in balls[c2]:
                            if coverage_count[j] <= 1:
                                removable2 = False
                                break
                        if not removable2:
                            continue
                        
                        # Remove both
                        best_code.remove(c1)
                        best_code.remove(c2)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        for j in balls[c2]:
                            coverage_count[j] -= 1
                        
                        new_size -= 2
                        improved = True
                        removed_any = True
                        break
                    
                    if removed_any:
                        break
                    
                    # Restore c1
                    for j in balls[c1]:
                        coverage_count[j] += 1
                    best_code.append(c1)
                    code_set.add(c1)
                
                if not removed_any:
                    # Revert the addition
                    best_code.remove(add_candidate)
                    code_set.discard(add_candidate)
                    for j in balls[add_candidate]:
                        coverage_count[j] -= 1
                    new_size = current_size
                else:
                    current_size = new_size
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                # Try a different initial solution
                for seed in range(20, 40):
                    if time.time() > deadline - 10:
                        break
                    code = greedy_cover(seed)
                    if code is not None and len(code) < current_size:
                        best_code = code[:]
                        code_set = set(best_code)
                        coverage_count = np.zeros(total_words, dtype=np.int32)
                        for idx in best_code:
                            for j in balls[idx]:
                                coverage_count[j] += 1
                        current_size = len(best_code)
                        no_improve_count = 0
                        break
    
    # Final verification and cleanup
    # Ensure full coverage
    cover_mask = 0
    for idx in best_code:
        cover_mask |= ball_masks[idx]
    
    if cover_mask != all_mask:
        # Add missing words
        uncovered = all_mask & ~cover_mask
        while uncovered:
            lb = uncovered & (-uncovered)
            idx = lb.bit_length() - 1
            uncovered ^= lb
            if idx not in code_set:
                best_code.append(idx)
                code_set.add(idx)
    
    # Remove duplicates
    best_code = list(set(best_code))
    
    return [list(digits_of[i]) for i in best_code]

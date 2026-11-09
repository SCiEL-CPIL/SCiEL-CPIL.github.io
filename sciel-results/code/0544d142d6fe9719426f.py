"""Rewritten program: greedy construction of A_2(18,6) with deadline guard."""
import time
import random

N = 18
MIN_D = 6
FULL_MASK = (1 << N) - 1

def _reed_muller_extended():
    """Build RM(1,4) extended to 18 bits (32 words, distance 8)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # positions 16 and 17 are zero (extension)
            words.append(word)
    return words

def _check_valid(code, candidate):
    """Check if candidate has Hamming distance >= MIN_D from all words in code."""
    for w in code:
        if (w ^ candidate).bit_count() < MIN_D:
            return False
    return True

def _greedy_build(seed_words, deadline, rng):
    """
    Greedy add candidates from random permutations until deadline.
    Returns the best code found.
    """
    code = list(seed_words)
    code_set = set(code)
    total_candidates = 1 << N
    
    # Pre-compute a list of all candidates once (or iterate over range)
    # We'll use random permutation in chunks to avoid memory blowup
    # Actually 2^18 = 262144 integers, that's fine to store.
    
    all_candidates = list(range(total_candidates))
    
    while time.time() < deadline:
        # Shuffle the remaining candidates (not yet in code)
        # Instead of shuffling the full list each time, we can just
        # iterate through a shuffled version and skip those already in code
        rng.shuffle(all_candidates)
        
        added_this_pass = 0
        for cand in all_candidates:
            if time.time() >= deadline:
                break
            if cand in code_set:
                continue
            if _check_valid(code, cand):
                code.append(cand)
                code_set.add(cand)
                added_this_pass += 1
                # Don't break, keep trying to add more
        # If no new words added in a full pass, we might be stuck
        # Try a few more passes with different shuffles
        if added_this_pass == 0:
            # Try up to 3 more random shuffles before giving up
            for _ in range(3):
                if time.time() >= deadline:
                    break
                rng.shuffle(all_candidates)
                added = 0
                for cand in all_candidates:
                    if time.time() >= deadline:
                        break
                    if cand in code_set:
                        continue
                    if _check_valid(code, cand):
                        code.append(cand)
                        code_set.add(cand)
                        added += 1
                if added > 0:
                    break
            else:
                # No improvement after multiple shuffles, break out
                # Actually, if truly no more can be added, break
                # But it's unlikely we've exhausted the space, so let's try one more approach
                # Let's just break; the code is maximal under this greedy strategy
                break
    
    return code

def _local_improvement(code, deadline, rng):
    """
    Try to improve the code by removing one word and trying to add up to two new ones.
    This is a simple 1-for-2 local search.
    """
    if len(code) < 2:
        return code
    
    code_list = list(code)
    code_set = set(code)
    total_candidates = 1 << N
    all_candidates = list(range(total_candidates))
    
    best_code = code_list
    best_size = len(code_list)
    
    # Try removing each word and seeing if we can add more
    # This is expensive, so we limit the number of trials
    indices = list(range(len(code_list)))
    rng.shuffle(indices)
    
    trials_done = 0
    max_trials = 200  # Limit to avoid timeout
    
    for idx in indices:
        if time.time() >= deadline or trials_done >= max_trials:
            break
        trials_done += 1
        
        # Remove word at idx
        removed = code_list[idx]
        temp_code = code_list[:idx] + code_list[idx+1:]
        temp_set = code_set - {removed}
        
        # Try to add new words
        rng.shuffle(all_candidates)
        additions = []
        for cand in all_candidates:
            if time.time() >= deadline:
                break
            if len(additions) >= 2:
                break
            if cand in temp_set or cand == removed:
                continue
            if _check_valid(temp_code + additions, cand):
                additions.append(cand)
        
        # If we added more than we removed, accept
        if len(additions) > 1:
            new_code = temp_code + additions
            if len(new_code) > best_size:
                best_code = new_code
                best_size = len(new_code)
        
        # Even if we didn't improve, the temp exploration might help
        # But for simplicity, we only accept strict improvements
    
    return best_code

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    rng = random.Random(42)
    
    # Start with RM(1,4) extended
    initial_words = _reed_muller_extended()
    
    # Phase 1: Greedy addition
    code = _greedy_build(initial_words, deadline, rng)
    
    # Phase 2: Local improvement if time remains
    if time.time() < deadline - 5.0 and len(code) > 32:
        code = _local_improvement(code, deadline, rng)
    
    # Final check: ensure all pairs have distance >= MIN_D
    # (Should be guaranteed by construction, but let's be safe)
    final_code = list(set(code))
    for i in range(len(final_code)):
        for j in range(i + 1, len(final_code)):
            if (final_code[i] ^ final_code[j]).bit_count() < MIN_D:
                # This shouldn't happen, but if it does, remove the problematic one
                final_code.pop(j)
                j -= 1
                break
    
    return final_code

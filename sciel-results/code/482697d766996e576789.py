# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Generate all words as integers
    total_words = q ** n
    q_powers = [q ** i for i in range(n)]
    
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Generate all possible words
    all_words = list(range(total_words))
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # neighbors[i] is a set of ints for O(1) lookup
    neighbors = [set() for _ in range(total_words)]
    for i in all_words:
        w = int_to_word(i)
        neighbors[i].add(i) # distance 0
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    neighbors[i].add(word_to_int(w))
                w[pos] = orig

    # 2. Initialize with Direct Sum Code (81 words for n=6, q=3)
    # Tetracode for first 4 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    if n >= 4:
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                init_code.append(word_to_int(h + list(tail)))
    else:
        # Fallback for small n
        init_code = all_words[:]

    # Ensure distinct and valid
    init_code = list(set(init_code))
    
    # 3. Setup State
    in_code = np.zeros(total_words, dtype=bool)
    coverage_count = np.zeros(total_words, dtype=np.int32)
    code_set = set(init_code)
    
    for c in init_code:
        in_code[c] = True
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    def try_remove():
        """Try to remove any codeword that doesn't break coverage. Returns True if removed."""
        removed_any = False
        candidates = [c for c in code_set if in_code[c]]
        random.shuffle(candidates)
        
        for c in candidates:
            if not in_code[c]:
                continue
            # Check if c is essential (covers any word uniquely)
            essential = False
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    essential = True
                    break
            if not essential:
                # Safe to remove
                in_code[c] = False
                code_set.remove(c)
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                removed_any = True
        return removed_any

    def try_swap_1_for_1():
        """Try to swap a codeword to enable further reduction."""
        if not code_set:
            return False
        
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c_out in candidates:
            if not in_code[c_out]:
                continue
            
            # Identify words uniquely covered by c_out
            unique_words = []
            for nb in neighbors[c_out]:
                if coverage_count[nb] == 1:
                    unique_words.append(nb)
            
            if not unique_words:
                continue
                
            # Find a candidate c_in that covers all unique_words
            all_candidates = [w for w in all_words if not in_code[w]]
            random.shuffle(all_candidates)
            
            for c_in in all_candidates:
                valid = True
                for u in unique_words:
                    if u not in neighbors[c_in]:
                        valid = False
                        break
                if valid:
                    # Perform Swap
                    in_code[c_out] = False
                    code_set.remove(c_out)
                    for nb in neighbors[c_out]:
                        coverage_count[nb] -= 1
                    
                    in_code[c_in] = True
                    code_set.add(c_in)
                    for nb in neighbors[c_in]:
                        coverage_count[nb] += 1
                    return True
        return False

    def try_swap_2_for_1():
        """Try to replace 2 codewords with 1. This allows reducing code size by 1."""
        if len(code_set) < 2:
            return False
        
        # Pick two random codewords from the set
        candidates = list(code_set)
        random.shuffle(candidates)
        
        # Limit pairs to check to avoid O(N^2) explosion in each step
        # We try a limited number of pairs
        max_pairs = 50
        for i in range(min(max_pairs, len(candidates))):
            for j in range(i + 1, min(max_pairs, len(candidates))):
                c1 = candidates[i]
                c2 = candidates[j]
                
                if not in_code[c1] or not in_code[c2]:
                    continue
                
                # Identify words uniquely covered by c1 or c2
                # A word is "critical" if it is covered by c1 or c2 and NOT by any other codeword
                # Actually, we need the set of words that MUST be covered by the new codeword.
                # These are words that are currently covered ONLY by c1 or ONLY by c2.
                # If a word is covered by c1 and c2, it's fine if the new word doesn't cover it?
                # No, if we remove c1 and c2, the word must be covered by c_new.
                # So c_new must cover:
                # 1. Words covered by c1 and not by any other codeword (including c2).
                # 2. Words covered by c2 and not by any other codeword (including c1).
                # 3. Words covered by BOTH c1 and c2 but by NO other codeword.
                
                # Efficient check:
                # Union of neighbors of c1 and c2.
                # For each u in union, check if coverage_count[u] == (1 if u in neighbors[c1] else 0) + (1 if u in neighbors[c2] else 0)
                # If yes, u is uniquely covered by {c1, c2}.
                
                unique_to_pair = set()
                # Iterate neighbors of c1
                for u in neighbors[c1]:
                    if coverage_count[u] == 1:
                        # Covered only by c1?
                        if u in neighbors[c1] and u not in neighbors[c2]:
                            unique_to_pair.add(u)
                    elif coverage_count[u] == 2:
                        # Covered by 2 codewords. Are they c1 and c2?
                        if u in neighbors[c1] and u in neighbors[c2]:
                            # Check if any other codeword covers u
                            # This is expensive. 
                            # Alternative: 
                            # If coverage_count[u] == 2, and u is in N(c1) and N(c2), 
                            # it implies ONLY c1 and c2 cover u.
                            # Because if a third one covered it, count would be >= 3.
                            # If only c1 covered it, count would be 1.
                            # So yes, if count is 2 and u is in both, they are the only coverers.
                            unique_to_pair.add(u)
                
                for u in neighbors[c2]:
                    if coverage_count[u] == 1:
                        if u in neighbors[c2] and u not in neighbors[c1]:
                            unique_to_pair.add(u)
                    elif coverage_count[u] == 2:
                        if u in neighbors[c2] and u in neighbors[c1]:
                            unique_to_pair.add(u)
                
                if not unique_to_pair:
                    continue
                
                # Find a candidate c_in that covers all unique_to_pair
                all_candidates = [w for w in all_words if not in_code[w] and w != c1 and w != c2]
                random.shuffle(all_candidates)
                
                # Optimization: Check only a subset of candidates if time is tight?
                # For now, check all. 729 * (size of unique_to_pair) is small.
                for c_in in all_candidates:
                    valid = True
                    for u in unique_to_pair:
                        if u not in neighbors[c_in]:
                            valid = False
                            break
                    if valid:
                        # Perform Swap: Remove c1, c2; Add c_in
                        in_code[c1] = False
                        code_set.remove(c1)
                        for nb in neighbors[c1]:
                            coverage_count[nb] -= 1
                        
                        in_code[c2] = False
                        code_set.remove(c2)
                        for nb in neighbors[c2]:
                            coverage_count[nb] -= 1
                        
                        in_code[c_in] = True
                        code_set.add(c_in)
                        for nb in neighbors[c_in]:
                            coverage_count[nb] += 1
                        return True
        return False

    # Main Loop
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        while try_remove():
            pass
        
        # Phase 2: Try Swaps to escape local minima or reduce size
        improved = False
        
        # Try 1-for-1 swaps (to enable removals)
        for _ in range(20):
            if try_swap_1_for_1():
                improved = True
            else:
                break
        
        # Try 2-for-1 swaps (direct size reduction)
        for _ in range(20):
            if try_swap_2_for_1():
                improved = True
            else:
                break
        
        if not improved:
            # If no swap worked, we are in a local minimum.
            # Try a random perturbation? Or just break?
            # Given the tight time limit, breaking is safer.
            # But we might try one more round of removals after a failed swap attempt?
            # try_remove was already run.
            break

    # Final Code
    final_code = [int_to_word(c) for c in code_set]
    
    return final_code
# EVOLVE-BLOCK-END

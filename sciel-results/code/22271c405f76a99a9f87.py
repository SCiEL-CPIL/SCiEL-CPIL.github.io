# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin for 150s limit
    
    total = q ** n
    # Each codeword covers 1 + n*(q-1) points
    cover_size = 1 + n * (q - 1)
    
    # Precompute all points as integers 0..total-1 (base-q representation)
    # Point i in base-q: digits[i]
    
    # For greedy set cover, we need to efficiently find which points a codeword covers
    # and track coverage.
    
    # Represent each point as an integer. A codeword c covers points at Hamming distance <= radius.
    # For radius=1, q=3, n=8: covers c itself + for each position, the other 2 values.
    
    # Precompute: for each position i and each value v, the set of points that differ from 
    # a given point only at position i with value v.
    # Actually, let's think differently.
    
    # For a codeword c (as integer), the covered points are:
    # c itself, and for each position j, c with position j changed to each other value.
    
    # Let's precompute the "neighbors" structure.
    # For position j, changing digit j from d to d' changes the integer by (d' - d) * q^j
    
    pow_q = [1] * n
    for i in range(1, n):
        pow_q[i] = pow_q[i-1] * q
    
    # For a codeword represented as digits d[0..n-1], 
    # the covered points (as integers) are:
    # sum(d[i]*q^i) for the codeword itself
    # plus for each position j and each alt value v != d[j]:
    #   sum(d[i]*q^i) - d[j]*q^j + v*q^j
    
    # Greedy set cover approach:
    # - Maintain a boolean array "covered" of size total
    # - At each step, find the codeword that maximizes new coverage
    # - This is expensive if done naively. We need a smarter approach.
    
    # Alternative: use a priority-based approach.
    # For each point, track how many codewords cover it.
    # Initially all 0. When we add a codeword, increment counts for all points it covers.
    # A point is "covered" if count > 0.
    # 
    # To find the best next codeword: for each candidate codeword, count how many of its 
    # covered points have count == 0 (i.e., newly covered).
    # 
    # This is still O(total * cover_size) per iteration which is too slow.
    
    # Better approach: randomized greedy with random restarts.
    # Pick random codewords, keep the ones that add new coverage.
    
    # Actually, let's use a different strategy:
    # 1. Generate a large random pool of candidate codewords
    # 2. Greedily select from the pool based on new coverage
    # 3. Then peel (remove redundant codewords)
    
    # For efficiency, let's work with numpy arrays.
    
    # Convert integer to digits
    def int_to_digits(x, n, q):
        digits = np.zeros(n, dtype=np.int32)
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    # Convert digits to integer
    def digits_to_int(digits):
        result = 0
        for i in range(n):
            result += digits[i] * pow_q[i]
        return result
    
    # Generate all possible "neighbor offsets" for radius 1
    # For each position j and each alt value delta (nonzero), the offset is delta * q^j
    # But we need to be careful: changing digit j from d to v means adding (v-d)*q^j
    # where v != d.
    
    # Precompute for each position j, the list of (delta, v) pairs where v != d for all d
    # Actually, for a given digit value d at position j, the alt values are all v in {0..q-1} \ {d}
    # The offset is (v - d) * q^j
    
    # For greedy, let's use a simpler approach:
    # Randomly generate codewords, use a coverage array, and do greedy selection.
    
    # Let's try: generate K random codewords, then greedily select a subset.
    
    # Actually, for n=8, q=3, the space is 6561. Let's try a more direct approach.
    
    # Strategy: 
    # 1. Use a structured initial code (like the tetracode product) as a starting point
    # 2. Randomly perturb/replace some codewords
    # 3. Greedy add/remove to optimize
    
    # Let's start with the tetracode product (729 words) and try to reduce it.
    
    # Tetracode: 9 words of length 4
    tetracode = []
    for a in range(q):
        for b in range(q):
            tetracode.append([a, b, (a + b) % q, (a + 2 * b) % q])
    
    # Product code: tetracode × all words of length n-4
    # That gives 9 * 3^4 = 729 words for n=8
    
    # Convert to integer representation
    def word_to_int(word):
        result = 0
        for i in range(n):
            result += word[i] * pow_q[i]
        return result
    
    def int_to_word(x):
        word = []
        for i in range(n):
            word.append(x % q)
            x //= q
        return word
    
    # Generate the product code
    code_words = []
    for t in tetracode:
        for tail in np.ndindex(*([q] * (n - 4))):
            full_word = list(t) + list(tail)
            code_words.append(full_word)
    
    code_ints = [word_to_int(w) for w in code_words]
    
    # Coverage array
    covered = np.zeros(total, dtype=np.uint8)
    
    # For each codeword, mark all points it covers
    def mark_covered(codeword_int, covered_arr):
        digits = int_to_digits(codeword_int, n, q)
        # Mark the codeword itself
        covered_arr[codeword_int] = 1
        # Mark all neighbors at distance 1
        for j in range(n):
            for v in range(q):
                if v != digits[j]:
                    neighbor = codeword_int + (v - digits[j]) * pow_q[j]
                    covered_arr[neighbor] = 1
    
    # Initialize coverage with the product code
    for ci in code_ints:
        mark_covered(ci, covered)
    
    # Check if fully covered
    if covered.all():
        pass  # Good, we have a valid covering
    
    # Now try to reduce: greedily remove codewords that don't uncover any point
    # For each codeword, check if all points it covers are also covered by other codewords.
    
    # Build a coverage count array
    cover_count = np.zeros(total, dtype=np.int16)
    for ci in code_ints:
        digits = int_to_digits(ci, n, q)
        cover_count[ci] += 1
        for j in range(n):
            for v in range(q):
                if v != digits[j]:
                    neighbor = ci + (v - digits[j]) * pow_q[j]
                    cover_count[neighbor] += 1
    
    # Peeling: iteratively remove codewords where all their covered points have count > 1
    current_code = list(code_ints)
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for idx in range(len(current_code) - 1, -1, -1):
            ci = current_code[idx]
            digits = int_to_digits(ci, n, q)
            # Check if all points covered by ci have count > 1
            can_remove = True
            # Check ci itself
            if cover_count[ci] <= 1:
                can_remove = False
            else:
                for j in range(n):
                    for v in range(q):
                        if v != digits[j]:
                            neighbor = ci + (v - digits[j]) * pow_q[j]
                            if cover_count[neighbor] <= 1:
                                can_remove = False
                                break
                    if not can_remove:
                        break
            if can_remove:
                # Remove this codeword
                cover_count[ci] -= 1
                for j in range(n):
                    for v in range(q):
                        if v != digits[j]:
                            neighbor = ci + (v - digits[j]) * pow_q[j]
                            cover_count[neighbor] -= 1
                current_code.pop(idx)
                changed = True
    
    # Convert back to word format
    result = [int_to_word(ci) for ci in current_code]
    
    # Try further optimization with random swaps
    if time.time() < deadline - 20:
        # Try to replace pairs of codewords with better ones
        best_code = list(current_code)
        best_count = len(best_code)
        
        # Random restarts with different initial codes
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            
            # Start from a random perturbation of the current best
            trial_code = list(best_code)
            trial_count = np.zeros(total, dtype=np.int16)
            for ci in trial_code:
                digits = int_to_digits(ci, n, q)
                trial_count[ci] += 1
                for j in range(n):
                    for v in range(q):
                        if v != digits[j]:
                            neighbor = ci + (v - digits[j]) * pow_q[j]
                            trial_count[neighbor] += 1
            
            # Try to remove more codewords
            t_changed = True
            while t_changed and time.time() < deadline - 15:
                t_changed = False
                for idx in range(len(trial_code) - 1, -1, -1):
                    ci = trial_code[idx]
                    digits = int_to_digits(ci, n, q)
                    can_remove = True
                    if trial_count[ci] <= 1:
                        can_remove = False
                    else:
                        for j in range(n):
                            for v in range(q):
                                if v != digits[j]:
                                    neighbor = ci + (v - digits[j]) * pow_q[j]
                                    if trial_count[neighbor] <= 1:
                                        can_remove = False
                                        break
                            if not can_remove:
                                break
                    if can_remove:
                        trial_count[ci] -= 1
                        for j in range(n):
                            for v in range(q):
                                if v != digits[j]:
                                    neighbor = ci + (v - digits[j]) * pow_q[j]
                                    trial_count[neighbor] -= 1
                        trial_code.pop(idx)
                        t_changed = True
            
            if len(trial_code) < best_count:
                best_count = len(trial_code)
                best_code = list(trial_code)
    
    result = [int_to_word(ci) for ci in best_code]
    
    # Verify the result is valid
    final_covered = np.zeros(total, dtype=np.uint8)
    for w in result:
        ci = word_to_int(w)
        digits = int_to_digits(ci, n, q)
        final_covered[ci] = 1
        for j in range(n):
            for v in range(q):
                if v != digits[j]:
                    neighbor = ci + (v - digits[j]) * pow_q[j]
                    final_covered[neighbor] = 1
    
    if not final_covered.all():
        # Fallback to the product code
        return code_words
    
    return result

# EVOLVE-BLOCK-END

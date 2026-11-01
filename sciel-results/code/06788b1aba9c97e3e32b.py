import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation of 1..n.
    
    Uses the standard recursive (Graham-Lagarias) construction as a starting point.
    For n=7, this yields a word of length 8720.
    Then applies a multi-stage optimization:
    1. Fast local deletion search (symbol-level).
    2. Permutation-ordering optimization (structural) for the final step.
    3. Further local deletion search.
    """
    if n <= 1:
        return "1"
    
    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Stage 1: Fast local deletion search
    word = _local_search(word, n, time_budget=30)
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Stage 2: Permutation ordering optimization for the final step
    # This is only meaningful for n >= 4 where the permutation order matters
    if n >= 4:
        word = _optimize_perm_order(word, n, time_budget=40)
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Stage 3: Final local deletion search
    remaining_time = max(0, deadline - time.time())
    if remaining_time > 5:
        word = _local_search(word, n, time_budget=remaining_time - 5)
    
    return "".join(map(str, word))

def _local_search(word, n, time_budget=30):
    """Try to shorten the word while maintaining superpermutation property."""
    start_time = time.time()
    word = list(word)
    
    target = math.factorial(n)
    
    def count_perms(w):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)
    
    current_count = count_perms(word)
    if current_count < target:
        return word  # Not valid, return as-is
    
    # Greedy deletion: repeatedly try to delete symbols
    # Use a more systematic approach: try positions in order of "redundancy"
    
    for outer_iter in range(500):
        if time.time() - start_time > time_budget:
            break
        
        if len(word) <= n:
            break
        
        found_deletion = False
        
        # Try random positions first (faster to find easy deletions)
        for attempt in range(200):
            if time.time() - start_time > time_budget:
                break
            
            pos = random.randint(1, len(word) - 2)
            new_word = word[:pos] + word[pos+1:]
            
            if count_perms(new_word) >= target:
                word = new_word
                found_deletion = True
                break
        
        if not found_deletion:
            # Try sequential scan from the middle outward
            mid = len(word) // 2
            for offset in range(1, len(word) // 2):
                if time.time() - start_time > time_budget:
                    break
                
                for pos in [mid - offset, mid + offset]:
                    if pos < 1 or pos > len(word) - 2:
                        continue
                    new_word = word[:pos] + word[pos+1:]
                    if count_perms(new_word) >= target:
                        word = new_word
                        found_deletion = True
                        break
                if found_deletion:
                    break
        
        if not found_deletion:
            # If we can't find any deletion, try a few more random attempts
            for attempt in range(100):
                if time.time() - start_time > time_budget:
                    break
                pos = random.randint(1, len(word) - 2)
                new_word = word[:pos] + word[pos+1:]
                if count_perms(new_word) >= target:
                    word = new_word
                    found_deletion = True
                    break
            
            if not found_deletion:
                # Give up if we've tried hard enough
                if outer_iter > 50:
                    break
    
    return word

def _optimize_perm_order(word, n, time_budget=40):
    """
    Optimize the ordering of permutations in the final step of the construction.
    
    The standard construction builds a word for n symbols by taking the word for n-1 symbols
    and for each permutation P of 1..n-1 (in order of appearance), appending P + [n] + P
    with maximum overlap. The order of these permutations affects the total length.
    
    This function extracts the permutations from the current word, computes the overlap
    matrix, and uses local search to find a better ordering.
    """
    start_time = time.time()
    
    # Extract the permutations of 1..n-1 from the word
    # In the standard construction, the word for n symbols is built by interleaving
    # blocks of the form P + [n] + P where P is a permutation of 1..n-1.
    
    # Find all occurrences of n in the word
    n_positions = [i for i, x in enumerate(word) if x == n]
    
    if len(n_positions) == 0:
        return word
    
    # Each occurrence of n should be in the middle of a block P + [n] + P
    # The permutation P is the (n-1) symbols before and after n
    # But due to overlaps, we need to be more careful
    
    # Let's extract the permutations by looking at the structure
    # In the standard construction, the word is a concatenation of blocks
    # Each block is P_i + [n] + P_i with overlaps
    
    # A simpler approach: extract all length-(n-1) windows that are permutations
    # and that appear adjacent to n
    
    perms = []
    seen = set()
    
    for pos in n_positions:
        # Check if there are n-1 symbols before and after
        if pos >= n - 1 and pos + n - 1 <= len(word):
            before = tuple(word[pos-(n-1):pos])
            after = tuple(word[pos+1:pos+n])
            
            # In the standard construction, before == after
            if before == after and len(set(before)) == n - 1:
                if before not in seen:
                    seen.add(before)
                    perms.append(before)
    
    if len(perms) < math.factorial(n - 1):
        # Fallback: just return the original word
        return word
    
    # Now we have the list of permutations in their current order
    # We want to find a better order
    
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # k can be 0 to n-2
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Check from max_k down to 0
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap_matrix[i][j] = found
    
    # Initial order: the order in which they appear in the current word
    # We need to map from permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # The current order is the order in which they appear
    # But we extracted them in order of appearance of n, so that's already the order
    order = list(range(num_perms))
    
    def calc_length(ord_list):
        # Length = (n-1) + 1 + (n-1) for first block = 2*(n-1) + 1
        # Plus for each subsequent block: 2*(n-1) + 1 - overlap
        total = 2 * (n - 1) + 1
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
        return total
    
    best_length = calc_length(order)
    best_order = order[:]
    
    # Local search: try swaps and reversals
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try adjacent swaps
        for i in range(len(order) - 1):
            if time.time() - start_time > time_budget:
                break
            
            # Swap i and i+1
            order[i], order[i+1] = order[i+1], order[i]
            new_len = calc_length(order)
            if new_len < best_length:
                best_length = new_len
                best_order = order[:]
                improved = True
            else:
                order[i], order[i+1] = order[i+1], order[i]
        
        # Try 2-opt: reverse a segment
        if time.time() - start_time > time_budget:
            break
        
        for i in range(len(order) - 1):
            if time.time() - start_time > time_budget:
                break
            
            for j in range(i + 1, len(order)):
                if time.time() - start_time > time_budget:
                    break
                
                # Reverse segment [i, j]
                segment = order[i:j+1]
                order[i:j+1] = segment[::-1]
                
                new_len = calc_length(order)
                if new_len < best_length:
                    best_length = new_len
                    best_order = order[:]
                    improved = True
                else:
                    order[i:j+1] = segment
    
    # Reconstruct the word with the new order
    new_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        
        # Find maximum overlap with current new_word
        overlap = 0
        max_ov = min(len(new_word), len(block))
        for o in range(max_ov, -1, -1):
            if o == 0:
                overlap = 0
                break
            if new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        new_word += block[overlap:]
    
    # Validate: check that all permutations are present
    target = math.factorial(n)
    seen_perms = set()
    for i in range(len(new_word) - n + 1):
        window = tuple(new_word[i:i+n])
        if len(set(window)) == n:
            seen_perms.add(window)
    
    if len(seen_perms) >= target:
        return new_word
    else:
        return word  # Fallback to original

# EVOLVE-BLOCK-END

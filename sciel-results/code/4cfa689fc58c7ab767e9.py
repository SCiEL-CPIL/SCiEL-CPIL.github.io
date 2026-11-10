"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187
    
    # Precompute neighbor lists using NumPy for speed
    # Each word has 1 + n*(q-1) = 1 + 7*2 = 15 neighbors
    all_neighbors = np.zeros((total_words, 15), dtype=np.int32)
    
    # Precompute digit decomposition for all words
    digits_table = np.zeros((total_words, n), dtype=np.int32)
    for w_int in range(total_words):
        temp = w_int
        for i in range(n):
            digits_table[w_int, n-1-i] = temp % q
            temp //= q
    
    # Build neighbor table
    for w_int in range(total_words):
        all_neighbors[w_int, 0] = w_int  # self
        idx = 1
        for i in range(n):
            orig = digits_table[w_int, i]
            for new_val in range(q):
                if new_val != orig:
                    # Compute the new integer value
                    # digits_table[w_int] with position i changed to new_val
                    val = 0
                    for j in range(n):
                        d = new_val if j == i else digits_table[w_int, j]
                        val = val * q + d
                    all_neighbors[w_int, idx] = val
                    idx += 1
    
    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    # Phase 1: Greedy set cover
    covered = np.zeros(total_words, dtype=np.bool_)
    code_ints = []
    code_set = set()
    remaining = total_words
    
    while remaining > 0 and time.time() < deadline - 35:
        best_gain = -1
        best_word = -1
        
        # Scan all words to find the one with max uncovered coverage
        for c in range(total_words):
            if c in code_set:
                continue
            gain = np.sum(~covered[all_neighbors[c]])
            if gain > best_gain:
                best_gain = gain
                best_word = c
        
        if best_gain <= 0:
            break
        
        # Add best_word to code
        code_ints.append(best_word)
        code_set.add(best_word)
        newly_covered = ~covered[all_neighbors[best_word]]
        remaining -= np.sum(newly_covered)
        covered[all_neighbors[best_word]] = True
    
    # Phase 2: Remove redundant codewords
    covered_count = np.zeros(total_words, dtype=np.int32)
    for c_int in code_ints:
        covered_count[all_neighbors[c_int]] += 1
    
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            if np.min(covered_count[all_neighbors[c_int]]) > 1:
                covered_count[all_neighbors[c_int]] -= 1
                code_ints.pop(idx)
                code_set.discard(c_int)
                changed = True
                break  # Restart scan since indices changed
    
    # Phase 3: Swap-based local search with targeted candidates
    # For each codeword, precompute which other codewords could potentially replace it
    # A replacement must cover all words that the codeword uniquely covers
    
    for iteration in range(20000):
        if time.time() > deadline - 10:
            break
        
        if len(code_ints) < 2:
            break
        
        # Pick a random codeword to try removing
        idx = random.randint(0, len(code_ints) - 1)
        c_int = code_ints[idx]
        c_neighbors = all_neighbors[c_int]
        
        # Check if we can remove it (all its neighbors are covered by others)
        if np.min(covered_count[c_neighbors]) <= 1:
            continue
        
        # Temporarily remove
        covered_count[c_neighbors] -= 1
        
        # Find newly uncovered words
        uncovered = np.where(covered_count == 0)[0]
        
        if len(uncovered) == 0:
            # No uncovered words, so we can just remove it
            code_ints.pop(idx)
            code_set.discard(c_int)
            continue
        
        # Try to find a single word that covers all uncovered words
        # Sample candidate positions from the uncovered words themselves
        # and their neighbors
        best_gain = -1
        best_pos = None
        
        # Build candidate set: uncovered words + their neighbors
        candidate_set = set()
        for u in uncovered:
            candidate_set.add(u)
            for nb in all_neighbors[u]:
                candidate_set.add(nb)
        
        # Also add some random candidates
        for _ in range(20):
            candidate_set.add(random.randint(0, total_words - 1))
        
        for cand in candidate_set:
            if cand in code_set:
                continue
            
            # Check if this candidate covers all uncovered words
            cand_neighbors = all_neighbors[cand]
            covers_all = True
            for u in uncovered:
                if u not in cand_neighbors:
                    covers_all = False
                    break
            
            if covers_all:
                # Check if adding this improves coverage (covers more than just the uncovered)
                gain = np.sum(covered_count[cand_neighbors] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_pos = cand
        
        if best_pos is not None:
            # Add the new codeword
            covered_count[all_neighbors[best_pos]] += 1
            code_ints.append(best_pos)
            code_set.add(best_pos)
            # Remove the old one
            code_ints.pop(idx)
            code_set.discard(c_int)
        else:
            # Roll back
            covered_count[c_neighbors] += 1
    
    # Phase 4: Final polish - try more aggressive swaps
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
        
        if len(code_ints) < 2:
            break
        
        # Pick two random codewords to try removing
        i1 = random.randint(0, len(code_ints) - 1)
        i2 = random.randint(0, len(code_ints) - 1)
        if i1 == i2:
            continue
        
        c1 = code_ints[i1]
        c2 = code_ints[i2]
        
        # Temporarily remove both
        covered_count[all_neighbors[c1]] -= 1
        covered_count[all_neighbors[c2]] -= 1
        
        # Find newly uncovered words
        uncovered = np.where(covered_count == 0)[0]
        
        if len(uncovered) == 0:
            # Can remove one of them
            code_ints.pop(i2 if i2 > i1 else i1)
            code_ints.pop(i1)
            code_set.discard(c1)
            code_set.discard(c2)
            continue
        
        # Try to find a single word that covers all uncovered words
        best_gain = -1
        best_pos = None
        
        # Build candidate set
        candidate_set = set()
        for u in uncovered:
            candidate_set.add(u)
            for nb in all_neighbors[u]:
                candidate_set.add(nb)
        
        for _ in range(10):
            candidate_set.add(random.randint(0, total_words - 1))
        
        for cand in candidate_set:
            if cand in code_set:
                continue
            
            cand_neighbors = all_neighbors[cand]
            covers_all = True
            for u in uncovered:
                if u not in cand_neighbors:
                    covers_all = False
                    break
            
            if covers_all:
                gain = np.sum(covered_count[cand_neighbors] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_pos = cand
        
        if best_pos is not None:
            covered_count[all_neighbors[best_pos]] += 1
            code_ints.append(best_pos)
            code_set.add(best_pos)
            # Remove both old ones
            code_ints.pop(i2 if i2 > i1 else i1)
            code_ints.pop(i1)
            code_set.discard(c1)
            code_set.discard(c2)
        else:
            # Roll back
            covered_count[all_neighbors[c1]] += 1
            covered_count[all_neighbors[c2]] += 1
    
    # Final verification and repair
    if np.any(covered_count == 0):
        for w in range(total_words):
            if covered_count[w] == 0:
                word = int_to_word(w)
                if w not in code_set:
                    code_ints.append(w)
                    code_set.add(w)
                    covered_count[all_neighbors[w]] += 1
    
    # One more pass of removal
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        removed = False
        for idx in range(len(code_ints)):
            c_int = code_ints[idx]
            if np.min(covered_count[all_neighbors[c_int]]) > 1:
                covered_count[all_neighbors[c_int]] -= 1
                code_ints.pop(idx)
                code_set.discard(c_int)
                removed = True
                break
        if not removed:
            break
    
    return [int_to_word(c) for c in code_ints]
# EVOLVE-BLOCK-END

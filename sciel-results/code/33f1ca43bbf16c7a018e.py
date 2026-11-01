import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses tetracode initialization with aggressive bitmask-based reduction and swap-based local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute neighbor bitmasks using numpy for speed
    if total_words <= 729:
        # For small problems, use integer bitmask representation
        pass
    else:
        # For larger problems, we'd need a different approach, but for this task total_words=729
        pass
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors and bitmasks for all words
    # Use numpy to speed up the initial construction
    neighbor_bitmasks = [0] * total_words
    for i in range(total_words):
        w = int_to_word(i, n, q)
        mask = 1 << i  # Include self
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        mask |= (1 << neighbor_idx)
                w[pos] = original_val
        neighbor_bitmasks[i] = mask

    full_mask = (1 << total_words) - 1

    # Tetracode construction for q=3, n=6, r=1
    def generate_tetracode_init():
        """Generate the 81-word tetracode direct sum code."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = set()
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    code.add(idx)
        return code

    # Greedy initialization using bitmasks
    def greedy_init():
        uncovered_mask = full_mask
        code = set()
        while uncovered_mask and time.time() < deadline:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in code:
                    continue
                covered = neighbor_bitmasks[candidate] & uncovered_mask
                count = bin(covered).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            code.add(best_center)
            uncovered_mask &= ~neighbor_bitmasks[best_center]
        return code

    # Try multiple initializations
    initializations = []
    if q == 3 and n == 6 and radius == 1:
        initializations.append(generate_tetracode_init())
    initializations.append(greedy_init())
    
    # Add random greedy variants for diversity
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        uncovered_mask = full_mask
        code = set()
        while uncovered_mask and time.time() < deadline:
            best_center = -1
            best_count = -1
            candidates = []
            for candidate in range(total_words):
                if candidate in code:
                    continue
                covered = neighbor_bitmasks[candidate] & uncovered_mask
                count = bin(covered).count('1')
                if count > best_count:
                    best_count = count
                    candidates = [candidate]
                elif count == best_count and count > 0:
                    candidates.append(candidate)
            if candidates:
                best_center = random.choice(candidates)
            else:
                break
            code.add(best_center)
            uncovered_mask &= ~neighbor_bitmasks[best_center]
        if uncovered_mask == 0:
            initializations.append(code)
    
    # Pick the best initial size
    best_initial = min(initializations, key=len)
    best_codewords = set(best_initial)
    best_size = len(best_codewords)
    
    # Fast redundancy check using bitmasks
    def is_redundant(c, codewords):
        """Check if codeword c is redundant (all its neighbors covered by others)."""
        others_coverage = 0
        for other in codewords:
            if other != c:
                others_coverage |= neighbor_bitmasks[other]
        return (neighbor_bitmasks[c] & ~others_coverage) == 0

    # Reduce using bitmask-based redundancy
    def reduce_codewords(codewords):
        codewords = set(codewords)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for c in list(codewords):
                if is_redundant(c, codewords):
                    codewords.remove(c)
                    changed = True
                    break
        return codewords

    # Compute uncovered mask
    def get_uncovered_mask(codewords):
        covered = 0
        for c in codewords:
            covered |= neighbor_bitmasks[c]
        return full_mask & ~covered

    # Greedy repair: add codewords to cover all uncovered words
    def greedy_repair(codewords, uncovered_mask):
        while uncovered_mask and time.time() < deadline:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                covered = neighbor_bitmasks[candidate] & uncovered_mask
                count = bin(covered).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered_mask &= ~neighbor_bitmasks[best_center]
        return codewords, uncovered_mask

    # Try swap: remove one codeword, add one that covers more
    def try_swap(codewords):
        """Try to find a swap that reduces size or maintains size with better coverage."""
        codewords = set(codewords)
        if len(codewords) < 2:
            return codewords
        
        # Compute current coverage
        current_coverage = 0
        for c in codewords:
            current_coverage |= neighbor_bitmasks[c]
        
        # Try removing each codeword and see if we can add a better one
        for c_remove in list(codewords):
            if time.time() > deadline:
                break
            
            # Coverage without c_remove
            coverage_without = 0
            for c in codewords:
                if c != c_remove:
                    coverage_without |= neighbor_bitmasks[c]
            
            # Words only covered by c_remove
            only_covered_by_removed = neighbor_bitmasks[c_remove] & ~coverage_without
            
            # If nothing is only covered by c_remove, we can just remove it
            if only_covered_by_removed == 0:
                codewords.remove(c_remove)
                return codewords
            
            # Try to find a single codeword that covers all of only_covered_by_removed
            # and doesn't add too much redundancy
            best_add = -1
            best_gain = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                # How many of only_covered_by_removed does this candidate cover?
                covers_needed = neighbor_bitmasks[candidate] & only_covered_by_removed
                count = bin(covers_needed).count('1')
                if count == bin(only_covered_by_removed).count('1'):
                    # This candidate covers all needed words
                    # Check if it's a good swap (doesn't add much redundancy)
                    new_coverage = coverage_without | neighbor_bitmasks[candidate]
                    # Count how many words in new_coverage are also covered by others
                    # For simplicity, just accept if it covers all needed
                    best_add = candidate
                    best_gain = count
                    break
            
            if best_add != -1:
                codewords.remove(c_remove)
                codewords.add(best_add)
                return codewords
        
        return codewords

    # ILS with adaptive perturbation
    current_codewords = set(best_codewords)
    no_improve_count = 0
    kick_size = 1
    
    while time.time() < deadline - 5:  # Leave 5s for final processing
        # 1. Reduce
        current_codewords = reduce_codewords(current_codewords)
        
        # 2. Try swaps
        current_codewords = try_swap(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            no_improve_count = 0
            kick_size = 1
        else:
            no_improve_count += 1
            if no_improve_count % 20 == 0:
                kick_size = min(kick_size + 1, 4)
        
        if not current_codewords:
            break
            
        # 3. Perturbation (Kick) - remove random codewords
        k = min(kick_size, max(1, len(current_codewords) - 1))
        if len(current_codewords) > k:
            to_remove = random.sample(list(current_codewords), k)
            for c in to_remove:
                current_codewords.remove(c)
        
        # 4. Repair (Greedy Add)
        uncovered_mask = get_uncovered_mask(current_codewords)
        current_codewords, uncovered_mask = greedy_repair(current_codewords, uncovered_mask)
        
        if uncovered_mask:
            current_codewords = set(best_codewords)
            continue

    # Final reduction on best result
    best_codewords = reduce_codewords(best_codewords)
    best_codewords = try_swap(best_codewords)
    
    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

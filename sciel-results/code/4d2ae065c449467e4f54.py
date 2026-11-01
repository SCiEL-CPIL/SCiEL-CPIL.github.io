import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses bitset-accelerated ILS with tetracode initialization and adaptive perturbation.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
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

    # Precompute neighbors for all words
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val

    # Build bitset representation of neighbors for fast coverage computation
    # Each word's coverage is a bitmask over total_words
    neighbor_bitmasks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for nb in neighbors[i]:
            mask |= (1 << nb)
        neighbor_bitmasks[i] = mask

    # Precompute neighbor sets for O(1) membership
    neighbor_sets = [set(nbs) for nbs in neighbors]

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
        uncovered_mask = (1 << total_words) - 1
        code = set()
        while uncovered_mask:
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
    
    # Add a few random greedy variants for diversity
    for _ in range(3):
        random.shuffle(neighbors)  # This won't work as intended; let's do random tie-breaking
        # Actually, let's just do standard greedy with random tie-breaking
        uncovered_mask = (1 << total_words) - 1
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
            if best_center == -1 and candidates:
                best_center = random.choice(candidates)
            elif not candidates:
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
        # Coverage of c by all other codewords
        others_coverage = 0
        for other in codewords:
            if other != c:
                others_coverage |= neighbor_bitmasks[other]
        # All neighbors of c must be covered
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
        return ((1 << total_words) - 1) & ~covered

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

    # ILS with adaptive perturbation
    current_codewords = set(best_codewords)
    no_improve_count = 0
    kick_size = 1
    
    while time.time() < deadline:
        # 1. Reduce
        current_codewords = reduce_codewords(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            no_improve_count = 0
            # Reset kick size when we improve
            kick_size = 1
        else:
            no_improve_count += 1
            # Increase kick size periodically to escape local minima
            if no_improve_count % 50 == 0:
                kick_size = min(kick_size + 1, 4)
        
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick) - remove random codewords
        k = min(kick_size, max(1, len(current_codewords) - 1))
        if len(current_codewords) > k:
            to_remove = random.sample(list(current_codewords), k)
            for c in to_remove:
                current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        uncovered_mask = get_uncovered_mask(current_codewords)
        current_codewords, uncovered_mask = greedy_repair(current_codewords, uncovered_mask)
        
        # If repair failed to cover everything, we need to restart from best
        if uncovered_mask:
            current_codewords = set(best_codewords)
            continue

    # Final reduction on best result
    best_codewords = reduce_codewords(best_codewords)
    
    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a hybrid approach:
    1. Start from the direct sum of the ternary tetracode (81 words for 3^6, radius 1).
    2. Perform intensive local search to remove redundant codewords and swap for better packing.
    3. Uses bitmasks for fast coverage checks and repair.
    """
    start_time = time.time()
    # 120s limit. Keep a safety margin. The evaluator kills at 150s, but we should aim for <120s.
    # Let's use 115s as deadline to be safe.
    deadline = start_time + 115.0

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
    # For radius=1, neighbors are the word itself and all words at distance 1.
    # Size of ball: 1 + n*(q-1). For q=3, n=6, radius=1: 1 + 6*2 = 13.
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
        # For this task (radius=1), we don't need radius 2 neighbors.
        # If radius > 1, this would need to be expanded, but the task specifies radius=1 for the main case.
        # The contract says "radius" is passed, but the specific task is K_3(6,1).
        # We assume radius=1 for the optimization logic. If radius != 1, we might fall back to a simpler greedy.
        
    # Build neighbor bitmasks for fast coverage checks
    # Python integers are arbitrary precision, so this is efficient for 729 bits.
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    full_mask = (1 << total_words) - 1
    
    # Precompute neighbor arrays for fast NumPy-based counting
    neighbor_arrays = [np.array(neighbors[i], dtype=np.int32) for i in range(total_words)]

    # Initialize with the direct sum of the tetracode
    # Tetracode: 9 words of length 4.
    # Direct sum with 2 more coordinates: 9 * 3 * 3 = 81 words.
    initial_codewords = set()
    
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
    else:
        # Fallback: Greedy construction from scratch
        uncovered_mask = full_mask
        while uncovered_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = bin(uncovered_mask & neighbor_masks[candidate]).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)

    def get_coverage_mask(codewords):
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def compute_min_counts(codewords):
        """
        Returns an array min_counts where min_counts[i] is the minimum coverage count
        among the neighbors of codeword i (where i indexes the list codewords).
        """
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            counts[neighbor_arrays[c]] += 1
        
        min_counts = np.zeros(len(codewords), dtype=np.int32)
        for i, c in enumerate(codewords):
            min_counts[i] = np.min(counts[neighbor_arrays[c]])
        return min_counts

    def reduce_codewords(codewords):
        """
        Iteratively remove redundant codewords until no more can be removed.
        A codeword is redundant if min_counts[i] >= 2.
        Optimized to avoid full recount if possible, but for simplicity and correctness
        within the time limit, we use a faster loop structure.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Compute counts once per iteration
            counts = np.zeros(total_words, dtype=np.int32)
            for c in codewords:
                counts[neighbor_arrays[c]] += 1
            
            codeword_list = list(codewords)
            redundant_idx = -1
            # Find any codeword whose ball is fully covered by others (min count >= 2)
            for i, c in enumerate(codeword_list):
                if np.min(counts[neighbor_arrays[c]]) >= 2:
                    redundant_idx = i
                    break
            
            if redundant_idx != -1:
                c_to_remove = codeword_list[redundant_idx]
                codewords.remove(c_to_remove)
                changed = True
        return codewords

    def greedy_repair(codewords):
        """
        Add codewords to cover any uncovered words.
        """
        covered = get_coverage_mask(codewords)
        uncovered = full_mask & ~covered
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                count = bin(uncovered & neighbor_masks[candidate]).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    def find_least_critical(codewords, k=1):
        """
        Find the k least critical codewords.
        Criticality is measured by the minimum coverage count in their ball.
        Lower minimum count -> more critical.
        We want the ones with the HIGHEST minimum count (least critical).
        """
        if len(codewords) <= k:
            return list(codewords)
        
        min_counts = compute_min_counts(codewords)
        codeword_list = list(codewords)
        pairs = list(zip(min_counts, range(len(codeword_list))))
        pairs.sort(key=lambda x: x[0], reverse=True)
        return [codeword_list[idx] for _, idx in pairs[:k]]

    # Precompute short-vector moves for group-structured perturbations in Z_3^n
    # A "short vector" changes one or two coordinates by ±1 mod q.
    short_vector_moves = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        # Single-coordinate moves (distance 1)
        for pos in range(n):
            orig = w[pos]
            for delta in [1, q - 1]:
                new_val = (orig + delta) % q
                w[pos] = new_val
                new_idx = word_to_int(w, q)
                if new_idx != i:
                    short_vector_moves[i].append(new_idx)
                w[pos] = orig
        # Two-coordinate moves (distance 2, changing two coords by ±1)
        if n >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    o1, o2 = w[pos1], w[pos2]
                    for d1 in [1, q - 1]:
                        for d2 in [1, q - 1]:
                            w[pos1] = (o1 + d1) % q
                            w[pos2] = (o2 + d2) % q
                            new_idx = word_to_int(w, q)
                            if new_idx != i:
                                short_vector_moves[i].append(new_idx)
                            w[pos1] = o1
                            w[pos2] = o2

    # Main Search Loop
    current_codewords = set(best_codewords)
    
    # First, reduce the initial solution
    current_codewords = reduce_codewords(current_codewords)
    if len(current_codewords) < best_size:
        best_size = len(current_codewords)
        best_codewords = set(current_codewords)

    # Iterated Local Search with Simulated Annealing-like acceptance
    # We keep a copy of the previous state to allow rollback
    prev_codewords = set(current_codewords)
    temperature = 5.0
    cooling_rate = 0.995
    min_temp = 0.5
    
    while time.time() < deadline - 8:
        # 1. Perturbation
        perturbation_type = random.random()
        
        if perturbation_type < 0.25:
            # Remove 1 least critical
            if len(current_codewords) > 1:
                least_critical = find_least_critical(current_codewords, 1)
                current_codewords.remove(least_critical[0])
        elif perturbation_type < 0.45:
            # Group-structured move: shift a codeword along a short vector
            if len(current_codewords) > 1:
                least_critical = find_least_critical(current_codewords, 1)
                to_move = least_critical[0]
                moves = short_vector_moves[to_move]
                if moves:
                    new_pos = random.choice(moves)
                    if new_pos not in current_codewords:
                        current_codewords.remove(to_move)
                        current_codewords.add(new_pos)
                    else:
                        # Collision: just remove
                        current_codewords.remove(to_move)
                else:
                    current_codewords.remove(to_move)
            else:
                current_codewords = set(prev_codewords)
        elif perturbation_type < 0.65:
            # Swap 1 (random neighbor)
            if len(current_codewords) > 1:
                least_critical = find_least_critical(current_codewords, 1)
                to_remove = least_critical[0]
                candidates = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if candidates:
                    to_add = random.choice(candidates)
                    current_codewords.remove(to_remove)
                    current_codewords.add(to_add)
                else:
                    current_codewords.remove(to_remove)
        elif perturbation_type < 0.80:
            # Remove 2
            if len(current_codewords) > 2:
                least_critical = find_least_critical(current_codewords, 2)
                for c in least_critical:
                    current_codewords.remove(c)
        elif perturbation_type < 0.90:
            # Group-structured: shift 2 codewords along short vectors
            if len(current_codewords) > 2:
                least_critical = find_least_critical(current_codewords, 2)
                for c in least_critical:
                    moves = short_vector_moves[c]
                    if moves:
                        new_pos = random.choice(moves)
                        if new_pos not in current_codewords:
                            current_codewords.remove(c)
                            current_codewords.add(new_pos)
                        else:
                            current_codewords.remove(c)
                    else:
                        current_codewords.remove(c)
        else:
            # Random removal
            k = random.randint(1, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 2. Repair
        current_codewords = greedy_repair(current_codewords)
        
        # 3. Reduce
        current_codewords = reduce_codewords(current_codewords)
        
        new_size = len(current_codewords)
        old_size = len(prev_codewords)
        
        # 4. Acceptance
        if new_size < old_size:
            # Always accept improvement
            prev_codewords = set(current_codewords)
        elif new_size == old_size:
            # Accept with probability 0.5
            if random.random() < 0.5:
                prev_codewords = set(current_codewords)
            else:
                current_codewords = set(prev_codewords)
        else:
            # Worse. Accept with prob exp(-delta/T)
            delta = new_size - old_size
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                prev_codewords = set(current_codewords)
            else:
                current_codewords = set(prev_codewords)
                
        # Update best
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
            if best_size <= 71:
                break

        # Cool down
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp

    # Final Polish: Try 2-swap moves (remove 2, add 1) on the best solution
    # This can sometimes reduce the size by 1.
    # Speedup: extract uncovered word indices as a list, then for each candidate
    # check if all uncovered words are in its neighbor set (via a precomputed set).
    
    # Precompute neighbor sets for O(1) membership
    neighbor_sets = [set(neighbors[i]) for i in range(total_words)]
    
    def final_polish(codewords, time_limit):
        """Try 2-for-0 and 2-for-1 improvements. Returns improved set or None."""
        improved = False
        codeword_list = list(codewords)
        random.shuffle(codeword_list)
        n_cw = len(codeword_list)
        
        for i in range(n_cw):
            if time.time() > time_limit:
                break
            if improved:
                break
            c1 = codeword_list[i]
            if c1 not in codewords:
                continue
            for j in range(i + 1, n_cw):
                if time.time() > time_limit:
                    break
                c2 = codeword_list[j]
                if c2 not in codewords:
                    continue
                
                # Tentatively remove c1 and c2
                test_set = set(codewords)
                test_set.remove(c1)
                test_set.remove(c2)
                
                covered = get_coverage_mask(test_set)
                uncovered_mask = full_mask & ~covered
                
                if not uncovered_mask:
                    # 2-for-0: removed 2, no loss of coverage
                    return test_set
                
                # Extract uncovered word indices
                # Convert mask to list of indices
                uncov_list = []
                m = uncovered_mask
                while m:
                    low_bit = m & (-m)
                    idx = low_bit.bit_length() - 1
                    uncov_list.append(idx)
                    m ^= low_bit
                
                # Try to find a single codeword that covers all uncovered
                best_add = -1
                for candidate in range(total_words):
                    if candidate in test_set:
                        continue
                    # Check if candidate covers all uncovered words
                    ns = neighbor_sets[candidate]
                    all_covered = True
                    for u in uncov_list:
                        if u not in ns:
                            all_covered = False
                            break
                    if all_covered:
                        best_add = candidate
                        break
                
                if best_add != -1:
                    test_set.add(best_add)
                    return test_set
        
        return None
    
    final_codewords = set(best_codewords)
    
    if time.time() < deadline - 2:
        result = final_polish(final_codewords, deadline - 2)
        if result is not None:
            final_codewords = result
            if len(final_codewords) < best_size:
                best_size = len(final_codewords)
                best_codewords = set(final_codewords)
    
    # Second polish pass if time remains (catches different pair orderings)
    if time.time() < deadline - 1 and len(best_codewords) > 71:
        result2 = final_polish(set(best_codewords), deadline - 1)
        if result2 is not None and len(result2) < best_size:
            best_size = len(result2)
            best_codewords = set(result2)

    # Convert back to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

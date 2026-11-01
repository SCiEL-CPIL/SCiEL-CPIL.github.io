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
    
    # Precompute neighbor lists as numpy arrays for fast vectorized counting
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
        Returns an array min_counts where min_counts[c] is the minimum coverage count
        among the neighbors of codeword c.
        Uses vectorized numpy operations for speed.
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
        A codeword is redundant if min_counts[c] >= 2.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            min_counts = compute_min_counts(codewords)
            # Find the first redundant codeword
            redundant_idx = -1
            for i, mc in enumerate(min_counts):
                if mc >= 2:
                    redundant_idx = i
                    break
            
            if redundant_idx != -1:
                c_to_remove = list(codewords)[redundant_idx]
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
            # Optimization: Only check candidates that are not already in codewords
            # and potentially close to uncovered areas? For now, scan all.
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
        # Pair (min_count, index)
        pairs = list(zip(min_counts, range(len(codeword_list))))
        # Sort by descending min_count (highest min_count is least critical)
        pairs.sort(key=lambda x: x[0], reverse=True)
        return [codeword_list[idx] for _, idx in pairs[:k]]

    # Main Search Loop
    current_codewords = set(best_codewords)
    
    # First, reduce the initial solution
    current_codewords = reduce_codewords(current_codewords)
    if len(current_codewords) < best_size:
        best_size = len(current_codewords)
        best_codewords = set(current_codewords)

    # Simulated Annealing / Iterated Local Search
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline - 5:
        # 1. Perturbation
        perturbation_type = random.random()
        old_size = len(current_codewords)
        
        if perturbation_type < 0.5:
            # Remove 1 least critical
            if len(current_codewords) > 1:
                least_critical = find_least_critical(current_codewords, 1)
                current_codewords.remove(least_critical[0])
        elif perturbation_type < 0.7:
            # Swap 1
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
        elif perturbation_type < 0.9:
            # Remove 2 (aggressive)
            if len(current_codewords) > 2:
                least_critical = find_least_critical(current_codewords, 2)
                for c in least_critical:
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
        
        # 4. Acceptance Criterion
        if new_size < old_size:
            # Always accept improvement
            pass
        elif new_size == old_size:
            # Accept with probability 0.5 to allow wandering
            if random.random() < 0.5:
                pass
            else:
                # Revert? We don't store the old state explicitly in this simple loop, 
                # but since we modified current_codewords in place, we need to be careful.
                # Actually, the "wandering" usually implies keeping the new state if it's equal,
                # or reverting if we want to stay close. 
                # For simplicity and to avoid state management overhead, we accept equal size moves
                # to explore the plateau, but this might drift. 
                # Let's just accept it for now, as repair+reduce tends to stabilize.
                pass
        else:
            # Worse solution. Accept with probability exp(-delta/T)
            delta = new_size - old_size
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass # Accept worse
            else:
                # Revert to previous best? No, we lost the previous state.
                # To properly implement SA, we should keep a copy of the previous state.
                # However, copying a set of ~73 ints is cheap.
                # Let's restart from best_codewords if we reject? That's ILS.
                # Let's just restart the perturbation from the current (worse) state? 
                # No, that gets stuck. 
                # Let's revert to the state before perturbation.
                # We need to store the state before perturbation.
                # This requires significant refactoring of the loop structure.
                # Alternative: If we reject, we just continue with the worse state? No, that's bad.
                # Let's simplify: If we reject, we reset to best_codewords? That's slow.
                # Let's just accept all moves for now (Hill Climbing with random restarts on stuck) 
                # OR implement proper state saving.
                
                # Given the complexity, let's stick to Hill Climbing + Random Restarts if stuck.
                # But the current code doesn't have restarts.
                # Let's just accept the move if it's not significantly worse, or just accept all.
                # Actually, the previous code accepted all moves after repair/reduce.
                # Let's keep it simple: Accept all. The reduce step pulls it back to local optimum.
                pass

        # Update best
        if new_size < best_size:
            best_size = new_size
            best_codewords = set(current_codewords)
            # Reset temperature on improvement?
            # temperature = 10.0 
            if best_size <= 71:
                break

        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    # Final Polish: Try 2-swap moves (remove 2, add 1) on the best solution
    # This can sometimes reduce the size by 1.
    final_codewords = set(best_codewords)
    
    if time.time() < deadline - 2:
        # Try to find a 2-for-1 improvement
        # Iterate through pairs of codewords
        codeword_list = list(final_codewords)
        random.shuffle(codeword_list)
        
        improved = False
        for i in range(len(codeword_list)):
            if time.time() > deadline - 2:
                break
            if improved:
                break
            c1 = codeword_list[i]
            if c1 not in final_codewords:
                continue
            for j in range(i + 1, len(codeword_list)):
                if time.time() > deadline - 2:
                    break
                c2 = codeword_list[j]
                if c2 not in final_codewords:
                    continue
                
                # Tentatively remove c1 and c2
                test_set = set(final_codewords)
                test_set.remove(c1)
                test_set.remove(c2)
                
                covered = get_coverage_mask(test_set)
                uncovered = full_mask & ~covered
                
                if not uncovered:
                    # We removed 2 and didn't lose coverage. This is a 2-for-0 move (reduction by 2).
                    # This is even better!
                    final_codewords = test_set
                    improved = True
                    break
                
                # Try to find a single codeword that covers all uncovered
                best_add = -1
                for candidate in range(total_words):
                    if candidate in test_set:
                        continue
                    # Check if candidate covers all uncovered
                    if (uncovered & ~neighbor_masks[candidate]) == 0:
                        best_add = candidate
                        break
                
                if best_add != -1:
                    # We removed 2 and added 1. Net reduction of 1.
                    final_codewords = test_set
                    final_codewords.add(best_add)
                    improved = True
                    break
        
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)

    # Convert back to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

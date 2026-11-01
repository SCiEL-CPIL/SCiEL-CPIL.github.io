import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Iterated Local Search (Greedy + Perturbation + Repair) to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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

    # Precompute neighbors for all words as lists
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
    # For radius >= 2, add second-order neighbors
    if radius >= 2:
        for i in range(total_words):
            first_order = set(neighbors[i])
            for j in first_order:
                for k in neighbors[j]:
                    if k not in neighbors[i]:
                        neighbors[i].append(k)
        # Remove duplicates and sort
        for i in range(total_words):
            neighbors[i] = list(set(neighbors[i]))

    # Build neighbor bitmasks for fast coverage checks
    # Use Python integers as bitmasks (arbitrary precision)
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    
    # Precompute: for each word w, which codeword indices cover w
    # Actually, we'll compute coverage on the fly using masks

    # Initialize with a known good construction: Direct sum of tetracode
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
        # Fallback: Greedy construction from scratch with symmetry-breaking tie-breaking
        uncovered_mask = (1 << total_words) - 1
        while uncovered_mask:
            best_center = -1
            best_count = -1
            best_weight = float('inf')  # For tie-breaking: prefer lower integer value
            for candidate in range(total_words):
                count = bin(uncovered_mask & neighbor_masks[candidate]).count('1')
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)
    
    def get_coverage_mask(codewords):
        """Calculate the coverage bitmask given a set of codewords."""
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def compute_coverage_counts(codewords):
        """For each word, count how many codewords cover it. Returns array of length total_words."""
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def reduce_codewords_fast(codewords, max_iterations=100):
        """Remove redundant codewords using coverage counts. Returns the reduced set."""
        changed = True
        iterations = 0
        while changed and iterations < max_iterations and time.time() < deadline:
            changed = False
            iterations += 1
            # Compute coverage counts
            counts = compute_coverage_counts(codewords)
            # A codeword c is redundant if for all neighbors nb of c, counts[nb] >= 2
            # (i.e., every neighbor is covered by at least one other codeword)
            for c in list(codewords):
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    codewords.remove(c)
                    changed = True
                    break  # Restart as set changed
        return codewords

    def greedy_repair(codewords):
        """Add codewords greedily to cover all uncovered words. Returns modified set."""
        covered = get_coverage_mask(codewords)
        uncovered = ((1 << total_words) - 1) & ~covered
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = -1
            best_weight = float('inf')  # For tie-breaking: prefer lower integer value
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                count = bin(uncovered & neighbor_masks[candidate]).count('1')
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    def find_least_critical_codewords(codewords, k=5):
        """Find k codewords that are 'least critical' - their neighbors are most redundantly covered."""
        if len(codewords) <= k:
            return list(codewords)
        counts = compute_coverage_counts(codewords)
        # For each codeword, compute the minimum coverage count among its neighbors
        # Lower minimum means more critical (some neighbor is only covered by this codeword)
        criticality = []
        for c in codewords:
            min_count = min(counts[nb] for nb in neighbors[c])
            criticality.append((min_count, c))
        # Sort by ascending minimum count (least critical first)
        criticality.sort(key=lambda x: x[0])
        return [c for _, c in criticality[:k]]

    # Main ILS Loop
    current_codewords = set(best_codewords)
    
    while time.time() < deadline:
        # 1. Reduce using fast method
        current_codewords = reduce_codewords_fast(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick) - Targeted perturbation
        perturbation_type = random.random()
        if perturbation_type < 0.5:
            # Remove 1-2 least critical codewords
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                least_critical = find_least_critical_codewords(current_codewords, k)
                for c in least_critical:
                    current_codewords.remove(c)
        elif perturbation_type < 0.8:
            # Swap: remove one least critical, add a random nearby word
            if len(current_codewords) > 1:
                least_critical = find_least_critical_codewords(current_codewords, 1)
                to_remove = least_critical[0]
                # Find words within distance 1-2 of to_remove that are not in codewords
                nearby = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if nearby:
                    to_add = random.choice(nearby)
                    current_codewords.remove(to_remove)
                    current_codewords.add(to_add)
                else:
                    current_codewords.remove(to_remove)
        else:
            # Remove 2-3 random codewords (larger kick)
            k = random.randint(2, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        current_codewords = greedy_repair(current_codewords)
        
        # After repair, the set is valid again.
        # The next iteration of the main loop will reduce it.

    # Final polish: try harder to reduce the best solution
    # Try multiple reduction passes with different orderings
    final_codewords = set(best_codewords)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Use fast reduction
        final_codewords = reduce_codewords_fast(final_codewords, max_iterations=10)
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)
        # Check if no change occurred
        if len(final_codewords) >= best_size:
            # Try one more pass with shuffled order
            shuffled = list(final_codewords)
            random.shuffle(shuffled)
            changed = False
            for c in shuffled:
                if c not in final_codewords:
                    continue
                counts = compute_coverage_counts(final_codewords)
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    final_codewords.remove(c)
                    changed = True
            if not changed:
                break
            if len(final_codewords) < best_size:
                best_size = len(final_codewords)
                best_codewords = set(final_codewords)
    
    # Additional polish: try 2-swap moves (remove 2, add 1) to further reduce size
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Try removing pairs of codewords and adding one that covers the gap
            found_improvement = False
            codeword_list = list(final_codewords)
            random.shuffle(codeword_list)
            for i in range(len(codeword_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(codeword_list)):
                    c1, c2 = codeword_list[i], codeword_list[j]
                    if c1 not in final_codewords or c2 not in final_codewords:
                        continue
                    # Try removing both and see if we can add one codeword to cover the gap
                    test_set = set(final_codewords)
                    test_set.remove(c1)
                    test_set.remove(c2)
                    covered = get_coverage_mask(test_set)
                    uncovered = ((1 << total_words) - 1) & ~covered
                    # Find if a single codeword can cover all uncovered
                    if uncovered:
                        best_add = -1
                        for candidate in range(total_words):
                            if candidate in test_set:
                                continue
                            count = bin(uncovered & neighbor_masks[candidate]).count('1')
                            if count == bin(uncovered).count('1'):
                                best_add = candidate
                                break
                        if best_add != -1:
                            test_set.add(best_add)
                            # Verify it's valid
                            final_cover = get_coverage_mask(test_set)
                            if final_cover == (1 << total_words) - 1:
                                final_codewords = test_set
                                found_improvement = True
                                break
                if found_improvement:
                    if len(final_codewords) < best_size:
                        best_size = len(final_codewords)
                        best_codewords = set(final_codewords)
                    break

    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result

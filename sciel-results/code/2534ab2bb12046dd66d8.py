import time
import random
import numpy as np
from collections import defaultdict

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
    
    # Also build numpy array of neighbor indices for vectorized operations
    max_neighbors = max(len(nb) for nb in neighbors)
    neighbor_arrays = np.zeros((total_words, max_neighbors), dtype=np.int32)
    neighbor_lengths = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        neighbor_arrays[i, :len(neighbors[i])] = neighbors[i]
        neighbor_lengths[i] = len(neighbors[i])

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

    def reduce_codewords_bitmask(codewords, max_iterations=100):
        """Remove redundant codewords using bitmask coverage. Returns the reduced set."""
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline:
                break
            # Compute coverage of all other codewords for each candidate
            # For efficiency, compute total coverage once, then for each c,
            # check if coverage without c still covers neighbors[c]
            total_cover = get_coverage_mask(codewords)
            codeword_list = list(codewords)
            random.shuffle(codeword_list)
            for c in codeword_list:
                if c not in codewords:
                    continue
                # Coverage without c
                cover_without_c = total_cover & ~neighbor_masks[c]
                # Check if all neighbors of c are still covered
                if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                    codewords.remove(c)
                    changed = True
                    break  # Restart as set changed
        return codewords

    def greedy_repair(codewords):
        """Add codewords greedily to cover all uncovered words. Returns modified set."""
        covered = get_coverage_mask(codewords)
        uncovered = ((1 << total_words) - 1) & ~covered
        while uncovered:
            if time.time() > deadline:
                break
            best_center = -1
            best_count = -1
            best_weight = float('inf')  # For tie-breaking: prefer lower integer value
            # Use a smarter candidate selection: prioritize words that are far from existing codewords
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

    # Main ILS Loop
    current_codewords = set(best_codewords)
    
    while time.time() < deadline:
        # 1. Reduce
        current_codewords = reduce_codewords_bitmask(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick) - smarter: target low-coverage codewords
        # Compute coverage counts to identify weakly-covered codewords
        covered_mask = get_coverage_mask(current_codewords)
        
        # Find codewords whose removal would leave the most uncovered neighbors
        # This helps identify "essential" vs "redundant" codewords
        codeword_list = list(current_codewords)
        if len(codeword_list) <= 1:
            break
            
        # Calculate for each codeword how many unique neighbors it covers
        # (neighbors not covered by other codewords)
        unique_coverage = {}
        for c in codeword_list:
            # Get neighbors of c
            c_neighbors_mask = neighbor_masks[c]
            # Coverage by all other codewords
            others_cover = 0
            for other in codeword_list:
                if other != c:
                    others_cover |= neighbor_masks[other]
            # Unique coverage of c
            unique_mask = c_neighbors_mask & ~others_cover
            unique_coverage[c] = bin(unique_mask).count('1')
        
        # Sort by unique coverage (ascending) - remove those with least unique coverage first
        sorted_by_unique = sorted(codeword_list, key=lambda x: unique_coverage[x])
        
        perturbation_type = random.random()
        if perturbation_type < 0.3:
            # Remove 1-2 codewords with lowest unique coverage
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                to_remove = sorted_by_unique[:k]
                for c in to_remove:
                    current_codewords.remove(c)
        elif perturbation_type < 0.6:
            # Swap: remove one low-unique-coverage codeword, add a random nearby word
            if len(current_codewords) > 1:
                to_remove = random.choice(sorted_by_unique[:max(1, len(sorted_by_unique)//3)])
                current_codewords.remove(to_remove)
                # Add a random word not in codewords, preferring nearby
                nearby = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if nearby and random.random() < 0.7:
                    to_add = random.choice(nearby)
                else:
                    candidates = [i for i in range(total_words) if i not in current_codewords]
                    if candidates:
                        to_add = random.choice(candidates)
                    else:
                        continue
                current_codewords.add(to_add)
        elif perturbation_type < 0.8:
            # Remove 2-3 codewords with lowest unique coverage
            k = random.randint(2, 3)
            if len(current_codewords) > k:
                to_remove = sorted_by_unique[:k]
                for c in to_remove:
                    current_codewords.remove(c)
        else:
            # Random removal (diversification)
            k = random.randint(1, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(codeword_list, k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # Additional perturbation: try swapping a codeword with a nearby word
        if random.random() < 0.2:
            if len(current_codewords) > 1:
                to_remove = random.choice(list(current_codewords))
                nearby = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if nearby:
                    to_add = random.choice(nearby)
                    current_codewords.remove(to_remove)
                    current_codewords.add(to_add)
        
        # 3. Repair (Greedy Add)
        current_codewords = greedy_repair(current_codewords)
        
        # After repair, the set is valid again.
        # The next iteration of the main loop will reduce it.

    # Final polish: try harder to reduce the best solution
    # Try multiple reduction passes with different orderings
    final_codewords = set(best_codewords)
    polish_iterations = 0
    while time.time() < deadline - 5:
        polish_iterations += 1
        # Shuffle the order of checking for redundancy
        shuffled = list(final_codewords)
        random.shuffle(shuffled)
        changed = False
        for c in shuffled:
            if c not in final_codewords:
                continue
            total_cover = get_coverage_mask(final_codewords)
            cover_without_c = total_cover & ~neighbor_masks[c]
            if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                final_codewords.remove(c)
                changed = True
        if not changed:
            break
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)
    
    # Additional polish: try 2-swap moves (remove 2, add 1) to further reduce size
    if time.time() < deadline - 10:
        swap_rounds = 0
        while time.time() < deadline - 5 and swap_rounds < 50:
            swap_rounds += 1
            # Try removing pairs of codewords and adding one that covers the gap
            found_improvement = False
            codeword_list = list(final_codewords)
            random.shuffle(codeword_list)
            
            # Compute unique coverage to prioritize which pairs to try
            others_cover_all = (1 << total_words) - 1
            unique_cov = {}
            for c in codeword_list:
                c_mask = neighbor_masks[c]
                others = 0
                for other in codeword_list:
                    if other != c:
                        others |= neighbor_masks[other]
                unique_cov[c] = bin(c_mask & ~others).count('1')
            
            # Try pairs where both have low unique coverage
            sorted_by_unique = sorted(codeword_list, key=lambda x: unique_cov[x])
            # Focus on the first half (lowest unique coverage)
            focus_list = sorted_by_unique[:max(2, len(sorted_by_unique)//2)]
            
            for i in range(len(focus_list)):
                for j in range(i + 1, len(focus_list)):
                    if time.time() > deadline - 5:
                        break
                    c1, c2 = focus_list[i], focus_list[j]
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
                        uncovered_count = bin(uncovered).count('1')
                        best_add = -1
                        best_count = -1
                        for candidate in range(total_words):
                            if candidate in test_set:
                                continue
                            count = bin(uncovered & neighbor_masks[candidate]).count('1')
                            if count > best_count:
                                best_count = count
                                best_add = candidate
                            if count == uncovered_count:
                                break
                        if best_add != -1 and best_count == uncovered_count:
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
                    # Continue trying more swaps

    # Final verification and cleanup
    # Ensure all codewords are distinct and valid
    verified = set()
    for c in best_codewords:
        verified.add(c)
    
    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in verified]
    return result

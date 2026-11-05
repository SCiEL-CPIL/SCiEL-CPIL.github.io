"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    # Deadline guard: finish well before the 120s wall-clock limit.
    deadline = time.time() + 115.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct product with a single symbol
        # This is not optimal for general cases but keeps the interface valid.
        # For the specific task (q=3, n=6, r=1) we use the specialized logic below.
        # A simple greedy construction for general case:
        codewords = []
        covered = [False] * (q ** n)
        # Map tuple to index
        def tup_to_idx(t):
            idx = 0
            for i in range(n):
                idx = idx * q + t[i]
            return idx
        # Precompute neighbors for speed if possible, but for general q,n it's heavy.
        # We'll just do a basic greedy here for non-target cases.
        for word in itertools.product(range(q), repeat=n):
            w_idx = tup_to_idx(word)
            if not covered[w_idx]:
                codewords.append(list(word))
                # Mark this and all neighbors as covered
                covered[w_idx] = True
                for pos in range(n):
                    for val in range(q):
                        if val == word[pos]:
                            continue
                        new_word = list(word)
                        new_word[pos] = val
                        new_idx = tup_to_idx(tuple(new_word))
                        covered[new_idx] = True
                if time.time() > deadline - 5:
                    break
        return codewords

    # Specialized for q=3, n=6, radius=1
    # Total words: 729
    # Start with the known optimal structure based on tetracode or a strong seed.
    
    # Generate all 729 words as tuples
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1) for each word
    # neighbor_mask[i] is a set of indices covered by word i
    # Actually, we need to check if a word is covered.
    # Let's use a boolean array 'covered' of size 729.
    
    def get_neighbors(w):
        """Yield indices of words within Hamming distance 1 of w."""
        yield word_to_idx[w]
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    yield word_to_idx[tuple(new_w)]

    # We will use a greedy set cover approach with randomization and local search.
    # Initial solution: A known good code of size 73.
    # The tetracode product gives 9 * 3^2 = 81 words. We can try to reduce this.
    # Or start with a random valid code and reduce it.
    
    # Strategy:
    # 1. Generate a valid code using a greedy algorithm (might be larger than 73).
    # 2. Try to remove redundant codewords.
    # 3. Try to swap codewords to reduce count.
    
    # Step 1: Greedy construction
    # To speed up, we can precompute the coverage sets.
    coverage = [None] * 729
    for i, w in enumerate(all_words):
        coverage[i] = set(get_neighbors(w))
    
    # Greedy Set Cover
    covered = [False] * 729
    code = []
    remaining = set(range(729))
    
    # Optimization: Instead of checking all 729 words at each step, we can just
    # pick a random uncovered word and add a random word that covers it? No, greedy
    # usually picks the word that covers the most uncovered points.
    # With 729 points, O(N^2) per step is 729*729 ~ 500k ops. 73 steps -> 36M ops.
    # This is fast enough in Python if optimized.
    
    # Let's try a faster greedy:
    # Maintain a list of uncovered indices.
    # For each candidate codeword c, count how many uncovered points it covers.
    # Pick the max.
    
    # This is expensive in pure Python. Let's try a different approach:
    # Start with the 81-word tetracode code and try to remove words.
    
    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(tuple(h + list(tail)))
    
    # Convert to indices
    initial_indices = [word_to_idx[w] for w in initial_code]
    
    # Check if valid (should be)
    # Now try to remove redundant codewords.
    
    best_code = set(initial_indices)
    best_size = len(best_code)
    
    # Local search: try to remove a codeword if the rest still covers everything.
    # To check coverage, we need to union the coverage sets of the remaining codewords.
    # This is O(K * 19) per check. K=73, so 73*19 ~ 1400 ops.
    # We can do this for each of the 73 codewords. 73 * 1400 ~ 100k ops per pass.
    # We can iterate this until no more removals are possible.
    
    def is_covered(code_set):
        covered = [False] * 729
        for idx in code_set:
            for nb in coverage[idx]:
                covered[nb] = True
        # Check if all 729 are covered
        return all(covered)
    
    # Optimization for is_covered: use a bitset or integer mask?
    # 729 bits is about 9 ints of 64 bits. Or just a boolean list.
    # Let's stick to boolean list for simplicity, it's fast enough.
    
    # Even better: maintain a count of how many times each point is covered.
    # When we remove a codeword, we decrement the counts for its neighbors.
    # If any count drops to 0, it's not covered.
    
    def try_remove_redundant(code_list):
        """Returns a reduced list of codewords."""
        code_set = set(code_list)
        # Count coverage
        counts = [0] * 729
        for idx in code_set:
            for nb in coverage[idx]:
                counts[nb] += 1
        
        # Try to remove each codeword
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if removing idx leaves any point with count 0
                # We need to check if any neighbor of idx has count == 1
                # If so, removing idx makes it 0 -> invalid.
                # If all neighbors of idx have count >= 2, we can remove it.
                can_remove = True
                for nb in coverage[idx]:
                    if counts[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove idx
                    code_set.remove(idx)
                    for nb in coverage[idx]:
                        counts[nb] -= 1
                    changed = True
        return list(code_set)
    
    # Run the reduction
    current_code = try_remove_redundant(initial_indices)
    best_size = len(current_code)
    
    # Now try to improve by swapping or adding/removing to find smaller codes.
    # Since 73 is the known best, and we started with 81, we might have already found 73 or close.
    # Let's check the size.
    
    # If we are still above 73, we can try random restarts or local search.
    # But 81 -> 73 is a significant reduction. The reduction algorithm above is greedy.
    # It might get stuck at a local minimum.
    
    # Let's add a randomization component: shuffle the order of removal attempts.
    # Or try multiple random initial codes? No, tetracode is a good structure.
    
    # Let's try a different local search:
    # If we can't remove a codeword, try to replace it with another one that allows removal of two others?
    # This is complex.
    
    # Given the time limit, let's just run the greedy reduction a few times with different orders.
    
    best_result = current_code
    
    # Try multiple random orderings of the initial code for the reduction process
    # to escape local minima.
    
    for _ in range(50):  # 50 restarts
        if time.time() > deadline - 10:
            break
        
        shuffled_code = initial_indices[:]
        random.shuffle(shuffled_code)
        
        # Run reduction with this specific order?
        # My try_remove_redundant function iterates over a set, so order is not fixed.
        # To make it order-dependent, we should iterate over a list.
        
        def try_remove_redundant_ordered(code_list):
            code_set = set(code_list)
            counts = [0] * 729
            for idx in code_set:
                for nb in coverage[idx]:
                    counts[nb] += 1
            
            changed = True
            # We'll use a list to control order
            current_list = code_list[:]
            while changed:
                changed = False
                for i in range(len(current_list)):
                    idx = current_list[i]
                    can_remove = True
                    for nb in coverage[idx]:
                        if counts[nb] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove
                        code_set.remove(idx)
                        for nb in coverage[idx]:
                            counts[nb] -= 1
                        current_list.pop(i)
                        changed = True
                        break  # Restart scan from beginning
            return list(code_set)
        
        res = try_remove_redundant_ordered(shuffled_code)
        if len(res) < len(best_result):
            best_result = res
    
    # Final check: ensure the best_result is valid
    # (It should be, but just in case)
    if not is_covered(set(best_result)):
        # Fallback to initial
        best_result = initial_indices
    
    # Convert back to lists of integers
    return [list(all_words[idx]) for idx in best_result]

# EVOLVE-BLOCK-END

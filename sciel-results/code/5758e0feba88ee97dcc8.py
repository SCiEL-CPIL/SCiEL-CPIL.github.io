import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = q ** n
    
    # Precompute powers for fast index conversion
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * q_powers[n-1-i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # For n=8, q=3, N=6561. 
    # We can use Python integers as bitsets for coverage masks since 2^6561 is large but Python handles big ints.
    # However, operations on 6561-bit ints are fast in C.
    
    # Precompute neighbor masks for all words
    # neighbor_masks[i] is a bitset of all words within radius of i
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    if radius == 1:
        for c_idx in range(N):
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            for i in range(n):
                pow_q = q_powers[n-1-i]
                orig_digit = (c_idx // pow_q) % q
                for new_val in range(q):
                    if new_val != orig_digit:
                        w_idx = c_idx + (new_val - orig_digit) * pow_q
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask
    else:
        # Generic case, slower but robust
        all_words = [idx_to_word(i) for i in range(N)]
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask

    def get_coders_for_word(w_idx):
        """Return list of codeword indices that cover w_idx"""
        coders = [w_idx]
        if radius == 1:
            for i in range(n):
                pow_q = q_powers[n-1-i]
                orig_digit = (w_idx // pow_q) % q
                for new_val in range(q):
                    if new_val != orig_digit:
                        c_idx = w_idx + (new_val - orig_digit) * pow_q
                        coders.append(c_idx)
        else:
             # Fallback for radius > 1 (unlikely in this specific task)
             w_word = [0]*n
             temp = w_idx
             for i in range(n-1, -1, -1):
                 w_word[i] = temp % q
                 temp //= q
             for dist in range(1, radius + 1):
                 for positions in itertools.combinations(range(n), dist):
                     for vals in itertools.product(range(q), repeat=dist):
                         if all(vals[j] == w_word[positions[j]] for j in range(dist)):
                             continue
                         new_word = w_word.copy()
                         for j, pos in enumerate(positions):
                             new_word[pos] = vals[j]
                         c_idx = word_to_idx(new_word)
                         coders.append(c_idx)
        return coders

    def greedy_cover(rng, permutation=None):
        """
        Greedy set cover. 
        permutation: list of indices 0..n-1. If provided, we process digits in this order 
        to break symmetry (spectral-like partitioning effect).
        """
        if permutation is None:
            permutation = list(range(n))
            
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # We prioritize words that are "hard" to cover? 
        # In regular graphs, all words are symmetric. 
        # However, the *choice* of codeword matters.
        # Standard greedy: pick codeword covering most uncovered.
        
        # Optimization: Instead of scanning all N codewords every time (O(N^2)),
        # we can pick a random uncovered word and find its best coder.
        # This is O(N * degree * N) worst case if we scan all coders for that word.
        # Degree is q*n = 24. So 25 candidates per step.
        # Total steps ~ N/24. Total ops ~ N/24 * 25 * (bit count).
        
        uncovered_list = list(range(N))
        rng.shuffle(uncovered_list)
        
        while uncovered_list:
            if time.time() > deadline - 5:
                break
                
            # Pick a random uncovered word
            # To speed up, we can just pick the first one in the list that is actually uncovered
            # But maintaining the list is O(N) per removal.
            # Better: Just iterate through a shuffled list of all words, and if it's uncovered, process it.
            # Actually, the current approach in the seed was: pick random uncovered, find best coder.
            
            # Let's use a more efficient "random probe" strategy.
            # Pick a random index from the remaining uncovered set.
            # Since we don't have an efficient dynamic set, we'll just pick a random word from 0..N-1 
            # and check if it's uncovered. If so, process it.
            # This might miss words if we're not careful, but with a deadline, we just need a good solution.
            
            # Alternative: Keep a list of uncovered indices and pop randomly.
            if not uncovered_list:
                break
            
            # Randomly select an element from uncovered_list
            idx_in_list = rng.randint(0, len(uncovered_list) - 1)
            target_word = uncovered_list[idx_in_list]
            
            # Check if it's still uncovered (it should be, unless logic error)
            if (uncovered >> target_word) & 1 == 0:
                # It was covered by a previous choice? No, we remove from list when covered.
                # Wait, the seed code removed from a set. Here I use a list.
                # I need to ensure I don't pick covered words.
                # Let's stick to the seed's logic which used a set for uncovered.
                # But sets are slow to iterate.
                
                # Let's revert to the seed's logic but optimized.
                # The seed's logic:
                # 1. Pick random uncovered word.
                # 2. Get its coders (neighbors).
                # 3. Pick the coder that covers the most *other* uncovered words.
                # 4. Add to code, update uncovered mask.
                
                # The bottleneck is `uncovered_set.discard` and `rng.choice(list(uncovered_set))`.
                # `list(uncovered_set)` is O(N).
                
                # Let's use a different strategy:
                # Iterate through all words in a random order.
                # If word is uncovered, find best coder.
                # This is O(N) passes.
                
                # Actually, the most efficient greedy for this size is:
                # 1. Start with all uncovered.
                # 2. While uncovered:
                #    a. Pick a random uncovered word w.
                #    b. Find c in C(w) that maximizes |C(c) \ Covered|.
                #    c. Add c to Code.
                
                # To avoid O(N) list creation, we can just pick random indices from 0..N-1 
                # and skip if covered.
                pass 

        # The above thought process suggests the seed's implementation might be slow due to set/list conversions.
        # Let's implement a cleaner Greedy that doesn't convert sets to lists repeatedly.
        
        code = []
        code_set = set()
        uncovered = (1 << N) - 1
        
        # Pre-compute a random order of words to try as "targets"
        # But we need to pick *uncovered* ones.
        
        # Efficient approach:
        # 1. Create a list of all indices 0..N-1.
        # 2. Shuffle it.
        # 3. Iterate through it.
        # 4. If the word is uncovered, find the best codeword for it.
        # 5. Mark covered.
        
        word_order = list(range(N))
        rng.shuffle(word_order)
        
        for w in word_order:
            if time.time() > deadline - 5:
                break
            if (uncovered >> w) & 1 == 0:
                continue
                
            # Find best codeword c that covers w
            # Candidates are neighbors of w (within radius)
            candidates = get_coders_for_word(w)
            
            best_c = -1
            best_gain = -1
            
            for c in candidates:
                if c in code_set:
                    continue
                # Count new coverage
                new_cover = neighbor_masks[c] & uncovered
                gain = new_cover.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c != -1:
                code.append(best_c)
                code_set.add(best_c)
                uncovered &= ~neighbor_masks[best_c]
            
        return code

    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        code_set = set(code)
        # Use numpy for cover counts for speed
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            np.add.at(cover_counts, neighbor_lists[c], 1)
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                c = code[i]
                # Check if any word is uniquely covered by c
                # We need to check cover_counts for neighbors of c
                # If all neighbors have count > 1, c is redundant
                if np.all(cover_counts[neighbor_lists[c]] > 1):
                    code.pop(i)
                    code_set.discard(c)
                    np.add.at(cover_counts, neighbor_lists[c], -1)
                    improved = True
                    break
                    
        return code

    def two_opt_local_search(code, rng):
        """
        2-opt: Replace one codeword with another to enable removal of another, 
        or simply replace to increase efficiency (reduce size).
        Here we focus on size reduction: 
        Find c1 such that removing it leaves few uncovered words.
        Find c2 such that adding c2 covers all those uncovered words AND allows removal of c3?
        Actually, standard 1-swap (remove c1, add c2) doesn't reduce size.
        2-swap (remove c1, c2, add c3, c4) or (remove c1, add c2, remove c3) does.
        
        Let's implement "Remove 1, Add 1, Remove 1" (Net -1).
        """
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            np.add.at(cover_counts, neighbor_lists[c], 1)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                    
                c1 = code[i]
                # If we remove c1, which words become uncovered?
                # Words where cover_count == 1
                neighbors_c1 = neighbor_lists[c1]
                # We need to check which of these have count 1
                # Using numpy:
                mask_unique = cover_counts[neighbors_c1] == 1
                if not np.any(mask_unique):
                    # c1 is redundant, just remove it
                    code.pop(i)
                    code_set.discard(c1)
                    np.add.at(cover_counts, neighbors_c1, -1)
                    improved = True
                    break
                    
                newly_uncovered = [neighbors_c1[j] for j in np.where(mask_unique)[0]]
                
                if len(newly_uncovered) > 4:
                    continue # Too hard to cover with one codeword
                
                # Find a codeword c2 NOT in code that covers all newly_uncovered
                # Candidates: neighbors of any word in newly_uncovered
                candidates = set()
                for w in newly_uncovered:
                    candidates.update(get_coders_for_word(w))
                
                # Filter candidates
                valid_candidates = []
                for c2 in candidates:
                    if c2 in code_set:
                        continue
                    # Check if c2 covers all newly_uncovered
                    mask_c2 = neighbor_masks[c2]
                    covers_all = True
                    for w in newly_uncovered:
                        if (mask_c2 >> w) & 1 == 0:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c2)
                        
                if not valid_candidates:
                    continue
                
                # For each valid c2, check if adding it allows removal of some c3
                best_move = None
                best_c3 = -1
                
                for c2 in valid_candidates[:20]: # Limit candidates
                    # Simulate adding c2
                    # New cover counts: old + neighbor_masks[c2]
                    # We want to find c3 such that after removing c1 and adding c2, 
                    # c3 becomes redundant.
                    
                    # This is expensive to simulate fully.
                    # Approximation: 
                    # If c2 covers many words that were uniquely covered by c3, c3 might become redundant?
                    # No, c3 becomes redundant if ALL its words are covered by others.
                    
                    # Let's just try the swap: Remove c1, Add c2. 
                    # Then run a quick "remove redundant" check.
                    # If we can remove c3, we net -1.
                    
                    # To save time, we only try if c2 is "good" (covers many new words)
                    new_cov = (neighbor_masks[c2] & ~neighbor_masks[c1]).bit_count()
                    
                    # If we find a c2, we perform the swap and then try to remove something else.
                    # This is getting complex. Let's stick to a simpler 1-swap that allows 
                    # a subsequent removal in the next iteration of the loop.
                    
                    # Actually, just performing the swap (Remove c1, Add c2) doesn't change size.
                    # We need a net reduction.
                    # So we need: Remove c1, Add c2, Remove c3.
                    
                    # Let's identify potential c3s:
                    # c3 is a codeword (not c1) such that most of its words are covered by c2 or others.
                    
                    # Heuristic:
                    # Pick c2 that maximizes overlap with c1's "unique" words (must cover all).
                    # Then, for each other codeword c3, check if it becomes redundant.
                    
                    # This is O(N^2) in worst case.
                    # Given time limits, we might skip complex 2-swaps and rely on 
                    # multiple restarts of 1-swap + remove.
                    
                    pass # Placeholder for complex logic, might be too slow
                    
        return code

    def perturb(code, rng, strength=0.05):
        """Randomly swap some codewords to escape local minima"""
        code = list(code)
        if not code:
            return code
        
        code_set = set(code)
        n_swaps = max(1, int(len(code) * strength))
        for _ in range(n_swaps):
            if len(code) < 2:
                break
            i = rng.randint(0, len(code) - 1)
            old = code[i]
            code_set.discard(old)
            # Pick a random new codeword
            # To maintain quality, pick a neighbor of old?
            # Or just random.
            new_c = rng.randint(0, N - 1)
            while new_c in code_set:
                new_c = rng.randint(0, N - 1)
            code[i] = new_c
            code_set.add(new_c)
        
        return code

    def full_pipeline(seed, permutation=None):
        """Run the full optimization pipeline with a given seed"""
        rng = random.Random(seed)
        code = greedy_cover(rng, permutation)
        code = local_search_remove(code)
        # Skip complex two_opt for now to ensure speed, rely on ILS
        return code

    # Initial solution with multiple permutations to break symmetry
    best_code = []
    best_size = float('inf')
    
    # Generate a few permutations
    perms = []
    base_perm = list(range(n))
    for _ in range(5):
        p = base_perm.copy()
        random.shuffle(p)
        perms.append(p)
    perms.append(base_perm)
    
    for i, perm in enumerate(perms):
        if time.time() > deadline - 10:
            break
        code = full_pipeline(42 + i, perm)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code

    if not best_code:
        # Fallback
        best_code = full_pipeline(42, None)
        best_size = len(best_code)

    # ILS
    rng_ils = random.Random(12345)
    current_code = list(best_code)
    current_size = best_size
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Perturb
        perturbed = perturb(current_code, rng_ils, strength=0.05)
        
        # Local Search
        improved = local_search_remove(perturbed)
        
        # Accept if better
        if len(improved) < current_size:
            current_code = improved
            current_size = len(improved)
            if current_size < best_size:
                best_code = list(current_code)
                best_size = current_size
        elif len(improved) == current_size and rng_ils.random() < 0.2:
            current_code = improved
            
        # Occasional fresh restart
        if iteration % 10 == 0 and time.time() < deadline - 15:
            perm = list(range(n))
            random.shuffle(perm)
            fresh = full_pipeline(rng_ils.randint(1, 10000), perm)
            if len(fresh) < current_size:
                current_code = fresh
                current_size = len(fresh)
                if current_size < best_size:
                    best_code = list(current_code)
                    best_size = current_size

    # Final verification and fix
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            # Find codeword covering most uncovered
            # This is O(N) per step.
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = []
    for i in best_code:
        result.append(idx_to_word(i))
        
    return result

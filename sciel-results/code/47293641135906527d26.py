"""Construct a ternary covering code K_3(7,1) via randomized greedy + local search with bitmasks."""
import time
import random
import itertools

def construct(q, n, radius):
    # Deadline guard: evaluator kills at 150s wall-clock; keep ~10s margin.
    deadline = time.time() + 140.0
    total_words = q ** n  # 2187
    all_indices = list(range(total_words))

    # Precompute neighbor sets (Hamming distance <= radius) as Python ints (bitmasks)
    # For each word, compute its own index and all words at distance 1.
    # We'll use a list of ints where bit i is set if word i is in the ball.
    
    # Generate all words as tuples
    words = list(itertools.product(range(q), repeat=n))
    
    # Build index from word tuple to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls: for each index, a bitmask of all words within Hamming distance <= radius
    balls = []
    for i, w in enumerate(words):
        mask = 0
        # The word itself
        mask |= (1 << i)
        # All words at distance 1
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = new_val
                nw = tuple(nw)
                j = word_to_idx[nw]
                mask |= (1 << j)
        balls.append(mask)
    
    # Union of all balls = full space
    full_mask = (1 << total_words) - 1
    
    def is_covered(covered_mask):
        return (covered_mask & full_mask) == full_mask
    
    def try_remove_redundant(codeword_indices, covered_mask):
        """Try to remove redundant codewords. Returns (new_indices, new_covered_mask, removed_count)."""
        # For each codeword, check if removing it still leaves full coverage
        # A codeword c is redundant if for every word in its ball, that word is also in the ball of some other codeword.
        # Efficient check: compute the union of balls of all other codewords, and see if it covers the ball of c.
        removed = 0
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for idx in list(codeword_indices):
                # Compute union of balls of all other codewords
                other_union = 0
                for j in codeword_indices:
                    if j != idx:
                        other_union |= balls[j]
                # Check if ball of idx is subset of other_union
                if (balls[idx] & other_union) == balls[idx]:
                    codeword_indices.remove(idx)
                    covered_mask ^= (balls[idx] & ~(other_union))  # Actually recompute properly
                    # Recompute covered_mask
                    covered_mask = 0
                    for j in codeword_indices:
                        covered_mask |= balls[j]
                    removed += 1
                    changed = True
                    break
        return codeword_indices, covered_mask, removed
    
    def greedy_initial(seed):
        """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
        rng = random.Random(seed)
        covered = 0
        codewords = []
        remaining = all_indices[:]
        rng.shuffle(remaining)
        
        # For efficiency, we track uncovered mask
        uncovered = full_mask
        
        while uncovered != 0 and time.time() < deadline - 5.0:
            best_idx = -1
            best_new_cover = 0
            for idx in remaining:
                new_cover = (balls[idx] & uncovered).bit_count()
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_idx = idx
            if best_idx == -1 or best_new_cover == 0:
                break
            codewords.append(best_idx)
            uncovered &= ~balls[best_idx]
            remaining.remove(best_idx)
        
        return codewords
    
    def local_search(codewords, seed, time_limit):
        """Local search: try adding one codeword and removing up to two, or swapping."""
        rng = random.Random(seed)
        best_codewords = list(codewords)
        best_len = len(best_codewords)
        best_covered = 0
        for c in best_codewords:
            best_covered |= balls[c]
        
        start = time.time()
        while time.time() < time_limit:
            # Compute current covered mask
            cur_covered = 0
            for c in best_codewords:
                cur_covered |= balls[c]
            
            if not is_covered(cur_covered):
                # Not covered, need to add
                uncovered = full_mask & ~cur_covered
                # Find best addition
                best_add = -1
                best_gain = 0
                for idx in range(total_words):
                    if idx not in best_codewords:
                        gain = (balls[idx] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_add = idx
                if best_add >= 0:
                    best_codewords.append(best_add)
                continue
            
            # Try removals
            for idx in list(best_codewords):
                # Check if removing idx still covers
                other_union = 0
                for j in best_codewords:
                    if j != idx:
                        other_union |= balls[j]
                if (balls[idx] & other_union) == balls[idx]:
                    best_codewords.remove(idx)
                    best_len = len(best_codewords)
                    break
            
            # Try swaps: remove one, add one
            for idx in list(best_codewords):
                other_union = 0
                for j in best_codewords:
                    if j != idx:
                        other_union |= balls[j]
                # Words only covered by idx
                only_by_idx = balls[idx] & ~other_union
                # Find an addition that covers all of only_by_idx
                for add_idx in range(total_words):
                    if add_idx in best_codewords:
                        continue
                    if (balls[add_idx] & only_by_idx) == only_by_idx:
                        # Check if this swap reduces size (it doesn't, same size) 
                        # But might enable future removals. Only do if we can also remove another.
                        # For now, just accept the swap if it doesn't increase size
                        # Actually swap keeps same size, so only useful if it enables removal.
                        # Skip pure swaps for now to save time.
                        pass
            
            # Try add one, remove two
            if len(best_codewords) > 2:
                # This is expensive; skip for now
                pass
        
        return best_codewords
    
    # Strategy: multiple randomized greedy starts, then local search
    best_result = None
    best_size = float('inf')
    
    # First, try a single greedy for a quick baseline
    greedy_code = greedy_initial(42)
    if best_result is None or len(greedy_code) < best_size:
        best_result = list(greedy_code)
        best_size = len(best_result)
    
    # Multiple iterations with different seeds
    num_iterations = 30
    for it in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
        seed = 1000 + it * 7919
        # Randomized greedy: shuffle the order of consideration
        rng = random.Random(seed)
        covered = 0
        codewords = []
        uncovered = full_mask
        remaining = all_indices[:]
        rng.shuffle(remaining)
        
        while uncovered != 0 and time.time() < deadline - 15.0:
            best_idx = -1
            best_new_cover = 0
            # Sample a subset for speed in later iterations
            if it < 5:
                candidates = remaining
            else:
                sample_size = min(len(remaining), 200)
                candidates = rng.sample(remaining, sample_size)
            
            for idx in candidates:
                new_cover = (balls[idx] & uncovered).bit_count()
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_idx = idx
            if best_idx == -1 or best_new_cover == 0:
                break
            codewords.append(best_idx)
            uncovered &= ~balls[best_idx]
            remaining.remove(best_idx)
        
        if uncovered != 0:
            # Fallback: finish greedily
            remaining2 = [i for i in all_indices if i not in codewords]
            while uncovered != 0 and time.time() < deadline - 15.0:
                best_idx = -1
                best_new_cover = 0
                for idx in remaining2:
                    new_cover = (balls[idx] & uncovered).bit_count()
                    if new_cover > best_new_cover:
                        best_new_cover = new_cover
                        best_idx = idx
                if best_idx == -1 or best_new_cover == 0:
                    break
                codewords.append(best_idx)
                uncovered &= ~balls[best_idx]
                remaining2.remove(best_idx)
        
        # Local search
        time_limit = min(deadline - 15.0, time.time() + 3.0)
        codewords = local_search(codewords, seed, time_limit)
        
        # Verify coverage
        covered_mask = 0
        for c in codewords:
            covered_mask |= balls[c]
        if is_covered(covered_mask) and len(codewords) < best_size:
            best_result = list(codewords)
            best_size = len(best_result)
    
    # Final cleanup: try to remove any redundant codewords
    if best_result is not None and time.time() < deadline - 5.0:
        # Multiple passes of removal
        for _ in range(3):
            changed = False
            for idx in list(best_result):
                other_union = 0
                for j in best_result:
                    if j != idx:
                        other_union |= balls[j]
                if (balls[idx] & other_union) == balls[idx]:
                    best_result.remove(idx)
                    changed = True
            if not changed:
                break
    
    # Convert indices back to codeword lists
    return [words[i] for i in best_result]

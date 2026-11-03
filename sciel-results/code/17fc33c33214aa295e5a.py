import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n
    
    # Generate all words as integers
    # We use integer representation for efficiency
    # Word i is represented by its base-q digits
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance <= radius of i
    # For q=3, n=7, radius=1: 1 + 7*(3-1) = 15 neighbors
    
    neighbors = np.zeros((total_space, 1 + n * (q - 1)), dtype=np.int32)
    
    # Helper to convert index to word and vice versa is needed for neighbor generation
    # But we can generate neighbors directly from index
    
    def get_neighbors(idx):
        # Convert idx to word
        w = []
        tmp = idx
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        nbs = [idx]
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    # Convert back to index
                    n_idx = 0
                    for v in nw:
                        n_idx = n_idx * q + v
                    nbs.append(n_idx)
        
        return nbs

    # Precompute all neighbors
    # This might take a bit of time, but it's done once
    # 2187 * 15 conversions is trivial
    for i in range(total_space):
        nbs = get_neighbors(i)
        # nbs should have length 1 + n*(q-1)
        # If radius > 1, this logic needs to be expanded, but task is radius 1
        neighbors[i] = nbs

    # Greedy Set Cover with NumPy
    def greedy_cover(randomize=False):
        # covered: boolean array, True if word is covered
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        # Precompute neighbor lists as Python lists for faster iteration in gain calc?
        # Or use NumPy fancy indexing.
        
        # To speed up gain calculation:
        # gain[i] = sum(1 for j in neighbors[i] if not covered[j])
        # We can vectorize this:
        # gains = np.sum(covered[neighbors] == 0, axis=1)
        
        while not np.all(covered):
            # Calculate gains for all candidates
            # covered[neighbors] creates a (total_space, 15) array
            # This is 2187 * 15 * 4 bytes = 130KB per iteration.
            # 180 iterations -> 23MB of memory traffic. Fast.
            
            # We only care about candidates that are not already in code?
            # Actually, adding a codeword already in code is useless.
            # But greedy naturally picks new ones if they have higher gain.
            # However, a codeword in `code` covers its neighbors, so its gain for *new* coverage is 0.
            # So we don't need to explicitly exclude them, their gain will be 0.
            
            gains = np.sum(covered[neighbors] == 0, axis=1)
            
            if randomize:
                # Find max gain
                max_gain = np.max(gains)
                # Get indices with max gain
                candidates = np.where(gains == max_gain)[0]
                # Pick randomly
                if len(candidates) > 0:
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            code.append(best_idx)
            
            # Update covered
            covered[neighbors[best_idx]] = True
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = list(code)
        
        # Build cover_count
        # cover_count[w] = number of codewords covering w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Efficiently build cover_count using bincount or scatter
        # Flatten neighbors of all codewords
        # neighbors is (N, 15). current_code is list of indices.
        # We want to count how many times each word index appears in neighbors[current_code]
        
        if len(current_code) == 0:
            return current_code
            
        # Flatten the neighbors of the selected codewords
        # shape: (len(current_code) * 15)
        flat_neighbors = neighbors[current_code].flatten()
        
        # Count occurrences
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                
                # Check if all words covered by idx are covered by at least one other codeword
                # i.e., cover_count[j] > 1 for all j in neighbors[idx]
                # If min(cover_count[neighbors[idx]]) > 1, we can remove idx
                
                min_count = np.min(cover_count[neighbors[idx]])
                
                if min_count > 1:
                    # Remove idx
                    # Decrement cover_count for its neighbors
                    cover_count[neighbors[idx]] -= 1
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    def local_search_swap(code):
        """
        Try to swap codeword i with j if it leads to a smaller set after reduction.
        Heuristic:
        1. Remove i.
        2. Find words that become uncovered (count drops to 0).
        3. Find j that covers the most of these uncovered words.
        4. Add j.
        5. Reduce.
        6. If size < original, keep.
        """
        if not code:
            return code
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        # Build cover_count
        if best_len == 0:
            return best_code
            
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        # We try a limited number of swaps to respect time
        # Number of swaps = len(code) * attempts_per_codeword
        attempts_per_codeword = 5
        
        for _ in range(len(current_code) * attempts_per_codeword):
            if time.time() > deadline:
                break
                
            # Pick a random codeword to swap out
            i = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i]
            
            # Tentatively remove idx_out
            cover_count[neighbors[idx_out]] -= 1
            current_code.pop(i)
            
            # Find uncovered words
            uncovered_mask = (cover_count == 0)
            uncovered_words = np.where(uncovered_mask)[0]
            
            if len(uncovered_words) == 0:
                # It was redundant, but local_search_reduce should have caught it.
                # If it didn't, maybe due to order. Just restore.
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Find best replacement
            # We want j that maximizes overlap with uncovered_words
            # gain[j] = sum(1 for w in neighbors[j] if uncovered_mask[w])
            
            # Vectorized:
            # neighbors is (N, 15). uncovered_mask is (N,).
            # We want sum of uncovered_mask[neighbors[j]] for each j.
            # This is equivalent to:
            # gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude current codewords?
            # Ideally yes, but if a current codeword has high gain, it means it covers many uncovered words.
            # But it's already in the set (except idx_out which we removed).
            # If we pick a codeword that is still in current_code, we are effectively just "keeping" it.
            # That's fine, it's a no-op swap.
            # But we want to find a *new* codeword or a different one.
            # Let's just pick the argmax.
            
            best_j = np.argmax(gains)
            
            # If the best gain is 0, no single codeword can cover the uncovered words?
            # That implies the set is not a cover, which shouldn't happen if we start from a valid cover
            # and remove one word. The uncovered words must be coverable by some other word.
            # Wait, if we remove idx_out, the uncovered words are those ONLY covered by idx_out.
            # They must be covered by some other word in the full space, but maybe not in the current set.
            # So gains > 0 should hold for at least one word in the full space.
            
            # Add best_j
            current_code.append(best_j)
            cover_count[neighbors[best_j]] += 1
            
            # Now run reduction to clean up
            # We need to rebuild cover_count or update it?
            # We updated it for the swap. Now we can run a quick reduction pass.
            
            # To save time, we can just check if the size is smaller after a simple reduction.
            # Let's run local_search_reduce on the new set.
            # But local_search_reduce rebuilds cover_count from scratch.
            # That's expensive if done for every swap.
            
            # Alternative: Just check if the new set is valid and smaller?
            # It's always valid if we added a word that covers the gaps.
            # Is it smaller? We removed 1, added 1. Size is same.
            # So we MUST run reduction to see if it gets smaller.
            
            # Optimized Reduction:
            # We already have cover_count.
            # We can try to remove redundant words from the current_code.
            
            new_code = list(current_code)
            # We need to be careful with cover_count consistency.
            # Let's just call a lightweight reduction.
            
            # Since we only added one word, maybe only words covered by best_j or idx_out are affected?
            # No, removing idx_out might make some words have count 1, which were previously 2.
            # Adding best_j might make some words count 2, which were previously 1.
            
            # Let's just run the full local_search_reduce but with a time check.
            # To avoid the overhead of bincount, we can pass the current cover_count?
            # Let's modify local_search_reduce to accept an optional cover_count.
            
            # For now, let's just do a quick check:
            # If len(new_code) < best_len:
            #    best_len = len(new_code)
            #    best_code = new_code
            # But we haven't reduced it yet.
            
            # Let's run a single pass of reduction on the modified cover_count.
            
            # Re-verify cover_count is correct.
            # We decremented for idx_out, incremented for best_j.
            
            # Try to remove redundant words
            improved = True
            while improved:
                improved = False
                random.shuffle(new_code)
                for k in range(len(new_code)):
                    idx = new_code[k]
                    if np.min(cover_count[neighbors[idx]]) > 1:
                        cover_count[neighbors[idx]] -= 1
                        new_code.pop(k)
                        improved = True
                        break
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                # Update cover_count for best_code?
                # We lost track of the exact cover_count for best_code if we modified new_code.
                # We should rebuild cover_count for best_code if we accept it?
                # Or just keep the current one and continue searching from this state?
                # It's safer to rebuild if we accept, but expensive.
                # Let's just continue the loop with the current state (which is best_code state).
                # But the `cover_count` variable now corresponds to `new_code`.
                # So we can continue using it.
                current_code = new_code
            else:
                # Revert
                # This is tricky because we modified cover_count and current_code.
                # We should have worked on a copy.
                # Let's restructure to work on copies.
                pass
                
        # This implementation of swap is flawed due to state mutation complexity.
        # Let's simplify: Just do random restarts of Greedy + Reduce.
        # That is robust and often effective.
        
        return best_code

    # Strategy:
    # 1. Run deterministic greedy + reduce.
    # 2. Run randomized greedy + reduce multiple times.
    # 3. Keep the best.

    best_code = greedy_cover(randomize=False)
    best_code = local_search_reduce(best_code)
    best_len = len(best_code)
    
    # Randomized restarts
    # How many can we afford?
    # Greedy is ~0.1s? Reduce is ~0.1s?
    # Let's assume 0.2s per iteration.
    # 100s / 0.2s = 500 iterations.
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        cand_code = greedy_cover(randomize=True)
        cand_code = local_search_reduce(cand_code)
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code
            
    # Convert indices back to words
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code]

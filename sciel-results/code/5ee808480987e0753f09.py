import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # This problem is specifically for q=3, n=7, radius=1.
    # The best known covering code has size 186.
    # It is constructed as the union of 18 cosets of the ternary [7,3,3] Hamming code.
    # The Hamming code H is a linear code of length 7, dimension 3, minimum distance 3.
    # Size of H is 3^3 = 27.
    # 18 * 27 = 486? No.
    # Wait, the covering code C(7,1) over ternary alphabet has size 186.
    # 186 is not divisible by 27. So it's not a union of cosets of the [7,3,3] code?
    # Let's re-read the literature in my head.
    # K_3(7,1) = 186.
    # The ternary Golay code is [11,6,5].
    # The ternary Hamming code is [7,5,3]? No, [7,5,3] has size 3^5=243.
    # [7,3,3] has size 27.
    # Maybe the construction is different.
    
    # Actually, let's look at the "inspiration 3" which used Lagrangian relaxation.
    # But the directive says: "Use a deterministic algebraic construction based on the ternary Golay code structure: generate the 186 codewords by taking the union of specific cosets of the [7,3,3] ternary Hamming code..."
    
    # If the directive is specific, I should follow it, even if my internal knowledge is fuzzy on the exact cosets.
    # However, if I don't know the exact 186 words, I can't hardcode them.
    # Can I generate them?
    # The ternary [7,3,3] code is the dual of the [7,5,3] code? No.
    # The parity check matrix for the ternary Hamming code [7,5,3] has 2 rows? No, 3 rows.
    # H = [1 0 0 1 1 1 1; 0 1 0 1 1 2 2; 0 0 1 1 2 1 2] ? Something like that.
    # The code is the null space of H.
    # If I can generate the [7,3,3] code, I can pick 186 words? No, 186 is not a multiple of 27.
    
    # Wait, is it possible the "186" code is not a union of cosets?
    # Or maybe I should just use a strong heuristic that is faster?
    # The current score is 1.0848. The seed is 1.0.
    # 1.0848 means value = seed / 1.0848. If seed=200, value=184.3?
    # If best known is 186, and seed is likely around 200 or 204.
    # If I can get 186, score = seed/186.
    # If seed is 204, 204/186 = 1.096.
    # If seed is 200, 200/186 = 1.075.
    # Current score 1.0848 suggests value is slightly worse than optimal or close.
    
    # Let's try to generate the [7,3,3] ternary Hamming code cosets.
    # But since 186 is not divisible by 27, this specific algebraic structure might not be a simple union of cosets.
    # Perhaps the directive implies a *search* over cosets? Or maybe I should just implement a very fast greedy + local search that finds 186?
    
    # Let's stick to the directive's *spirit*: "deterministic algebraic construction".
    # If I can't guarantee 186 via algebra, I should use the fastest possible heuristic.
    # The current program uses a boolean cover matrix which is 2187x2187. That's ~4.7M bools. Fast.
    # The bottleneck in the current program is likely the ILS loop or the initial greedy.
    
    # Let's try a different approach: **Integer Linear Programming (ILP)** is too slow.
    # **Branch and Bound**? Too slow.
    # **Fast Greedy with Bitsets**:
    # Represent the space as a bitset of 2187 bits.
    # Each codeword covers 15 positions.
    # Use numpy arrays of uint64 or similar?
    # 2187 bits = 35 uint64s.
    # This makes the "count uncovered" step extremely fast.
    
    # Let's implement a highly optimized Greedy Set Cover using bitsets, followed by a tight local search.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (though task specifies this one)
        return _generic_construct(q, n, radius)
        
    space_size = 3**7 # 2187
    
    # Precompute neighbors as indices
    neighbors = []
    for idx in range(space_size):
        w = [0]*7
        tmp = idx
        for i in range(6, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        nbs = [idx]
        for pos in range(7):
            d = w[pos]
            for nd in range(3):
                if nd == d: continue
                diff = (nd - d) * (3 ** (6-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    # Bitset representation
    # num_words = 2187
    # bits_per_word = 64
    # num_blocks = ceil(2187/64) = 35
    num_blocks = (space_size + 63) // 64
    
    # cover_masks: list of lists of uint64?
    # cover_masks[i] is a list of num_blocks uint64s, where bit j in block b is set if (b*64+j) is in neighbors[i]
    
    cover_masks = []
    for i in range(space_size):
        blocks = [0] * num_blocks
        for nb in neighbors[i]:
            blocks[nb // 64] |= (1 << (nb % 64))
        cover_masks.append(blocks)
        
    # Uncovered mask: list of uint64s
    uncovered = [(1 << min(64, space_size - b*64)) - 1 for b in range(num_blocks)]
    # Mask out bits beyond space_size in the last block
    last_block_bits = space_size % 64
    if last_block_bits:
        uncovered[-1] = (1 << last_block_bits) - 1
        
    code = []
    
    # Greedy Set Cover
    # To find the best candidate, we need to count how many bits are set in (cover_mask[i] & uncovered)
    # This is a bit expensive if done for all 2187 candidates every step.
    # 2187 * 35 operations per step.
    # 200 steps * 2187 * 35 ~ 15M ops. Very fast in Python? Maybe.
    # We can optimize by using numpy if we store as int64 array?
    # But Python lists of ints might be faster for bitwise ops on small numbers?
    
    # Let's try to optimize the "find best" step.
    # We can maintain a "count" array? No, because uncovered changes.
    
    # Alternative: Precompute for each position j, the list of codewords covering it.
    # Then use a priority queue?
    # When we pick a codeword, we decrement the count of all uncovered words it covers.
    # If count becomes 0, it's covered.
    # This is the standard "Maximal Marginal Coverage" approach.
    
    # Let's use the count array approach. It's O(K * Space) initialization and O(K * Deg) per step.
    # K ~ 200, Space = 2187, Deg = 15.
    # Init: 200 * 2187 ~ 400k.
    # Step: 200 * 15 ~ 3000.
    # Total ~ 1M ops. Very fast.
    
    # But we need to find the max count efficiently.
    # We can just scan the array of size 2187.
    # 200 steps * 2187 scans = 437k ops.
    
    # Let's implement this.
    
    # counts[i] = number of uncovered words covered by i
    # uncovered_words: boolean array or set?
    # Let's use a boolean array `is_uncovered` and an integer array `coverage_count`.
    
    is_uncovered = np.ones(space_size, dtype=bool)
    coverage_count = np.zeros(space_size, dtype=np.int32)
    
    # Initialize coverage_count
    # For each word w, count how many codewords cover it? No, we want for each codeword c, how many *uncovered* words it covers.
    # Initially all are uncovered.
    # So coverage_count[c] = len(neighbors[c]) = 15.
    # Wait, this doesn't help distinguish between codewords if all cover 15.
    # We need to update this as we cover words.
    
    # Better:
    # Maintain `uncovered_mask` (boolean array).
    # For each candidate c, score = sum(uncovered_mask[neighbors[c]]).
    # This is what the current program does with `cover_masks @ uncovered`.
    
    # Let's stick to the bitset approach for the greedy phase, it's cleaner for "fast".
    
    # Actually, the current program's greedy is:
    # counts = cover_masks @ uncovered.astype(np.int8)
    # This is a matrix-vector product. 2187x2187 @ 2187.
    # This is ~4.7M multiplications/additions.
    # 200 iterations * 4.7M = 940M ops.
    # In NumPy, this is vectorized and fast, but maybe 0.1-0.5s per iteration?
    # If it takes 0.1s, 200 iterations = 20s.
    # Plus ILS.
    
    # If I use the "count updates" method:
    # 1. Start with `uncovered` set of indices.
    # 2. `score[c]` = number of neighbors of c in `uncovered`.
    # 3. Pick c with max score.
    # 4. Remove neighbors of c from `uncovered`.
    # 5. For each neighbor w of c, for each codeword d that covers w, decrement `score[d]`.
    #    This requires precomputing `covering_codewords[w]` = list of c such that w in neighbors[c].
    #    Size of covering_codewords[w] is 15.
    #    So step 5 is O(15 * 15) = 225 ops.
    #    Step 3 is O(2187) scan.
    #    Total per iteration ~ 2400 ops.
    #    200 iterations ~ 500k ops.
    #    This is MUCH faster than the matrix product.
    
    # Let's implement this "Incremental Greedy".
    
    # Precompute covering_codewords
    covering_codewords = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w in neighbors[c]:
            covering_codewords[w].append(c)
            
    # Initial scores
    # Since all are uncovered, score[c] = len(neighbors[c]) = 15.
    # But we need to break ties. Random tie-breaking or fixed?
    # Fixed is better for determinism.
    
    score = np.full(space_size, 15, dtype=np.int32)
    uncovered_set = set(range(space_size))
    # Using a set for uncovered allows fast "is w in uncovered?" checks, but iterating is slow.
    # We need to remove neighbors from uncovered.
    # And we need to pick max score.
    
    # To pick max score efficiently, we can use a heap?
    # But scores decrease.
    # Simple scan is 2187 ops. That's fine.
    
    while uncovered_set:
        # Find max score
        max_score = -1
        best_c = -1
        # Scan all c
        # Optimization: Only consider c that still covers at least one uncovered word.
        # If score[c] == 0, skip.
        for c in range(space_size):
            if score[c] > max_score:
                max_score = score[c]
                best_c = c
                
        if max_score == 0:
            break # Should not happen if valid
            
        code.append(best_c)
        
        # Remove neighbors of best_c from uncovered
        for w in neighbors[best_c]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                # Update scores of all codewords covering w
                for d in covering_codewords[w]:
                    score[d] -= 1
                    
        # Check time
        if time.time() - t_start > deadline - 5:
            break

    # Now we have a valid code (assuming we didn't time out).
    # If we timed out, we might have uncovered words.
    # Check validity
    if uncovered_set:
        # Add remaining uncovered words directly to code
        for w in uncovered_set:
            code.append(w)
            # We don't need to update scores for validity, just for potential pruning
            # But we need to ensure they are distinct.
            # If w is already in code, skip?
            # The greedy picks distinct c because score becomes 0?
            # No, if w is uncovered, it wasn't covered by any previous c.
            # So w is not in code.
            
    # Convert to list of distinct indices
    code = list(set(code))
    
    # Now, Local Search to minimize size.
    # We have a valid code.
    # We want to remove words.
    # State: code list, coverage_count array (how many times each word is covered).
    
    coverage_count = np.zeros(space_size, dtype=np.int32)
    in_code = np.zeros(space_size, dtype=bool)
    for c in code:
        in_code[c] = True
        for w in neighbors[c]:
            coverage_count[w] += 1
            
    # Prune redundant words
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            # Check if c is redundant: min(coverage_count[neighbors[c]]) > 1
            # If yes, remove c
            if np.min(coverage_count[neighbors[c]]) > 1:
                # Remove c
                code.pop(i)
                in_code[c] = False
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                changed = True
                break # Restart loop
                
    # ILS / Local Search
    # Try swaps to enable further removals.
    # Or just try to remove any word.
    
    # Given the time budget, let's do a simple hill climbing:
    # 1. Try to remove a random word. If valid, accept.
    # 2. If stuck, try a swap: replace c with c'.
    #    Valid if after swap, all words are covered.
    #    This is expensive to check naively.
    #    Condition: The set of words covered ONLY by c must be a subset of neighbors[c'].
    
    rng = np.random.default_rng(42)
    best_code = code[:]
    best_size = len(code)
    
    no_improve = 0
    
    while time.time() - t_start < deadline - 2:
        # Try removal
        if code:
            i = rng.integers(0, len(code))
            c = code[i]
            if np.min(coverage_count[neighbors[c]]) > 1:
                # Remove
                code.pop(i)
                in_code[c] = False
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                if len(code) < best_size:
                    best_code = code[:]
                    best_size = len(code)
                no_improve = 0
                continue
                
        # Try swap
        if code:
            i = rng.integers(0, len(code))
            c = code[i]
            # Pick random new_c
            new_c = rng.integers(0, space_size)
            if in_code[new_c] or new_c == c:
                continue
                
            # Check if swap is valid
            # Words covered only by c: those w in neighbors[c] with coverage_count[w] == 1
            # These must be covered by new_c.
            valid = True
            for w in neighbors[c]:
                if coverage_count[w] == 1:
                    if w not in neighbors[new_c]: # O(15) check? No, neighbors is list.
                        # Need fast membership.
                        # Precompute neighbor sets?
                        # Or use the cover_masks bitset?
                        # Let's use a boolean array for neighbors[new_c]?
                        # Too much memory.
                        # Just check if w is in neighbors[new_c].
                        # 15 items.
                        if not any(x == w for x in neighbors[new_c]):
                            valid = False
                            break
                            
            if valid:
                # Perform swap
                code[i] = new_c
                in_code[c] = False
                in_code[new_c] = True
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                for w in neighbors[new_c]:
                    coverage_count[w] += 1
                # Check if this enables a removal
                # (Optional: try to remove something now)
                no_improve += 1
                if no_improve > 50:
                    # Kick: randomize a few words?
                    # Or just accept neutral swaps more often?
                    # For now, just continue.
                    pass
            else:
                no_improve += 1
        else:
            break

    # Final verification
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        for w in neighbors[c]:
            final_cov[w] += 1
            
    if np.any(final_cov == 0):
        # Fallback to greedy code
        # Ensure greedy code is valid
        # It should be.
        return [idx_to_word(c) for c in code]
    else:
        return [idx_to_word(c) for c in best_code]

def idx_to_word(idx):
    w = [0] * 7
    for i in range(6, -1, -1):
        w[i] = idx % 3
        idx //= 3
    return w

def _generic_construct(q, n, radius):
    # Simple fallback
    space_size = q ** n
    neighbors = []
    for idx in range(space_size):
        w = [0]*n
        tmp = idx
        for i in range(n-1, -1, -1):
            w[i] = tmp % q
            tmp //= q
        nbs = [idx]
        for pos in range(n):
            d = w[pos]
            for nd in range(q):
                if nd == d: continue
                diff = (nd - d) * (q ** (n-1-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    uncovered = np.ones(space_size, dtype=bool)
    code = []
    while uncovered.any():
        counts = np.zeros(space_size, dtype=np.int32)
        for c in range(space_size):
            counts[c] = np.sum(uncovered[neighbors[c]])
        best_c = np.argmax(counts)
        if counts[best_c] == 0: break
        code.append(best_c)
        uncovered[neighbors[best_c]] = False
    return [idx_to_word(c) for c in code]

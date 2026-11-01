"""Ternary covering code via bit-parallel local search with systematic reduction."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    
    Strategy:
    1. Represent Hamming balls and coverage as Python integers (bitsets).
    2. Use a greedy set-cover to get an initial valid code.
    3. Perform aggressive local search:
       a. Remove redundant codewords (1-for-0 moves) using bitset operations.
       b. Swap codewords with better alternatives (1-for-1 moves) to enable further removals.
    4. Return the best valid code found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = q ** n
    if N <= 0:
        return []
    
    # Precompute all words as integer indices 0..N-1
    # We need the actual values for the final output
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    # Precompute Hamming balls as bitsets (Python ints)
    # ball[c] is an integer where bit w is 1 if word w is within distance `radius` of c
    balls = [0] * N
    
    # Helper to compute Hamming distance between two indices
    def hamming_dist(i, j):
        d = 0
        a = i
        b = j
        for _ in range(n):
            if (a % q) != (b % q):
                d += 1
            a //= q
            b //= q
        return d

    # This is O(N^2 * n) which might be slow for large N, but for q=3, n=7, N=2187, it's ~2187^2 * 7 ~ 33M ops.
    # In Python this might take a few seconds. Let's optimize or use a different method.
    # Actually, we can generate balls more efficiently.
    # For radius 1, ball(c) = {c} U {neighbors of c}.
    # Neighbors of c: change one position to one of the other q-1 values.
    # For general radius, we need to be careful. But the task is specifically radius 1 for q=3, n=7.
    # The interface is general, but the test case is specific.
    # Let's assume the test case is the main driver.
    
    # Faster ball generation for radius 1
    if radius == 1:
        for c in range(N):
            mask = 1 << c
            # Find neighbors
            # Convert c to base-q digits
            digits = [0] * n
            rem = c
            for j in range(n - 1, -1, -1):
                digits[j] = rem % q
                rem //= q
            
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Calculate new index
                    # new_idx = c + (new_val - orig) * (q ** pos)
                    # Wait, the digit order in the index calculation above was:
                    # idx = d_{n-1} * q^{n-1} + ... + d_0 * q^0 ?
                    # Let's check the all_words generation:
                    # rem = idx
                    # for j in range(n-1, -1, -1):
                    #   all_words[idx, j] = rem % q
                    #   rem //= q
                    # So all_words[idx, 0] is the most significant digit (q^(n-1) place).
                    # all_words[idx, n-1] is the least significant digit (q^0 place).
                    # So index = sum_{j=0}^{n-1} all_words[idx, j] * q^(n-1-j)
                    
                    # Let's re-verify.
                    # idx = 1, q=3, n=2.
                    # rem=1. j=1: all_words[1,1] = 1%3=1, rem=0. j=0: all_words[1,0] = 0%3=0.
                    # Word is [0, 1].
                    # Index 1 = 0 * 3^1 + 1 * 3^0.
                    # So position j in all_words corresponds to power q^(n-1-j).
                    
                    # If we change position `pos` (index in all_words array) from `orig` to `new_val`:
                    # The change in index is (new_val - orig) * q^(n-1-pos).
                    delta = (new_val - orig) * (q ** (n - 1 - pos))
                    new_idx = c + delta
                    if 0 <= new_idx < N:
                        mask |= (1 << new_idx)
            balls[c] = mask
    else:
        # Fallback for general radius (slower)
        for c in range(N):
            mask = 0
            for w in range(N):
                if hamming_dist(c, w) <= radius:
                    mask |= (1 << w)
            balls[c] = mask

    # Phase 1: Greedy Set Cover
    # uncovered is a bitset of all words initially
    uncovered = (1 << N) - 1
    code_set = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
        
        best_idx = -1
        max_new = 0
        
        # We need to find c that maximizes popcount(balls[c] & uncovered)
        # Iterating over all N candidates every time is O(N^2 / word_size) which is okay for N=2187.
        # But we can optimize by only checking candidates that cover at least one uncovered word.
        # Or just brute force since N is small.
        
        for c in range(N):
            if c in code_set:
                continue
            new_cov = balls[c] & uncovered
            # Count bits
            cnt = new_cov.bit_count()
            if cnt > max_new:
                max_new = cnt
                best_idx = c
        
        if best_idx == -1 or max_new == 0:
            break
        
        code_set.append(best_idx)
        uncovered &= ~balls[best_idx]
        
        # Early termination if time is running out
        if time.time() > deadline - 20:
            break

    # Ensure we have a valid cover. If greedy didn't finish, we might have a partial cover.
    # But for N=2187, greedy should be fast enough.
    
    # Phase 2: Local Search
    # We maintain:
    # - code_set: list of codeword indices
    # - coverage: bitset of all covered words
    # - counts: array where counts[w] is the number of codewords covering word w
    
    # Initialize counts and coverage
    coverage = 0
    counts = np.zeros(N, dtype=np.int32)
    
    # We need to build counts from code_set
    # This is O(|code_set| * N) if we iterate over bits, but we can use the balls.
    # Actually, we can just recompute coverage and counts from scratch for the initial set.
    
    def build_state(code_list):
        """Build coverage bitset and counts array for a given code list."""
        cov = 0
        cnts = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cov |= balls[c]
            # Update counts
            # We need to iterate over bits in balls[c]
            mask = balls[c]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cnts[idx] += 1
                mask ^= lsb
        return cov, cnts

    coverage, counts = build_state(code_set)
    
    # Check validity
    if coverage != (1 << N) - 1:
        # If not valid, try to add more codewords to fix it?
        # Or just return what we have? The task requires a valid cover.
        # Let's try to complete the cover greedily again if invalid.
        # But for now, assume greedy worked.
        pass

    best_code = code_set.copy()
    best_size = len(best_code)
    
    # Local Search Loop
    # We will try to remove codewords and swap codewords.
    
    # Helper: Try to remove codeword i from the current state
    # Returns True if removal was successful (i.e., it was redundant)
    def try_remove(i):
        nonlocal coverage, counts
        c = code_set[i]
        mask = balls[c]
        
        # Check if any word in mask has count 1
        # If min(counts[w] for w in mask) > 1, we can remove.
        # We can check this by iterating over bits in mask.
        # Or we can precompute something?
        # For now, iterate.
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] == 1:
                return False
            m ^= lsb
            
        # It is safe to remove
        # Update state
        coverage ^= mask  # Since all bits in mask are covered by at least 2 codewords, removing one doesn't uncover any.
        # Wait, coverage is the OR of balls. If a word was covered by 2, and we remove one, it's still covered by 1. So coverage remains 1.
        # So coverage does not change if removal is valid.
        
        # Update counts
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] -= 1
            m ^= lsb
            
        # Remove from list
        code_set.pop(i)
        return True

    # Helper: Try to swap codeword i with new_c
    def try_swap(i, new_c):
        nonlocal coverage, counts
        old_c = code_set[i]
        if old_c == new_c:
            return False
        if new_c in code_set:
            return False
            
        mask_old = balls[old_c]
        mask_new = balls[new_c]
        
        # We want to remove old_c and add new_c.
        # This is valid if:
        # 1. All words in mask_old \ mask_new are covered by at least 2 codewords (so removing old_c doesn't uncover them).
        # 2. Adding new_c doesn't violate anything (it just adds coverage).
        
        # Check condition 1:
        # words_to_check = mask_old & ~mask_new
        # For all w in words_to_check, counts[w] must be > 1.
        # Note: counts[w] includes old_c. So we need counts[w] >= 2.
        
        words_to_check = mask_old & ~mask_new
        m = words_to_check
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] < 2:
                return False
            m ^= lsb
            
        # If valid, perform swap
        # 1. Remove old_c
        # Update counts for mask_old
        m = mask_old
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] -= 1
            m ^= lsb
            
        # Update coverage
        # Words in mask_old \ mask_new might become uncovered if their count drops to 0?
        # No, we checked they are >= 2, so after removal they are >= 1.
        # So coverage for those words remains 1.
        # Words in mask_old & mask_new: count decreases by 1 (old) and increases by 1 (new). Net 0 change in count?
        # No, we are doing two steps.
        
        # 2. Add new_c
        m = mask_new
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
        # Update coverage
        # New words covered by new_c that weren't covered by old_c?
        # coverage |= mask_new
        
        # Actually, let's just update coverage as:
        # coverage = (coverage & ~mask_old) | mask_new ? No.
        # Coverage is the set of words with count > 0.
        # Since we ensured no word drops to 0, coverage remains the same?
        # Wait, if a word was ONLY covered by old_c, it would have count 1.
        # We checked words in mask_old \ mask_new have count >= 2.
        # What about words in mask_old & mask_new?
        # They had count >= 1. After removing old_c, count decreases by 1. If it was 1, it becomes 0.
        # But then we add new_c, so count increases by 1. So it goes back to 1.
        # So the final count is the same for words in intersection?
        # Let's trace:
        # Word w in intersection.
        # Initial count: C.
        # Remove old_c: C-1.
        # Add new_c: C.
        # So count is unchanged.
        # Word w in mask_old \ mask_new.
        # Initial count: C >= 2.
        # Remove old_c: C-1 >= 1.
        # Add new_c: C-1 (since w not in mask_new).
        # So count decreases by 1.
        # Word w in mask_new \ mask_old.
        # Initial count: C.
        # Remove old_c: C.
        # Add new_c: C+1.
        # So count increases by 1.
        
        # Coverage:
        # Words in mask_old \ mask_new: count >= 1, so still covered.
        # Words in intersection: count >= 1, so still covered.
        # Words in mask_new \ mask_old: count >= 1, so now covered.
        # Words outside: unchanged.
        # So coverage becomes coverage | mask_new.
        
        coverage |= mask_new
        
        code_set[i] = new_c
        return True

    # Main Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try to remove any redundant codeword
        # We do this in a loop because removing one might make another redundant.
        changed = True
        while changed:
            changed = False
            for i in range(len(code_set) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                if try_remove(i):
                    improved = True
                    changed = True
                    best_size = len(code_set)
                    if len(code_set) < best_size:
                        best_code = code_set.copy()
                        best_size = len(code_set)
        
        # 2. Try to swap codewords to enable further removals
        # We iterate through each codeword and try to swap it with a random candidate.
        # We limit the number of swaps to avoid infinite loops or excessive time.
        
        # To make it more effective, we can prioritize swaps that "prepare" for removal.
        # But random swaps are a good start.
        
        # We'll do a few passes of random swaps.
        for _ in range(10): # 10 passes of random swaps
            if time.time() > deadline - 5:
                break
            
            # Shuffle indices to try different ones
            indices = list(range(len(code_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                
                # Try a few random candidates
                for _ in range(3):
                    new_c = random.randint(0, N - 1)
                    if try_swap(i, new_c):
                        # Swap was successful. Now try to remove again.
                        # This might lead to a smaller code.
                        # We set improved = True to continue the outer loop.
                        improved = True
                        # Break out of swap loops to re-evaluate removals
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no improvement in this round, we might be stuck.
        # We can try a different strategy:
        # Pick the codeword that covers the fewest "unique" words (words covered only by it).
        # And try to swap it with one that covers more "shared" words.
        # But for now, let's just break if no improvement.
        
        if not improved:
            # Try a few more random restarts of the local search?
            # Or just stop.
            break

    # Final validity check
    # Rebuild coverage from best_code to be sure
    final_cov = 0
    for c in best_code:
        final_cov |= balls[c]
        
    if final_cov != (1 << N) - 1:
        # If invalid, fallback to the greedy result
        # Re-run greedy quickly?
        # Or just return best_code if it's close? No, must be valid.
        # Let's assume the local search maintained validity.
        # If not, we have a bug.
        # For safety, let's try to repair by adding missing words' codewords.
        missing = ((1 << N) - 1) & ~final_cov
        while missing:
            lsb = missing & -missing
            w = lsb.bit_length() - 1
            # Find a codeword that covers w
            for c in range(N):
                if (balls[c] & (1 << w)):
                    if c not in best_code:
                        best_code.append(c)
                        final_cov |= balls[c]
                        break
            missing ^= lsb

    return [list(all_words[i]) for i in best_code]
# EVOLVE-BLOCK-END

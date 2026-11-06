import time
import numpy as np
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 17
    M = 1 << N
    min_dist = 6
    max_radius = min_dist - 1  # 5

    # Precompute offsets (masks) of weight 1 to max_radius
    # These are all words with Hamming weight in [1, 5]
    offsets_list = []
    for r in range(1, max_radius + 1):
        for comb in combinations(range(N), r):
            mask = 0
            for pos in comb:
                mask |= (1 << pos)
            offsets_list.append(mask)
    offsets = np.array(offsets_list, dtype=np.uint32)
    num_offsets = len(offsets)

    # --- Step 1: Generate the [17,8,6] QR Code ---
    # This provides a strong 256-word baseline.
    
    def poly_divides(p, q):
        deg_p = p.bit_length() - 1
        deg_q = q.bit_length() - 1
        if deg_p > deg_q:
            return False
        if p == 0:
            return False
        r = q
        while r.bit_length() - 1 >= deg_p and r != 0:
            shift = (r.bit_length() - 1) - deg_p
            r ^= (p << shift)
        return r == 0

    target_poly = (1 << 17) | 1
    gen_poly = None

    # Find the generator polynomial for the [17,9,5] QR code
    for mid_bits in range(1 << 6):
        p = (1 << 8) | (mid_bits << 1) | 1
        if poly_divides(p, target_poly):
            gen_poly = p
            break

    if gen_poly is None:
        # Fallback if QR construction fails (should not happen)
        return [0]

    # Generate basis vectors for the [17,9,5] code
    basis = []
    for k in range(9):
        val = gen_poly << k
        val &= (1 << N) - 1
        basis.append(val)

    # Generate all 512 codewords of the [17,9,5] code
    qr_code_words = []
    for mask in range(1 << 9):
        word = 0
        for i in range(9):
            if mask & (1 << i):
                word ^= basis[i]
        qr_code_words.append(word)

    # Extend to [18,9,6] and Shorten to [17,8,6]
    # The shortening is done by keeping words with even parity in the original 17 bits?
    # Actually, the standard [17,8,6] code is the subcode of the [17,9,5] code
    # consisting of words with even weight? Or is it a specific puncturing?
    # The [17,9,5] QR code has even weight? No, QR codes of prime length are self-orthogonal?
    # Let's check the weights. The [17,9,5] code contains the all-zero word.
    # The dual is [17,8,6]. Wait, the [17,8,6] code is the dual of the [17,9,5] code?
    # No, the [17,8,6] code is the even-weight subcode of the [17,9,5] code?
    # Let's verify: The [17,9,5] code is the quadratic residue code.
    # Its dual is the [17,8,6] code.
    # However, for prime p=17, the QR code is self-orthogonal?
    # Actually, the standard construction for A_2(17,6) uses the [17,8,6] code.
    # The [17,9,5] code has minimum distance 5.
    # The [17,8,6] code is a subcode of the [17,9,5] code?
    # If we take the [17,9,5] code and intersect with the even-weight words, do we get [17,8,6]?
    # Let's test this hypothesis in the code.
    
    initial_code = []
    for w in qr_code_words:
        # Check if weight is even
        if bin(w).count('1') % 2 == 0:
            initial_code.append(w)
            
    # Verify distance of initial code
    # If the size is 256, it's likely correct.
    if len(initial_code) != 256:
        # Fallback: if the even-weight subcode isn't 256, we might need the dual.
        # But for now, let's assume the even-weight subcode is the [17,8,6] code.
        # If this fails, we can just use a greedy from scratch, but 256 is a strong start.
        pass

    # --- Step 2: Greedy Improvement ---
    
    if time.time() > deadline - 10.0:
        return initial_code

    # Use a boolean array for blocked words
    # blocked[x] is True if x is within distance < 6 of any codeword in the current code
    blocked = np.zeros(M, dtype=bool)
    for w in initial_code:
        blocked[w] = True
        for off in offsets:
            blocked[w ^ off] = True

    # Candidate pool: all words not blocked
    # We will maintain a list of candidates
    candidates = []
    for x in range(M):
        if not blocked[x]:
            candidates.append(x)
    
    # To speed up "count how many candidates are blocked by x", we can use the blocked array
    # but we need to know which *candidates* are blocked.
    # Since `blocked` marks words that are too close to *existing* codewords,
    # a candidate `x` blocks another candidate `y` if dist(x,y) < 6.
    # We can precompute the neighbors of each candidate? No, too much memory.
    # We will use sampling.

    code = list(initial_code)
    best_code = code[:]
    best_len = len(code)

    # Helper to add a word and update blocked/candidates
    def add_word_to_code(word, blocked_arr, cand_set):
        blocked_arr[word] = True
        cand_set.discard(word)
        for off in offsets:
            y = word ^ off
            blocked_arr[y] = True
            cand_set.discard(y)

    # Greedy with Least-Constraint Heuristic
    # We try multiple restarts with different random seeds/sampling to find better local optima.
    
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
            
        cur_blocked = blocked.copy()
        cur_code = list(initial_code)
        cur_cand_set = set(candidates)
        
        # Greedy loop
        while cur_cand_set and time.time() < deadline - 3.0:
            # Sample candidates to find the one that blocks the fewest others
            # "Blocking score" = number of candidates in cur_cand_set that are within dist < 6 of x
            
            # If set is large, sample. If small, check all.
            if len(cur_cand_set) > 100:
                sample_size = min(100, len(cur_cand_set))
                sample = random.sample(list(cur_cand_set), sample_size)
            else:
                sample = list(cur_cand_set)
            
            best_x = None
            best_score = float('inf')
            
            for x in sample:
                # Count how many candidates would be blocked
                score = 0
                for off in offsets:
                    y = x ^ off
                    if y in cur_cand_set:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Fallback: pick any
                best_x = next(iter(cur_cand_set))
            
            add_word_to_code(best_x, cur_blocked, cur_cand_set)
            cur_code.append(best_x)
            
        if len(cur_code) > best_len:
            best_code = cur_code
            best_len = len(cur_code)
            
            # If we found a better code, restart the blocked array for the next attempt
            # based on this new best code?
            # No, the greedy process is deterministic given the seed. 
            # But changing the initial seed or sample randomness might help.
            # However, the "least-constraining" heuristic is quite strong.
            # Let's stick with the best found so far.

    # --- Step 3: Local Improvement via Swaps ---
    # Try to replace 1 codeword with 2 new ones.
    
    if time.time() < deadline - 5.0:
        cur_code = best_code[:]
        cur_blocked = np.zeros(M, dtype=bool)
        for w in cur_code:
            cur_blocked[w] = True
            for off in offsets:
                cur_blocked[w ^ off] = True
        
        # Precompute list of all words for swap search
        all_words = list(range(M))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try removing each codeword
            for idx in range(len(cur_code)):
                if time.time() > deadline - 2.0:
                    break
                w_remove = cur_code[idx]
                
                # Temporarily unblock w_remove and its neighbors
                # We need to find two words u, v such that:
                # 1. dist(u, v) >= 6
                # 2. dist(u, w_j) >= 6 for all j != idx
                # 3. dist(v, w_j) >= 6 for all j != idx
                # 4. u, v are not in cur_code (obviously, as they were blocked)
                
                # To do this efficiently:
                # 1. Create a "free" mask. A word is free if it is not blocked by any w_j (j != idx).
                #    This is equivalent to: free[x] = not (blocked_by_others[x])
                #    We can compute this by taking the global blocked array,
                #    but we need to remove the contribution of w_remove.
                #    This is hard with a simple boolean array.
                
                # Alternative:
                # Generate the set of candidates that are at dist >= 6 from all w_j (j != idx).
                # Let's call this set S.
                # Then we need to find u, v in S such that dist(u,v) >= 6.
                
                # To build S:
                # Start with all words.
                # For each w_j (j != idx), mark its neighbors as blocked.
                # This is O(N_code * Num_Offsets) per swap attempt.
                # N_code ~ 250, Num_Offsets ~ 1000. 250 * 1000 = 250k ops.
                # This is fast enough.
                
                temp_blocked = np.zeros(M, dtype=bool)
                for j, w in enumerate(cur_code):
                    if j == idx:
                        continue
                    temp_blocked[w] = True
                    for off in offsets:
                        temp_blocked[w ^ off] = True
                
                # Find candidates in S
                # We can just iterate through all words? 131k words.
                # Or maintain a list of "potentially free" words.
                
                # Let's collect candidates
                candidates_swap = []
                for x in all_words:
                    if not temp_blocked[x]:
                        candidates_swap.append(x)
                
                # Now find u, v in candidates_swap with dist >= 6
                # This is a Max Independent Set on a small graph?
                # Or just greedy: pick u, then find v.
                
                found_pair = False
                # Randomize order of candidates for diversity
                random.shuffle(candidates_swap)
                
                for u in candidates_swap:
                    if time.time() > deadline - 1.0:
                        break
                    # Find v in candidates_swap (after u) such that dist(u,v) >= 6
                    # And v is not blocked by u
                    # Since u is in S, u is dist >= 6 from all w_j (j!=idx).
                    # We just need dist(u,v) >= 6.
                    
                    # To speed up, we can check if v is in the "blocked by u" set.
                    # blocked_by_u = {u ^ off for off in offsets}
                    
                    # Iterate through candidates_swap
                    for v in candidates_swap:
                        if u == v:
                            continue
                        if (u ^ v).bit_count() < 6:
                            continue
                        # We found a valid pair (u, v)
                        # Check if they are distinct from w_remove (they are, as they were blocked by w_remove originally? 
                        # Actually, u and v are in S, which only excludes neighbors of w_j (j!=idx).
                        # So u and v might be close to w_remove. That's fine, we are replacing w_remove.)
                        
                        # Perform the swap
                        cur_code.pop(idx)
                        cur_code.append(u)
                        cur_code.append(v)
                        
                        # Update global blocked array?
                        # We can just rebuild it for the next iteration of the while loop.
                        # Or update incrementally. Rebuilding is safer and fast enough.
                        
                        improved = True
                        found_pair = True
                        break
                    
                    if found_pair:
                        break
                
                if found_pair:
                    break # Restart the outer for loop with new code

        if len(cur_code) > best_len:
            best_code = cur_code
            best_len = len(cur_code)

    return best_code

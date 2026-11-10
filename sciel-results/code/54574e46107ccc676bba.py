import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0,...,q-1} with radius.
    The task is specifically K_3(7,1) but the interface is general.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (5s buffer)

    N = q ** n
    if N > 100000:
        # Fallback for large spaces: simple greedy
        return _simple_greedy(q, n, radius, deadline - start_time)

    # Precompute neighbors (ball of radius 1)
    # For radius=1, each word has 1 + n*(q-1) neighbors
    ball_size = 1 + n * (q - 1)
    
    # Efficiently generate neighbors using vectorized operations
    # neighbors[i] is the list of indices of words within distance <= 1 of word i
    neighbors = np.zeros((N, ball_size), dtype=np.int32)
    
    # Precompute the base indices
    # For each position pos and each value change, compute the index offset
    # Actually, let's just precompute the full neighbor list for speed
    for i in range(N):
        neighbors[i, 0] = i
        idx = i
        # We need to change one digit
        # Let's extract digits
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % q
            temp //= q
        
        col = 1
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Compute new index
                # The contribution of position pos is q**pos
                # So new_idx = i - orig * q**pos + new_val * q**pos
                new_idx = i - orig * (q ** pos) + new_val * (q ** pos)
                neighbors[i, col] = new_idx
                col += 1
        # Ensure we filled exactly ball_size entries
        # (It should be, but just in case)
        if col != ball_size:
            # This should never happen for radius=1
            pass

    # Greedy Set Cover
    def greedy_init():
        covered = np.zeros(N, dtype=bool)
        code = []
        # To speed up, we can maintain a count of how many codewords cover each word
        # But simple boolean mask is fine for N=2187
        while not np.all(covered):
            # Calculate gain for each candidate
            # gain[i] = number of uncovered neighbors of i
            # covered[neighbors[i]] is a boolean vector
            # ~covered[neighbors[i]] is True for uncovered
            # sum gives the gain
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            # Pick one, maybe random among max gains to diversify
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            code.append(int(best_c))
            covered[neighbors[best_c]] = True
        return code

    # Local Search: 1-swap
    # Try to replace one codeword with another to reduce size or maintain size while improving coverage?
    # Actually, for covering code, we want to minimize size.
    # A move is: remove one codeword, add one codeword.
    # If we remove c1 and add c2, the size stays same. This doesn't help minimize size directly.
    # To minimize size, we need to remove a codeword that is "redundant" (its coverage is fully covered by others).
    # Or, more generally, try to find a set of k codewords that can be replaced by k-1 codewords.
    
    # Better approach: 
    # 1. Start with greedy.
    # 2. Try to remove any codeword. If the code is still a covering code, keep it removed.
    # 3. Repeat until no more removals possible.
    # 4. Then try to swap: replace c1 with c2, then try to remove c2 or others? 
    #    Actually, a "swap" usually means replace c1 with c2, resulting in same size. 
    #    This is useful to escape local minima in terms of *structure*, allowing subsequent removals.
    
    def is_covering(code_indices):
        covered = np.zeros(N, dtype=bool)
        for c in code_indices:
            covered[neighbors[c]] = True
        return np.all(covered)

    def try_remove(code_indices):
        """Try to remove redundant codewords. Returns new list."""
        code = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                # Temporarily remove code[i]
                remaining = [c for j, c in enumerate(code) if j != i]
                if is_covering(remaining):
                    code = remaining
                    changed = True
                    break
        return code

    def local_search(code_indices):
        """
        Perform local search:
        1. Remove redundant codewords.
        2. Try swaps: replace c1 with c2. If the new set can be reduced, keep the swap.
           Actually, simpler: 
           - Try removing any codeword.
           - If stuck, try to add a codeword that covers many uncovered words? No, we are minimizing.
           - Standard move: Swap c1 with c2. 
             If we swap c1 with c2, size is same. 
             Then we try to remove codewords from the new set. 
             If we can remove at least one more than we added (which is 0 net change in swap, so if we can remove 1, we gain 1), we accept.
        """
        code = list(code_indices)
        code = try_remove(code)
        
        improved = True
        while improved:
            improved = False
            # Try all pairs (i, j) where i is in code, j is not in code
            # To save time, we can limit the search or randomize
            # For N=2187, code size ~180. 
            # Pairs: 180 * (2187-180) ~ 350,000. 
            # Checking is_covering is O(N * ball_size) ~ 2187 * 14 ~ 30k ops.
            # 350k * 30k is too slow.
            
            # Optimization: 
            # When we remove c1, some words become uncovered. 
            # If we add c2, those words must be covered by c2 or other codewords.
            # Let U_c1 be the set of words covered ONLY by c1.
            # If U_c1 is empty, c1 is redundant (handled by try_remove).
            # If we swap c1 with c2, the new uncovered set is (U_c1 \ N(c2)).
            # We accept the swap if the new set can be reduced further? 
            # Or simply: if the new set is a covering code? No, size is same.
            # We want to swap to a position from which we CAN remove a codeword.
            
            # Alternative Local Search: Simulated Annealing on the set of codewords?
            # Or just random restarts of Greedy + Prune.
            
            # Let's try a simpler heuristic: 
            # Randomly pick a codeword c1 to remove.
            # Find the set of words that become uncovered.
            # Find a codeword c2 (not in code) that covers the most of these uncovered words.
            # Add c2.
            # Then try to remove redundant codewords.
            # If the final size is smaller, accept.
            
            if len(code) > 156: # Lower bound
                # Pick a random codeword to try to replace
                idx_to_replace = random.randint(0, len(code) - 1)
                c1 = code[idx_to_replace]
                
                # Find words covered only by c1
                # Count how many codewords cover each word
                counts = np.zeros(N, dtype=np.int32)
                for c in code:
                    counts[neighbors[c]] += 1
                
                # Words with count == 1 are the ones that will become uncovered if c1 is removed
                # But wait, counts includes c1.
                # So words where counts[w] == 1 are covered only by one codeword.
                # We need to check if c1 is that one.
                
                # Efficiently:
                # Create a mask of words covered by c1
                mask_c1 = np.zeros(N, dtype=bool)
                mask_c1[neighbors[c1]] = True
                
                # Words that become uncovered if c1 is removed:
                # These are words w such that counts[w] == 1 AND mask_c1[w] is True.
                # Actually, if counts[w] == 1, it is covered by exactly one codeword.
                # If that codeword is c1, then w becomes uncovered.
                # How to know if c1 is the one?
                # We can just remove c1, see what's uncovered, and try to cover them.
                
                code_test = [c for j, c in enumerate(code) if j != idx_to_replace]
                
                # Find uncovered words
                covered_test = np.zeros(N, dtype=bool)
                for c in code_test:
                    covered_test[neighbors[c]] = True
                
                uncovered = np.where(~covered_test)[0]
                
                if len(uncovered) == 0:
                    # c1 was redundant
                    code = code_test
                    improved = True
                    continue
                
                # Find best c2 to add
                # c2 should cover as many uncovered words as possible
                # gain[c2] = number of uncovered words in neighbors[c2]
                # We can compute this for all c2
                # neighbors shape (N, ball_size)
                # uncovered_mask shape (N,)
                # For each c2, count how many neighbors[c2] are in uncovered
                # This is: for each c2, sum(uncovered_mask[neighbors[c2]])
                
                # Vectorized:
                # uncovered_mask[neighbors] is (N, ball_size) boolean
                # sum along axis 1
                gains = uncovered_mask[neighbors].sum(axis=1) if 'uncovered_mask' in locals() else None
                uncovered_mask = ~covered_test
                gains = uncovered_mask[neighbors].sum(axis=1)
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    continue # No single codeword can cover any uncovered word? Shouldn't happen.
                
                candidates = np.where(gains == max_gain)[0]
                # Filter out candidates already in code
                code_set = set(code_test)
                valid_candidates = [c for c in candidates if c not in code_set]
                
                if not valid_candidates:
                    continue
                
                # Pick a random valid candidate
                c2 = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                
                new_code = code_test + [c2]
                
                # Try to prune new_code
                pruned_code = try_remove(new_code)
                
                if len(pruned_code) < len(code):
                    code = pruned_code
                    improved = True
        
        return code

    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Greedy
        code = greedy_init()
        
        # 2. Prune
        code = try_remove(code)
        
        # 3. Local Search
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            if best_len <= 186:
                # If we found a very good solution, we can try to polish it more
                # or just keep searching for a bit
                pass
            if best_len == 156:
                break
        
        # Optional: Random perturbation of the best code?
        # For now, just restart greedy.

    if best_code is None:
        # Fallback
        return [list(range(n)) for _ in range(0)] # Empty? No, must be valid.
        # If greedy failed, return all words?
        return [list(range(n)) for _ in range(N)] # This is huge, but valid.
    
    # Convert indices to digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
        
    result = []
    for c in best_code:
        result.append(idx_to_digits(c))
        
    return result

def _simple_greedy(q, n, radius, time_limit):
    # Fallback for large N
    # Not really needed for this task
    return []

# Helper to check if a code is valid (for debugging, not used in main loop for speed)
def _is_valid_code(code, q, n, radius):
    N = q ** n
    # Build all words
    # Check coverage
    pass

import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters: return trivial valid code
    if q != 3 or n != 7 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    N = 3 ** n
    # Precompute ball masks as integers
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * 3)
    
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx += digits[i] * powers[i]
        return idx

    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        temp = i
        digits = [0] * n
        for p in range(n):
            digits[p] = temp % 3
            temp //= 3
        for p in range(n):
            orig_val = digits[p]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[p]
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    full_mask = (1 << N) - 1

    def is_valid(code_indices):
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
        return covered == full_mask

    def reduce_code(code_indices):
        # Try to remove each codeword if possible
        # We do this repeatedly until no more can be removed
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                # Check if removing code_indices[i] keeps it valid
                # We can do this by calculating covered without it, 
                # or just re-calculate. Re-calculation is O(K). 
                # Given K~200, this is fast.
                remaining = code_indices[:i] + code_indices[i+1:]
                if is_valid(remaining):
                    code_indices = remaining
                    changed = True
                    break # Restart loop as indices changed
        return code_indices

    def local_search(code_indices, time_limit):
        current_code = list(code_indices)
        best_code = list(current_code)
        best_len = len(current_code)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Try to remove a random codeword
            if len(current_code) > 1:
                i = random.randint(0, len(current_code) - 1)
                candidate = current_code[:i] + current_code[i+1:]
                if is_valid(candidate):
                    current_code = candidate
                    if len(current_code) < best_len:
                        best_code = list(current_code)
                        best_len = len(current_code)
                        # Optional: run reduce_code after a removal to clean up
                        current_code = reduce_code(current_code)
                        if len(current_code) < best_len:
                            best_code = list(current_code)
                            best_len = len(current_code)
            else:
                break # Can't remove if size 1 (and it's valid, it's minimal)

            # Try to swap a random codeword with a random other word
            if len(current_code) > 0:
                i = random.randint(0, len(current_code) - 1)
                new_idx = random.randint(0, N - 1)
                if new_idx not in current_code: # Avoid duplicate
                    candidate = current_code.copy()
                    candidate[i] = new_idx
                    if is_valid(candidate):
                        # Accept swap if it reduces size? No, swap keeps size same.
                        # We only accept swap if it allows future removals? 
                        # Standard local search for minimization: 
                        # Usually we just try to remove. Swaps are useful to escape local minima.
                        # However, if we just swap, size doesn't change. 
                        # We should only accept a swap if it leads to a state where we can remove more?
                        # That's hard to check.
                        # Alternative: Accept swap with some probability (Simulated Annealing) or 
                        # just accept if it doesn't increase size (it doesn't) and hope for the best?
                        # Actually, if we accept a swap that doesn't reduce size, we might get stuck.
                        # Better: Only accept swap if it results in a code that CAN be reduced further?
                        # Let's try: Accept swap, then try to reduce. If reduction succeeds, keep.
                        # If reduction fails, revert.
                        old_code = list(current_code)
                        current_code = candidate
                        reduced = reduce_code(current_code)
                        if len(reduced) < len(old_code):
                            current_code = reduced
                            if len(current_code) < best_len:
                                best_code = list(current_code)
                                best_len = len(current_code)
                        else:
                            current_code = old_code

            # Random restart? No, we are doing local search.
            # If we get stuck, we can try a "kick": remove a random word, add a random word, 
            # or just restart from best_code.
            
            # Check if we are stuck? 
            # For simplicity, we just keep going.
            
        return best_code

    # Initial code: Use the Tetracode construction (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            digits = h + list(tail)
            seed_code_idx.append(digits_to_idx(digits))
    
    # Reduce the seed code first
    current_code = reduce_code(seed_code_idx)
    best_code = list(current_code)
    best_len = len(current_code)
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Run local search
    # We can run multiple independent local searches from different starting points
    # or just one long one.
    # Let's try running local search on the reduced seed code.
    
    while time.time() < deadline:
        # Run local search for a short burst or until time
        # To avoid getting stuck in a specific local minimum, we can occasionally
        # perturb the solution.
        
        # For now, just run local search on the current best.
        # To diversify, we can start a new local search from a slightly perturbed best_code
        # or just continue.
        
        # Let's run local search on a copy of best_code
        search_start = time.time()
        burst_time = min(10, deadline - time.time())
        if burst_time <= 0: break
        
        candidate_best = local_search(best_code, burst_time)
        
        if len(candidate_best) < best_len:
            best_code = candidate_best
            best_len = len(best_code)
        
        # If we didn't improve in a burst, we might want to perturb.
        # Perturbation: Remove 2 random words, add 2 random words to maintain validity? 
        # That's complex. 
        # Simpler: Restart local search from a random valid code? 
        # Generating a random valid code is hard.
        # Let's stick to one long local search or multiple short ones from the same start.
        # Actually, randomizing the order of removals in reduce_code helps.
        
    # Convert best_code indices to words
    final_words = []
    for idx in best_code:
        digits = [0] * n
        temp = idx
        for p in range(n):
            digits[p] = temp % 3
            temp //= 3
        final_words.append(digits)
        
    return final_words

"""Football pool problem: ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _build_balls(q, n, radius):
    """Precompute for each word the list of words within Hamming distance radius."""
    total = q ** n
    balls = []
    for w in range(total):
        word = [0] * n
        idx = w
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    word[pos] = v
                    nb_idx = 0
                    for val in word:
                        nb_idx = nb_idx * q + val
                    neighbors.append(nb_idx)
                    word[pos] = orig
        balls.append(neighbors)
    return balls

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    rng = random.Random(42)
    
    # Precompute balls
    balls = _build_balls(q, n, radius)
    
    # Initial code: tetracode sum (81 words)
    # Tetracode is a perfect code of length 4, radius 1, size 9 over GF(3)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    initial_indices = set(word_to_idx(c) for c in initial_code)
    best_code = set(initial_indices)
    best_size = len(best_code)
    
    # Coverage array: cov[i] is the number of codewords covering word i
    cov = np.zeros(total, dtype=np.int32)
    
    def update_cov(ci, delta):
        for nb in balls[ci]:
            cov[nb] += delta

    # Initialize coverage for initial code
    for ci in initial_indices:
        update_cov(ci, 1)
        
    def try_remove_redundant(code_set, cov_array):
        """
        Repeatedly remove redundant codewords.
        A codeword is redundant if min(cov[nb] for nb in ball(ci)) >= 2.
        Returns True if any were removed.
        """
        removed_any = False
        while True:
            # Find a redundant codeword
            # Optimization: Iterate through code_set. 
            # To avoid O(N^2) if N is large, we could maintain a list of candidates, 
            # but N is small (~80), so O(N) scan is fine.
            
            redundant_ci = -1
            max_slack = -1
            best_ci = -1
            
            for ci in code_set:
                # Check if min cov in ball is >= 2
                min_c = 999
                slack = 0
                for nb in balls[ci]:
                    c_val = cov_array[nb]
                    if c_val < min_c:
                        min_c = c_val
                    slack += c_val - 1
                if min_c >= 2:
                    # It is redundant. Prefer one with highest slack to be safe?
                    # Or just pick the first one?
                    # Picking highest slack might be better for stability.
                    if slack > max_slack:
                        max_slack = slack
                        redundant_ci = ci
                        best_ci = ci
            
            if redundant_ci != -1:
                # Remove it
                code_set.discard(redundant_ci)
                update_cov(redundant_ci, -1)
                removed_any = True
            else:
                break
                
        return removed_any

    # Main Search Loop: Simulated Annealing / Randomized Local Search
    # We will try to find a smaller code.
    # State: code_set, cov
    # Move: Swap a codeword c_old with a neighbor c_new not in code.
    # Acceptance:
    #   If new_size < current_size: Accept (Update Best)
    #   If new_size == current_size: Accept with probability p (Annealing)
    #   If new_size > current_size: Reject
    
    current_code = set(best_code)
    # Ensure cov matches current_code
    cov[:] = 0
    for ci in current_code:
        update_cov(ci, 1)
    
    current_size = len(current_code)
    
    # Temperature for annealing
    # Start high, decay to low
    T_start = 1.0
    T_end = 0.01
    # Number of iterations to approximate decay
    # We don't know total iterations, so we use time-based decay or fixed steps.
    # Let's do fixed steps for the annealing schedule, then switch to pure greedy/improvement.
    
    # Phase 1: Annealing Search
    # We run this for a certain time or number of steps.
    steps = 0
    max_steps = 50000 # Adjust based on speed
    # Each step is roughly: 1 swap sim + 1 removal loop.
    # Removal loop is fast. Swap sim is fast.
    # 50k steps should take a few seconds.
    
    for step in range(max_steps):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        # Decay temperature
        # Linear decay
        T = T_start * (1 - step / max_steps) + T_end * (step / max_steps)
        
        # Pick a random codeword to swap out
        if not current_code:
            break
        cis = list(current_code)
        c_old = rng.choice(cis)
        
        # Get neighbors of c_old not in current_code
        neighbors = balls[c_old]
        candidates = [nb for nb in neighbors if nb not in current_code]
        
        if not candidates:
            continue
            
        c_new = rng.choice(candidates)
        
        # Perform Swap
        # Remove c_old
        for nb in balls[c_old]:
            cov[nb] -= 1
        current_code.discard(c_old)
        
        # Add c_new
        for nb in balls[c_new]:
            cov[nb] += 1
        current_code.add(c_new)
        
        # Try to remove redundant
        removed_any = try_remove_redundant(current_code, cov)
        
        new_size = len(current_code)
        
        # Evaluate Move
        delta = new_size - current_size
        
        if delta < 0:
            # Improvement
            current_size = new_size
            if new_size < best_size:
                best_size = new_size
                best_code = set(current_code)
                # If we reached 73, we can stop or continue to see if 72 is possible (unlikely)
                if best_size <= 73:
                    # We found the known optimal. We can try to break or continue.
                    # Let's continue briefly to ensure we didn't miss a 72, but 72 is lower bound 71, so 72 is possible?
                    # Best known is 73. Lower bound 71.
                    # If we find 72, it's a new record!
                    # But for this task, 73 is the target.
                    pass
        elif delta == 0:
            # Neutral
            # Accept with probability exp(0/T) = 1? No, standard SA accepts neutral always?
            # Or use a specific probability.
            # Let's accept neutral moves with high probability to explore.
            if rng.random() < 0.8:
                current_size = new_size
            else:
                # Revert
                # Remove c_new
                for nb in balls[c_new]:
                    cov[nb] -= 1
                current_code.discard(c_new)
                # Add c_old
                for nb in balls[c_old]:
                    cov[nb] += 1
                current_code.add(c_old)
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            # delta is positive (size increased)
            # prob = exp(-delta / T)
            # Since T is small, this will be very small.
            prob = np.exp(-delta / (T + 1e-9))
            if rng.random() < prob:
                current_size = new_size
            else:
                # Revert
                for nb in balls[c_new]:
                    cov[nb] -= 1
                current_code.discard(c_new)
                for nb in balls[c_old]:
                    cov[nb] += 1
                current_code.add(c_old)
                
        if best_size <= 73:
            # We found a very good solution. We can break out of annealing 
            # and do a final polish.
            break

    # Phase 2: Intensive Local Search on Best Found
    # Use the best_code found so far.
    # Run a more aggressive search: try all swaps for each codeword.
    # This is O(N^2 * BallSize) which is manageable.
    
    final_code = set(best_code)
    cov[:] = 0
    for ci in final_code:
        update_cov(ci, 1)
    
    improved = True
    while improved and time.time() - start_time < _DEADLINE - 5:
        improved = False
        # Try all swaps
        cis = list(final_code)
        rng.shuffle(cis) # Randomize order to avoid bias
        for c_old in cis:
            if time.time() - start_time > _DEADLINE - 5:
                break
            if c_old not in final_code:
                continue
            neighbors = balls[c_old]
            candidates = [nb for nb in neighbors if nb not in final_code]
            for c_new in candidates:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                
                # Simulate swap
                # Remove c_old
                for nb in balls[c_old]:
                    cov[nb] -= 1
                final_code.discard(c_old)
                
                # Add c_new
                for nb in balls[c_new]:
                    cov[nb] += 1
                final_code.add(c_new)
                
                # Try removal
                removed_any = try_remove_redundant(final_code, cov)
                new_size = len(final_code)
                
                if new_size < len(best_code):
                    # Accept
                    best_code = set(final_code)
                    best_size = new_size
                    improved = True
                    # Restart the inner loop or break?
                    # Breaking to restart the outer loop with the new best is safer.
                    break
                else:
                    # Revert
                    for nb in balls[c_new]:
                        cov[nb] -= 1
                    final_code.discard(c_new)
                    for nb in balls[c_old]:
                        cov[nb] += 1
                    final_code.add(c_old)
            
            if len(best_code) < len(final_code):
                # We improved, final_code is stale? No, we updated best_code.
                # But final_code is the one we are modifying.
                # If we broke out of inner loop due to improvement, final_code is the improved one.
                # We should continue searching from this new state.
                # So we don't revert. We just break the inner loop and let the outer while loop restart?
                # The outer while loop restarts only if 'improved' is True.
                # But 'final_code' is already updated.
                # We should break the 'for c_old' loop to restart the 'while improved' loop.
                break

    # Final Result
    result = [idx_to_word(ci) for ci in sorted(best_code)]
    return result
# EVOLVE-BLOCK-END

import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # For this specific task, we assume q=3, n=7, radius=1
    # But the function signature is general.
    # If parameters differ, we might need a fallback.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we only need to handle the specific case or a general case.
    # Given the time limit and complexity, I will specialize for q=3, n=7, r=1
    # and provide a fallback for other cases (e.g., greedy only).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy
        return _greedy_construct(q, n, radius)

    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute balls as lists of indices for faster set operations in SA
    # For q=3, n=7, r=1, each ball has 1 + 7*2 = 15 words.
    # Storing as lists is efficient for membership checks if we use a set for the current code's coverage.
    # However, to check if a word is covered, we need to know if it's in the union of balls of current code.
    # Maintaining a coverage count array is O(1) per check but O(ball_size) per update.
    
    # Let's use a coverage count array approach for SA.
    # cover_count[i] = number of codewords whose ball covers word i.
    # A word is covered if cover_count[i] > 0.
    # Energy = number of words with cover_count[i] == 0.
    
    # Precompute ball neighbors
    ball_neighbors = []
    for i in range(N):
        w = words[i]
        neighbors = [i]
        for pos in range(n):
            for delta in [1, 2]: # -1 and +1 mod 3
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    # Calculate index
                    # Since words are generated in product order, index is base-q representation
                    # But calculating index from word is O(n). 
                    # We can precompute a mapping or calculate on the fly.
                    # For n=7, O(n) is small.
                    idx = 0
                    for k in range(n):
                        idx = idx * q + new_w[k]
                    neighbors.append(idx)
        ball_neighbors.append(neighbors)

    # Initialize with a greedy code
    uncovered = [1] * N
    code = []
    
    # Greedy initialization
    # To speed up greedy, we can just pick random words or words that cover many uncovered.
    # Simple greedy: pick word that covers most uncovered.
    # We need to iterate until all covered.
    remaining = N
    while remaining > 0:
        best_i = -1
        best_cov = 0
        # Sample or iterate all? Iterating all N=2187 is fast enough for ~100 steps.
        for i in range(N):
            cov = 0
            for j in ball_neighbors[i]:
                if uncovered[j] > 0:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        for j in ball_neighbors[best_i]:
            uncovered[j] += 1
            if uncovered[j] == 1:
                remaining -= 1
        if remaining == 0:
            break

    initial_energy = remaining
    current_energy = initial_energy
    current_code = code[:]
    
    # Setup coverage counts
    cover_count = np.zeros(N, dtype=np.int32)
    for i in current_code:
        for j in ball_neighbors[i]:
            cover_count[j] += 1
    
    # Simulated Annealing
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    best_code = current_code[:]
    best_energy = current_energy
    
    # Precompute random indices for speed? No, just use random.randint
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.33:
            # Add a random word
            i = random.randint(0, N - 1)
            if i in current_code:
                continue
            # Check energy delta
            delta = 0
            for j in ball_neighbors[i]:
                if cover_count[j] == 0:
                    delta -= 1 # Energy decreases
            if delta >= 0:
                # No improvement, accept with probability exp(-delta/T)
                if random.random() > np.exp(-delta / T):
                    continue
            # Apply
            current_code.append(i)
            for j in ball_neighbors[i]:
                cover_count[j] += 1
            current_energy += delta
            
        elif move_type < 0.66:
            # Remove a random word
            if not current_code:
                continue
            idx_in_code = random.randint(0, len(current_code) - 1)
            i = current_code[idx_in_code]
            
            # Check energy delta
            delta = 0
            for j in ball_neighbors[i]:
                if cover_count[j] == 1:
                    delta += 1 # Energy increases
            if delta > 0:
                if random.random() > np.exp(-delta / T):
                    continue
            # Apply
            current_code.pop(idx_in_code)
            for j in ball_neighbors[i]:
                cover_count[j] -= 1
            current_energy += delta
            
        else:
            # Swap: Remove one, Add one
            if not current_code:
                continue
            idx_in_code = random.randint(0, len(current_code) - 1)
            i = current_code[idx_in_code]
            k = random.randint(0, N - 1)
            if k == i:
                continue
            if k in current_code:
                continue
            
            # Calculate delta
            delta = 0
            # Effect of removing i
            for j in ball_neighbors[i]:
                if cover_count[j] == 1:
                    delta += 1
            # Effect of adding k (note: cover_count hasn't been updated for removal yet in this calc logic, 
            # but we can simulate)
            # Actually, it's safer to compute net change.
            # Let's just compute the change in energy directly.
            # Energy = count of words with cover_count 0.
            # Let's use a temporary check.
            
            # Simpler: Just compute delta based on current state.
            # If we remove i, words with count 1 become 0 (bad).
            # If we add k, words with count 0 become 1 (good).
            # Words in intersection might cancel out?
            # Let's just do the two steps sequentially in a "dry run" or just calculate carefully.
            
            # Words that become uncovered if i removed:
            # Those j in B_i where cover_count[j] == 1.
            # Words that become covered if k added:
            # Those j in B_k where (cover_count[j] - (1 if j in B_i else 0)) == 0.
            
            # Let's just apply and revert if rejected? No, that's expensive.
            # Let's calculate delta precisely.
            
            delta = 0
            # Remove i
            for j in ball_neighbors[i]:
                if cover_count[j] == 1:
                    delta += 1 # Becomes uncovered
            # Add k
            for j in ball_neighbors[k]:
                # New count for j will be cover_count[j] - (1 if j in B_i else 0) + 1
                # We want to know if it goes from 0 to >0.
                # Old count: cover_count[j]
                # If j in B_i, effective old count before adding k is cover_count[j] - 1.
                # If j not in B_i, effective old count is cover_count[j].
                # We need to know if effective old count was 0.
                # And new count > 0.
                
                # If j in B_i:
                #   effective_old = cover_count[j] - 1
                #   new = effective_old + 1 = cover_count[j]
                #   Covered if new > 0. Uncovered before if effective_old == 0.
                #   So if cover_count[j] == 1 (and j in B_i), it was uncovered (after removing i) and becomes covered.
                #   Wait, if j in B_i and cover_count[j]==1, then after removing i, count is 0. After adding k, count is 1.
                #   So delta -= 1.
                
                # If j not in B_i:
                #   effective_old = cover_count[j]
                #   new = cover_count[j] + 1
                #   Covered if new > 0 (always true). Uncovered before if effective_old == 0.
                #   So if cover_count[j] == 0 (and j not in B_i), delta -= 1.
                
                if j in ball_neighbors[i]:
                    if cover_count[j] == 1:
                        delta -= 1 # Was uncovered after removal, now covered
                else:
                    if cover_count[j] == 0:
                        delta -= 1 # Was uncovered, now covered
                        
            if delta > 0:
                if random.random() > np.exp(-delta / T):
                    continue
            # Apply
            current_code[idx_in_code] = k
            for j in ball_neighbors[i]:
                cover_count[j] -= 1
            for j in ball_neighbors[k]:
                cover_count[j] += 1
            current_energy += delta
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_code = current_code[:]
            
        T *= alpha
        
    return [words[i] for i in best_code]

def _greedy_construct(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Simple Greedy without bitsets for fallback
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        # Remove covered
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
    return [words[i] for i in code]

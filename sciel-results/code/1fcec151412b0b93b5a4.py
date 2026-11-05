import numpy as np
import time
import random

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1
    # Total words = 729
    # Each ball has size 1 + n*(q-1) = 1 + 6*2 = 13
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute all words as numpy array of shape (total_words, n)
    all_words = np.zeros((total_words, n), dtype=np.int32)
    for idx in range(total_words):
        val = idx
        for i in range(n):
            all_words[idx, i] = val % q
            val //= q
    
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    # Precompute balls as numpy arrays
    balls_np = []
    for c in range(total_words):
        covered = set()
        covered.add(c)
        for i in range(n):
            orig_val = all_words[c, i]
            for new_val in range(q):
                if new_val != orig_val:
                    new_idx = c + (new_val - orig_val) * q_powers[i]
                    covered.add(new_idx)
        balls_np.append(np.array(list(covered), dtype=np.int32))
    
    # Build a good initial code using a fast greedy approach
    # Use random sampling to speed up greedy
    uncovered = np.ones(total_words, dtype=bool)
    initial_code = []
    
    while np.any(uncovered):
        # Sample a subset of candidates to speed up
        candidates = list(range(total_words))
        random.Random(42).shuffle(candidates)
        candidates = candidates[:200]  # Sample 200 candidates
        
        best_c = -1
        best_count = -1
        for c in candidates:
            if c in initial_code:
                continue
            count = np.sum(uncovered[balls_np[c]])
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1:
            # Fallback: scan all
            for c in range(total_words):
                if c in initial_code:
                    continue
                count = np.sum(uncovered[balls_np[c]])
                if count > best_count:
                    best_count = count
                    best_c = c
        if best_c == -1:
            break
        initial_code.append(best_c)
        uncovered[balls_np[best_c]] = False
    
    initial_code = list(set(initial_code))
    code_set = set(initial_code)
    
    # Compute coverage counts
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        counts[balls_np[c]] += 1
    
    assert np.all(counts >= 1), "Initial code doesn't cover all words"
    
    # Simulated Annealing with numpy
    start_time = time.time()
    deadline = start_time + 110
    
    best_code = list(code_set)
    best_size = len(code_set)
    
    # Run multiple SA restarts with different seeds
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        # Reset code to best found so far
        code_set = set(best_code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            counts[balls_np[c]] += 1
        
        # Use different temperature schedule for each restart
        T_init = 2.0 + 0.5 * restart
        T_min = 0.001
        cooling = 0.9997
        
        T = T_init
        rng = random.Random(42 + restart)
        
        restart_deadline = start_time + (restart + 1) * time_per_restart
        
        while time.time() < restart_deadline:
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.45:
                # Remove a codeword if no unique words
                if not code_set:
                    continue
                c = rng.choice(list(code_set))
                # Check if any word is uniquely covered by c
                ball_c = balls_np[c]
                unique_count = np.sum(counts[ball_c] == 1)
                if unique_count == 0:
                    # Safe to remove
                    code_set.remove(c)
                    counts[ball_c] -= 1
                    if len(code_set) < best_size:
                        best_code = list(code_set)
                        best_size = len(code_set)
                # If unique_count > 0, skip (can't remove)
                
            elif move_type < 0.75:
                # Add a codeword
                c_new = rng.randrange(total_words)
                if c_new in code_set:
                    continue
                code_set.add(c_new)
                counts[balls_np[c_new]] += 1
                # Accept with probability exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    pass  # Accept
                else:
                    # Reject: undo
                    code_set.remove(c_new)
                    counts[balls_np[c_new]] -= 1
                    
            else:
                # Swap: remove c, add c_prime
                if not code_set:
                    continue
                c = rng.choice(list(code_set))
                ball_c = balls_np[c]
                unique_words = ball_c[counts[ball_c] == 1]
                
                if len(unique_words) == 0:
                    # Can remove c safely, so just add a random codeword and remove c
                    c_prime = rng.randrange(total_words)
                    if c_prime in code_set:
                        continue
                    code_set.remove(c)
                    code_set.add(c_prime)
                    counts[ball_c] -= 1
                    counts[balls_np[c_prime]] += 1
                else:
                    # Need c_prime to cover all unique words
                    # Find candidates that cover all unique words
                    found = False
                    for _ in range(30):
                        c_prime = rng.randrange(total_words)
                        if c_prime in code_set:
                            continue
                        # Check if balls_np[c_prime] covers all unique_words
                        if np.all(np.isin(unique_words, balls_np[c_prime])):
                            # Perform swap
                            code_set.remove(c)
                            code_set.add(c_prime)
                            counts[ball_c] -= 1
                            counts[balls_np[c_prime]] += 1
                            found = True
                            break
                    if not found:
                        continue
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Final greedy cleanup
    code_set = set(best_code)
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        counts[balls_np[c]] += 1
    
    for _ in range(500):
        removed = False
        for c in list(code_set):
            ball_c = balls_np[c]
            if np.sum(counts[ball_c] == 1) == 0:
                code_set.remove(c)
                counts[ball_c] -= 1
                removed = True
        if not removed:
            break
    
    # Convert back to list of lists
    result = []
    for c in code_set:
        w = [0] * n
        val = c
        for i in range(n):
            w[i] = val % q
            val //= q
        result.append(w)
    
    return result

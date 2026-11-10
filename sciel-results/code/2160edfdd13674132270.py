import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin
    
    n = 128
    target_range = range(128, 257)
    
    # Helper to evaluate a set A (list of indices or boolean array)
    # Returns (is_valid, max_mult)
    def evaluate(A_bool):
        # A_bool is np.array of bools length 129
        # Convolution of A_bool with itself
        # C[t] = sum A[i] * A[t-i]
        # We only care about t in [128, 256]
        
        # Using numpy convolve
        # Full convolution length is 129+129-1 = 257
        C = np.convolve(A_bool, A_bool, mode='full')
        
        # C[t] is count of ordered pairs (i, j) such that i+j=t
        # We need unordered pairs {i, j} with i <= j
        # R[t] = (C[t] + (1 if t is even and A[t//2] else 0)) // 2
        
        max_mult = 0
        valid = True
        
        # We can vectorize the check
        # targets = np.arange(128, 257)
        # counts = C[128:257]
        # diag = np.where(targets % 2 == 0, A_bool[targets//2], 0)
        # reps = (counts + diag) // 2
        
        # Let's do it simply to avoid indexing errors in thought process
        # But for speed, vectorize.
        
        targets = np.arange(128, 257)
        counts = C[128:257]
        
        # Determine diagonal contribution
        # t is even -> t//2 is integer
        # Check if t//2 is in A
        half_targets = targets // 2
        is_even = (targets % 2 == 0)
        
        # A_bool[half_targets] gives 1 if the half element is in A
        # But we only want it if t is even.
        diag_contrib = np.where(is_even, A_bool[half_targets], 0)
        
        reps = (counts + diag_contrib) // 2
        
        if np.any(reps == 0):
            return False, 1000000 # Invalid
            
        return True, np.max(reps)

    best_set = None
    best_score = 1000000
    
    # Initial population or single trajectory?
    # Let's do a few random restarts with local search.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Try sizes around 35-45
        k = random.randint(35, 45)
        A_idx = random.sample(range(129), k)
        A_bool = np.zeros(129, dtype=bool)
        A_bool[A_idx] = True
        
        is_valid, score = evaluate(A_bool)
        
        if not is_valid:
            # Try to repair by adding random elements
            for repair_step in range(100):
                if time.time() > deadline: break
                # Find a missing sum
                C = np.convolve(A_bool, A_bool, mode='full')
                targets = np.arange(128, 257)
                counts = C[128:257]
                half_targets = targets // 2
                is_even = (targets % 2 == 0)
                diag_contrib = np.where(is_even, A_bool[half_targets], 0)
                reps = (counts + diag_contrib) // 2
                missing = np.where(reps == 0)[0]
                if len(missing) == 0:
                    break
                t_missing = targets[missing[random.randint(0, len(missing)-1)]]
                # Add an element x such that x + (t_missing - x) in A
                # Ideally pick x such that t_missing - x is in A
                # Or just pick random x and hope it helps
                # Better: pick x randomly, check if it improves coverage
                x = random.randint(0, 128)
                if not A_bool[x]:
                    A_bool[x] = True
            is_valid, score = evaluate(A_bool)
            if not is_valid:
                continue
                
        if score < best_score:
            best_score = score
            best_set = A_bool.copy()
            
        # Local Search (Simulated Annealing style)
        current_score = score
        current_set = A_bool.copy()
        
        temp = 1.0
        for step in range(1000):
            if time.time() > deadline: break
            
            # Pick a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            new_set = current_set.copy()
            
            if move_type == 'add':
                x = random.randint(0, 128)
                if not new_set[x]:
                    new_set[x] = True
                else:
                    continue
            elif move_type == 'remove':
                if np.sum(new_set) <= 20: # Don't get too small
                    continue
                x = random.choice(np.where(new_set)[0])
                new_set[x] = False
            else: # swap
                if np.sum(new_set) == 0: continue
                x = random.choice(np.where(new_set)[0])
                y = random.randint(0, 128)
                if new_set[y]: continue
                new_set[x] = False
                new_set[y] = True
                
            is_valid, new_score = evaluate(new_set)
            
            if not is_valid:
                # Invalid sets are heavily penalized
                new_score = 1000000
                
            if new_score < current_score:
                current_score = new_score
                current_set = new_set
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set.copy()
            else:
                # Accept with probability exp(-delta/temp)
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < np.exp(-delta / temp):
                        current_score = new_score
                        current_set = new_set
                else:
                    # Should not happen if new_score >= current_score
                    current_score = new_score
                    current_set = new_set
                    
            temp *= 0.995 # Cooling
            
            # If we found a very good score, stop early?
            if best_score <= 3:
                break
                
        if best_score <= 3:
            break

    if best_set is None:
        # Fallback to seed
        return list(range(64, 129))
        
    # Convert boolean array to list of indices
    return list(np.where(best_set)[0])

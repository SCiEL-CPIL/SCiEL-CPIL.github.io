import time
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def solve_sum_free(A, time_limit=30.0):
        n = len(A)
        if n == 0:
            return []
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(A)}
        
        # Precompute triples (i, j, k) such that A[i] + A[j] = A[k]
        triples = []
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    triples.append((i, j, k))
        
        # For fast update, map each index to the triples it participates in
        idx_to_triples = [[] for _ in range(n)]
        for t_idx, (i, j, k) in enumerate(triples):
            idx_to_triples[i].append(t_idx)
            idx_to_triples[j].append(t_idx)
            idx_to_triples[k].append(t_idx)
        
        # State: boolean array in_set
        # Start with odd numbers heuristic
        in_set = [False] * n
        for i in range(n):
            if A[i] % 2 != 0:
                in_set[i] = True
        
        # Calculate initial penalty (number of violated triples)
        penalty = 0
        for t_idx, (i, j, k) in enumerate(triples):
            if in_set[i] and in_set[j] and in_set[k]:
                penalty += 1
        
        current_size = sum(in_set)
        current_energy = -current_size + penalty
        
        best_in_set = in_set[:]
        best_size = current_size
        best_energy = current_energy
        best_penalty = penalty
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        start_time = time.time()
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Pick a random index to toggle
            i = random.randint(0, n - 1)
            
            old_state = in_set[i]
            new_state = not old_state
            
            # Calculate change in penalty
            delta_penalty = 0
            for t_idx in idx_to_triples[i]:
                a_idx, b_idx, c_idx = triples[t_idx]
                
                old_count = (1 if in_set[a_idx] else 0) + (1 if in_set[b_idx] else 0) + (1 if in_set[c_idx] else 0)
                
                if new_state:
                    new_count = old_count + 1
                else:
                    new_count = old_count - 1
                
                old_violated = 1 if old_count == 3 else 0
                new_violated = 1 if new_count == 3 else 0
                delta_penalty += (new_violated - old_violated)
            
            new_penalty = penalty + delta_penalty
            new_size = current_size + (1 if new_state else -1)
            new_energy = -new_size + new_penalty
            
            delta_E = new_energy - current_energy
            
            if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                in_set[i] = new_state
                penalty = new_penalty
                current_size = new_size
                current_energy = new_energy
                
                # Update best if valid and larger, or better energy if valid
                if penalty == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set[:]
                        best_energy = current_energy
                        best_penalty = 0
                    elif current_energy < best_energy:
                        best_in_set = in_set[:]
                        best_size = current_size
                        best_energy = current_energy
                        best_penalty = 0
            
            T *= cooling
        
        # If best_in_set is not valid (should not happen), repair it
        if best_penalty > 0:
             # Greedy repair: remove elements involved in violations
             # This is a fallback, unlikely to be needed
             in_set = best_in_set[:]
             penalty = best_penalty
             while penalty > 0:
                 # Find a violated triple
                 found = False
                 for t_idx, (i, j, k) in enumerate(triples):
                     if in_set[i] and in_set[j] and in_set[k]:
                         # Remove one of them, say k
                         in_set[k] = False
                         penalty -= 1
                         found = True
                         break
                 if not found:
                     break
             best_in_set = in_set
             best_size = sum(in_set)
        
        return [A[i] for i in range(n) if best_in_set[i]]

    results = []
    total_start = time.time()
    for idx, (size, seed) in enumerate(((60, 801), (120, 802), (240, 803))):
        A = instance(size, seed)
        # Allocate remaining time, min 5s, max 40s
        elapsed = time.time() - total_start
        remaining = 140.0 - elapsed # 10s safety margin
        per_instance = max(5.0, min(40.0, remaining / (3 - idx)))
        subset = solve_sum_free(A, time_limit=per_instance)
        results.append(subset)
        
    return results

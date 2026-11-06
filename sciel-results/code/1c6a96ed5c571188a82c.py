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
    
    def solve_sum_free(A):
        n = len(A)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(A)}
        
        triples = []
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    triples.append((i, j, k))
        
        idx_to_triples = [[] for _ in range(n)]
        for t_idx, (i, j, k) in enumerate(triples):
            idx_to_triples[i].append(t_idx)
            if i != j:
                idx_to_triples[j].append(t_idx)
            idx_to_triples[k].append(t_idx)
        
        in_set = [False] * n
        for i in range(n):
            if A[i] % 2 != 0:
                in_set[i] = True
        
        penalty = 0
        for t_idx, (i, j, k) in enumerate(triples):
            if in_set[i] and in_set[j] and in_set[k]:
                penalty += 1
        
        current_size = sum(in_set)
        current_energy = -current_size + penalty
        
        best_in_set = in_set[:]
        best_size = current_size
        best_energy = current_energy
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        start_time = time.time()
        time_limit = 2.0
        
        while T > min_T and (time.time() - start_time) < time_limit:
            i = random.randint(0, n - 1)
            
            old_state = in_set[i]
            new_state = not old_state
            
            delta_penalty = 0
            for t_idx in idx_to_triples[i]:
                a, b, c = triples[t_idx]
                # Count occurrences of i in (a, b, c)
                count_i = (1 if a == i else 0) + (1 if b == i else 0) + (1 if c == i else 0)
                
                old_count = (1 if in_set[a] else 0) + (1 if in_set[b] else 0) + (1 if in_set[c] else 0)
                
                if new_state:
                    new_count = old_count + count_i
                else:
                    new_count = old_count - count_i
                
                old_violated = 1 if old_count == 3 else 0
                new_violated = 1 if new_count == 3 else 0
                delta_penalty += (new_violated - old_violated)
            
            new_penalty = penalty + delta_penalty
            new_size = current_size + (1 if new_state else -1)
            new_energy = -new_size + new_penalty
            
            delta_E = new_energy - current_energy
            
            if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
                in_set[i] = new_state
                penalty = new_penalty
                current_size = new_size
                current_energy = new_energy
                
                if penalty == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set[:]
                        best_energy = current_energy
            
            T *= cooling
        
        # Final safety check: ensure best_in_set is valid
        # If not, repair it
        def is_valid(s):
            s_set = set(s)
            for a in s:
                for b in s:
                    if a + b in s_set:
                        return False
            return True
        
        result_vals = [A[i] for i in range(n) if best_in_set[i]]
        if not is_valid(result_vals):
            # Greedy repair: remove elements causing conflicts
            # This is O(n^2) check, but n is small enough for repair
            # We can just re-run a simple greedy removal
            current_set = set(result_vals)
            changed = True
            while changed:
                changed = False
                for a in list(current_set):
                    for b in current_set:
                        if a + b in current_set:
                            # Remove one, e.g., a+b
                            current_set.remove(a + b)
                            changed = True
                            break
                    if changed:
                        break
            result_vals = list(current_set)
        
        return result_vals

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        A = instance(size, seed)
        subset = solve_sum_free(A)
        results.append(subset)
        
    return results

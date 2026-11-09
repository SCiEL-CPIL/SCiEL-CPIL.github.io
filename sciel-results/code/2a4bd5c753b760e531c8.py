import time
import random
import numpy as np
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    # Generate all 5-smooth numbers (2^a * 3^b * 5^c) <= max_val
    # Store as (value, (a, b, c))
    smooth_numbers = []
    for a in range(15):  # 2^14 = 16384 > 10000
        for b in range(10):  # 3^9 = 19683 > 10000
            for c in range(7):  # 5^6 = 15625 > 10000
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= max_val:
                    smooth_numbers.append((val, (a, b, c)))
    
    # Sort by value
    smooth_numbers.sort()
    # Extract just the values and exponents
    smooth_vals = [s[0] for s in smooth_numbers]
    smooth_exps = [s[1] for s in smooth_numbers]
    val_to_idx = {v: i for i, v in enumerate(smooth_vals)}
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        a = np.array(A, dtype=np.int64)
        sumset = a[:, None] + a[None, :]
        prodset = a[:, None] * a[None, :]
        s_size = len(np.unique(sumset))
        p_size = len(np.unique(prodset))
        return max(s_size, p_size)

    def evaluate_from_exps(exp_triples):
        """Evaluate score given exponent triples (a,b,c) for 5-smooth numbers."""
        # Convert to values
        vals = [(2**a) * (3**b) * (5**c) for (a, b, c) in exp_triples]
        return evaluate(vals)

    def get_smooth_numbers(k, limit, count):
        """Generate smallest 'count' k-smooth numbers <= limit."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        primes = [p for p in primes if p <= k]
        smooth_list = []
        for x in range(1, limit + 1):
            t = x
            for p in primes:
                while t % p == 0:
                    t //= p
            if t == 1:
                smooth_list.append(x)
                if len(smooth_list) == count:
                    return smooth_list
        return smooth_list

    # 1. Generate Initial Candidates
    
    candidates = []
    
    # Candidate 1: Smallest 20 5-smooth numbers
    cands_5 = smooth_vals[:n]
    candidates.append(cands_5)
        
    # Candidate 2: Smallest 20 7-smooth numbers
    cands_7 = get_smooth_numbers(7, max_val, n)
    if len(cands_7) == n:
        candidates.append(cands_7)

    # Candidate 3: Smallest 20 11-smooth numbers
    cands_11 = get_smooth_numbers(11, max_val, n)
    if len(cands_11) == n:
        candidates.append(cands_11)
        
    # Candidate 4: Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # Candidate 5: Random 5-smooth numbers
    for _ in range(3):
        random_indices = random.sample(range(len(smooth_vals)), n)
        cand = sorted([smooth_vals[i] for i in random_indices])
        candidates.append(cand)

    # 2. Lattice-based Local Search in Exponent Space
    
    def lattice_local_search(start_set, max_iter=50000):
        """Local search in the 5-smooth exponent lattice."""
        # Convert start_set to exponent triples
        current_exps = []
        for val in start_set:
            if val in val_to_idx:
                current_exps.append(smooth_exps[val_to_idx[val]])
            else:
                # Not 5-smooth, use nearest 5-smooth approximation
                # Find closest 5-smooth number
                best_idx = min(range(len(smooth_vals)), key=lambda i: abs(smooth_vals[i] - val))
                current_exps.append(smooth_exps[best_idx])
        
        # Deduplicate: if two exponents map to same value, adjust one
        seen_vals = set()
        unique_exps = []
        for exp in current_exps:
            val = (2**exp[0]) * (3**exp[1]) * (5**exp[2])
            if val not in seen_vals:
                unique_exps.append(exp)
                seen_vals.add(val)
        
        # If we don't have enough, fill with random 5-smooth numbers
        while len(unique_exps) < n:
            idx = random.randint(0, len(smooth_vals) - 1)
            exp = smooth_exps[idx]
            val = (2**exp[0]) * (3**exp[1]) * (5**exp[2])
            if val not in seen_vals:
                unique_exps.append(exp)
                seen_vals.add(val)
        
        current_exps = unique_exps[:n]
        current_vals = [(2**a) * (3**b) * (5**c) for (a, b, c) in current_exps]
        current_score = evaluate(current_vals)
        
        local_best_exps = current_exps[:]
        local_best_score = current_score
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random exponent triple to modify
            idx = random.randint(0, n - 1)
            a, b, c = current_exps[idx]
            
            # Generate neighbor in exponent space
            # Random walk in (a,b,c) space
            new_a = a + random.randint(-2, 2)
            new_b = b + random.randint(-2, 2)
            new_c = c + random.randint(-2, 2)
            
            # Clamp to valid ranges
            new_a = max(0, min(14, new_a))
            new_b = max(0, min(9, new_b))
            new_c = max(0, min(6, new_c))
            
            new_val = (2**new_a) * (3**new_b) * (5**new_c)
            
            # Check if new_val is already in current set
            if new_val in current_vals:
                continue
            
            # Check bounds
            if new_val < 1 or new_val > max_val:
                continue
            
            # Create new set
            new_exps = current_exps[:]
            new_exps[idx] = (new_a, new_b, new_c)
            new_vals = [(2**a2) * (3**b2) * (5**c2) for (a2, b2, c2) in new_exps]
            
            # Ensure all distinct
            if len(set(new_vals)) != n:
                continue
            
            new_score = evaluate(new_vals)
            
            if new_score < current_score:
                current_exps = new_exps
                current_vals = new_vals
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_exps = current_exps[:]
            else:
                # Simulated annealing
                if random.random() < 0.03:
                    if new_score < current_score * 1.3:
                        current_exps = new_exps
                        current_vals = new_vals
                        current_score = new_score
        
        # Convert best exponents back to values
        best_vals = sorted([(2**a) * (3**b) * (5**c) for (a, b, c) in local_best_exps])
        return best_vals, local_best_score

    # Run lattice local search from each candidate
    for cand in candidates:
        if time.time() > deadline:
            break
            
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            continue
            
        res_set, res_score = lattice_local_search(cand)
        
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    # 3. Final polish: standard local search on best set found
    if best_set is not None and time.time() < deadline - 5.0:
        current = best_set[:]
        current_score = best_score
        
        for i in range(20000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # Try replacing with a 5-smooth number
            if random.random() < 0.7:
                # Pick random 5-smooth number
                new_val = smooth_vals[random.randint(0, len(smooth_vals) - 1)]
            else:
                # Random number
                new_val = random.randint(1, max_val)
            
            if new_val in current:
                continue
            if new_val < 1 or new_val > max_val:
                continue
            
            temp_set = current[:]
            temp_set[idx] = new_val
            temp_set.sort()
            
            new_score = evaluate(temp_set)
            
            if new_score < current_score:
                current = temp_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            elif random.random() < 0.02 and new_score < current_score * 1.2:
                current = temp_set
                current_score = new_score

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set

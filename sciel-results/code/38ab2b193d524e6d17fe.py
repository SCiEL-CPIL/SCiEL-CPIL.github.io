import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    def score(A):
        # A is a tuple or list of 20 distinct ints
        sum_set = set()
        prod_set = set()
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        # Note: The set comprehension above adds a+b for i<=j.
        # Since addition/multiplication is commutative, a+b = b+a.
        # The set of all sums a+b for i,j in 0..n-1 is the same as i<=j.
        return max(len(sum_set), len(prod_set))
    
    def score_incremental(A, old_idx, old_val, new_val):
        """Compute new score efficiently by only updating affected pairs."""
        # First compute full sum_set and prod_set
        sum_set = set()
        prod_set = set()
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        
        # Remove old values involving old_idx
        for j in range(len(A)):
            if j == old_idx:
                continue
            old_sum = old_val + A[j]
            old_prod = old_val * A[j]
            # Check if this value appears elsewhere before removing
            # Simple approach: just remove and let duplicates be handled
            sum_set.discard(old_sum)
            prod_set.discard(old_prod)
        # Also remove old_val + old_val and old_val * old_val
        sum_set.discard(old_val + old_val)
        prod_set.discard(old_val * old_val)
        
        # Add new values
        for j in range(len(A)):
            if j == old_idx:
                continue
            new_sum = new_val + A[j]
            new_prod = new_val * A[j]
            sum_set.add(new_sum)
            prod_set.add(new_prod)
        sum_set.add(new_val + new_val)
        prod_set.add(new_val * new_val)
        
        return max(len(sum_set), len(prod_set))
    
    def generate_initial():
        # Try a few heuristics
        candidates = []
        # 1. AP
        candidates.append(tuple(range(1, 21)))
        # 2. GP
        gp = tuple(1 << i for i in range(20))
        if max(gp) <= 10000:
            candidates.append(gp)
        else:
            # Scale down GP? 2^19 is 524288 > 10000.
            # So pure GP of 20 terms with ratio 2 doesn't fit in 10000.
            # Max term must be <= 10000.
            # 2^13 = 8192. So we can have at most 14 terms of pure powers of 2.
            # For 20 terms, we need a different structure or smaller ratio.
            # Ratio 1.5? Not integers.
            # So GP is not a valid single-structure candidate for n=20, max=10000.
            pass
        
        # 3. Random
        for _ in range(5):
            candidates.append(tuple(random.sample(range(1, 10001), 20)))
            
        # Evaluate and return best
        best_c = None
        best_s = float('inf')
        for c in candidates:
            s = score(c)
            if s < best_s:
                best_s = s
                best_c = c
        return best_c, best_s

    current_A, current_score = generate_initial()
    best_A = current_A
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick a random index to modify
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Generate new value
        # Bias towards smaller numbers to encourage collisions?
        # Or random in [1, 10000]?
        # Let's try a mix: 50% small numbers [1, 100], 50% random [1, 10000]
        if random.random() < 0.5:
            new_val = random.randint(1, 100)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_A:
            continue
            
        # Create new tuple
        new_A_list = list(current_A)
        new_A_list[idx] = new_val
        new_A = tuple(new_A_list)
        
        new_score = score_incremental(current_A, idx, old_val, new_val)
        
        # Acceptance criterion
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if T > T_min:
                if random.random() < pow(2.718281828, -delta / T):
                    current_A = new_A
                    current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = 5.0 # Reheat occasionally? Or just stop.
            # Let's restart T if we are stuck?
            # Simpler: just let it cool.
            
    # Final local search: try all possible single-element replacements
    if time.time() < deadline - 5:
        for idx in range(20):
            old_val = current_A[idx]
            # Try replacing with nearby values
            for new_val in range(1, 101):
                if new_val == old_val or new_val in current_A:
                    continue
                new_A_list = list(current_A)
                new_A_list[idx] = new_val
                new_A = tuple(new_A_list)
                new_score = score_incremental(current_A, idx, old_val, new_val)
                if new_score < current_score:
                    current_A = new_A
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A
            if time.time() > deadline - 5:
                break
    
    return list(best_A)

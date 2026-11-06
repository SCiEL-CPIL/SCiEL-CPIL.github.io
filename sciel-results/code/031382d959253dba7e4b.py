import random
import itertools
import time

def is_sidon_4(a, b, c, d):
    # Check unique sums for i<=j
    s = [2*a, a+b, a+c, a+d, 2*b, b+c, b+d, 2*c, c+d, 2*d]
    return len(set(s)) == 10

def count_sidon_4(S):
    count = 0
    for subset in itertools.combinations(S, 4):
        if is_sidon_4(*subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    best_S = list(range(12))
    best_count = count_sidon_4(best_S)
    
    if best_count == 0:
        return best_S

    # Local search with incremental updates
    for restart in range(200):
        if time.time() - start_time > 55:
            break
            
        # Initialize S
        S = [random.randint(0, 128) for _ in range(12)]
        # Ensure distinct
        while len(set(S)) < 12:
            idx = random.randint(0, 11)
            S[idx] = random.randint(0, 128)
            
        current_count = count_sidon_4(S)
        
        if current_count < best_count:
            best_count = current_count
            best_S = S[:]
            
        if best_count == 0:
            return best_S

        # Local search loop
        stuck_counter = 0
        for step in range(2000):
            if time.time() - start_time > 55:
                break
                
            if current_count == 0:
                return S

            # Pick a random index to move
            idx = random.randint(0, 11)
            old_val = S[idx]
            
            # Try to find a better value for S[idx]
            best_new_val = old_val
            best_new_count = current_count
            
            # Sample a few random new values
            candidates = set()
            for _ in range(20):
                candidates.add(random.randint(0, 128))
            
            # Pre-compute affected subsets for this idx
            other_indices = [i for i in range(12) if i != idx]
            affected = []
            for combo in itertools.combinations(other_indices, 3):
                # vals[0] will be the value at idx
                vals = [S[idx]] + [S[i] for i in combo]
                affected.append(vals)
                
            # Count how many affected subsets are currently Sidon
            old_affected_count = 0
            for vals in affected:
                if is_sidon_4(*vals):
                    old_affected_count += 1
            
            # Evaluate candidates
            for new_val in candidates:
                if new_val == old_val or new_val in S:
                    continue
                
                new_affected_count = 0
                for vals in affected:
                    # vals[0] is the old value at idx
                    if is_sidon_4(new_val, vals[1], vals[2], vals[3]):
                        new_affected_count += 1
                
                # Unaffected count is current_count - old_affected_count
                new_total_count = (current_count - old_affected_count) + new_affected_count
                
                if new_total_count < best_new_count:
                    best_new_count = new_total_count
                    best_new_val = new_val
            
            if best_new_val != old_val:
                S[idx] = best_new_val
                current_count = best_new_count
                stuck_counter = 0
                if current_count < best_count:
                    best_count = current_count
                    best_S = S[:]
                    if best_count == 0:
                        return best_S
            else:
                stuck_counter += 1
                if stuck_counter > 50:
                    break # Restart
                    
    return best_S

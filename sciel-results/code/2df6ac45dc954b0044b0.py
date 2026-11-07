import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    N = 128
    
    # Use bitmask for fast membership testing
    # S_mask has bit i set if (i+1) is in the set
    def make_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask
    
    def in_set(mask, x):
        return (mask >> (x - 1)) & 1
    
    def is_valid_with_mask(mask, candidate, S_list):
        """Check if adding candidate to set represented by mask maintains validity.
        S_list is the list of elements currently in the set (not including candidate)."""
        
        # New mask would include candidate
        new_mask = mask | (1 << (candidate - 1))
        
        # 1. Check triples where candidate is a member: (candidate, x, y)
        #    Violation if (candidate+x), (candidate+y), (x+y) are all in new set
        for i in range(len(S_list)):
            x = S_list[i]
            for j in range(i + 1, len(S_list)):
                y = S_list[j]
                s1 = candidate + x
                s2 = candidate + y
                s3 = x + y
                if s1 <= N and s2 <= N and s3 <= N:
                    if in_set(new_mask, s1) and in_set(new_mask, s2) and in_set(new_mask, s3):
                        return False
        
        # 2. Check triples where candidate is a sum: (a, b, c) with a+b == candidate
        #    Violation if a+c and b+c are in S (original set, since candidate is the sum)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if a + b == candidate:
                    # Check if there is a c in S (c != a, c != b)
                    # such that a+c in S and b+c in S
                    for k in range(len(S_list)):
                        c = S_list[k]
                        if c == a or c == b:
                            continue
                        ac = a + c
                        bc = b + c
                        if ac <= N and bc <= N:
                            if in_set(mask, ac) and in_set(mask, bc):
                                return False
                            
        return True
    
    # Start with the safe base: {65, ..., 128}
    # This has 64 elements and no pair sum can be in S (all sums > 128)
    base = set(range(65, N + 1))
    base_list = sorted(base)
    base_mask = make_mask(base)
    
    # Greedy: try to add elements from 1 to 64
    S = set(base)
    S_list = list(base)
    S_mask = base_mask
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid_with_mask(S_mask, x, S_list):
            S.add(x)
            S_list.append(x)
            S_mask |= (1 << (x - 1))
    
    # Local search: try to add more elements by removing some
    # Try random restarts with different orderings
    best_S = set(S)
    best_list = list(S)
    best_mask = S_mask
    
    # Try multiple random orderings
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Start with base and try random ordering of small elements
        S_try = set(base)
        S_try_list = list(base)
        S_try_mask = base_mask
        small_elements = list(range(1, 65))
        rng.shuffle(small_elements)
        
        for x in small_elements:
            if time.time() > deadline:
                break
            if is_valid_with_mask(S_try_mask, x, S_try_list):
                S_try.add(x)
                S_try_list.append(x)
                S_try_mask |= (1 << (x - 1))
        
        if len(S_try) > len(best_S):
            best_S = set(S_try)
            best_list = list(S_try)
            best_mask = S_try_mask
    
    # Further local search: try to add any missing element
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        S_try = set(best_S)
        S_try_list = list(best_S)
        S_try_mask = best_mask
        improved = False
        
        # Try to add any element not in S_try
        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            if is_valid_with_mask(S_try_mask, x, S_try_list):
                S_try.add(x)
                S_try_list.append(x)
                S_try_mask |= (1 << (x - 1))
                improved = True
        
        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
            best_list = list(S_try)
            best_mask = S_try_mask
    
    # Swap-based local search: remove one element, try to add two
    for trial in range(200):
        if time.time() > deadline:
            break
        
        S_try = set(best_S)
        S_try_list = list(best_S)
        S_try_mask = best_mask
        
        # Try removing each element and see if we can add two others
        elems_to_try_remove = list(best_S)
        rng.shuffle(elems_to_try_remove)
        
        for rem in elems_to_try_remove:
            if time.time() > deadline:
                break
            
            # Remove rem
            S_try.discard(rem)
            S_try_list = [x for x in S_try_list if x != rem]
            S_try_mask &= ~(1 << (rem - 1))
            
            # Try to add missing elements
            missing = [x for x in range(1, N + 1) if x not in S_try]
            rng.shuffle(missing)
            
            added_count = 0
            for x in missing:
                if time.time() > deadline:
                    break
                if is_valid_with_mask(S_try_mask, x, S_try_list):
                    S_try.add(x)
                    S_try_list.append(x)
                    S_try_mask |= (1 << (x - 1))
                    added_count += 1
                    if added_count >= 2:
                        break
            
            # Only keep if we added at least 2 (net gain of 1)
            if added_count >= 2 and len(S_try) > len(best_S):
                best_S = set(S_try)
                best_list = list(S_try)
                best_mask = S_try_mask
            else:
                # Restore
                S_try.add(rem)
                S_try_list.append(rem)
                S_try_mask |= (1 << (rem - 1))
    
    # Simulated annealing phase: Max-SAT-inspired energy minimization
    # Energy = number of violated forbidden patterns - lambda * |S|
    # We flip single bits and accept based on Metropolis criterion
    
    def count_violations(mask, S_list):
        """Count the number of distinct forbidden triple patterns in the set."""
        violations = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    ab = a + b
                    ac = a + c
                    bc = b + c
                    if ab <= N and ac <= N and bc <= N:
                        if in_set(mask, ab) and in_set(mask, ac) and in_set(mask, bc):
                            violations += 1
        return violations
    
    # Start SA from the best solution found so far
    if time.time() < deadline - 10:
        S_sa = set(best_S)
        S_sa_list = list(best_S)
        S_sa_mask = best_mask
        
        # Compute initial energy
        initial_violations = count_violations(S_sa_mask, S_sa_list)
        lambda_penalty = 0.5  # penalty per violation to encourage validity
        initial_energy = initial_violations * lambda_penalty - len(S_sa)
        
        T_start = 2.0
        T_end = 0.01
        sa_steps = 5000
        sa_rng = random.Random(123)
        
        for step in range(sa_steps):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule: exponential cooling
            T = T_start * (T_end / T_start) ** (step / sa_steps)
            
            # Choose a random element to flip
            candidate = sa_rng.randint(1, N)
            
            if in_set(S_sa_mask, candidate):
                # Try removing candidate
                new_mask = S_sa_mask & ~(1 << (candidate - 1))
                new_list = [x for x in S_sa_list if x != candidate]
                new_violations = count_violations(new_mask, new_list)
                new_energy = new_violations * lambda_penalty - len(new_list)
                delta_E = new_energy - initial_energy
            else:
                # Try adding candidate
                new_mask = S_sa_mask | (1 << (candidate - 1))
                new_list = S_sa_list + [candidate]
                new_violations = count_violations(new_mask, new_list)
                new_energy = new_violations * lambda_penalty - len(new_list)
                delta_E = new_energy - initial_energy
            
            # Metropolis acceptance
            if delta_E < 0 or sa_rng.random() < math.exp(-delta_E / T):
                # Accept the move
                if in_set(S_sa_mask, candidate):
                    S_sa.discard(candidate)
                    S_sa_list = [x for x in S_sa_list if x != candidate]
                    S_sa_mask = new_mask
                else:
                    S_sa.add(candidate)
                    S_sa_list.append(candidate)
                    S_sa_mask = new_mask
                initial_energy = new_energy
                
                # If we found a valid larger set, update best
                if new_violations == 0 and len(S_sa) > len(best_S):
                    best_S = set(S_sa)
                    best_list = list(S_sa)
                    best_mask = S_sa_mask
        
        # After SA, try to clean up: remove elements that cause violations
        # to get back to a valid set if SA left us in an invalid state
        if count_violations(best_mask, best_list) > 0:
            # Fall back to the last known valid solution
            pass  # best_S should still be valid from previous phases
    
    # Final polish: try to add any remaining elements to the best valid set
    S_final = set(best_S)
    S_final_list = list(best_S)
    S_final_mask = best_mask
    missing_final = [x for x in range(1, N + 1) if x not in S_final]
    sa_rng2 = random.Random(456)
    sa_rng2.shuffle(missing_final)
    for x in missing_final:
        if time.time() > deadline - 2:
            break
        if is_valid_with_mask(S_final_mask, x, S_final_list):
            S_final.add(x)
            S_final_list.append(x)
            S_final_mask |= (1 << (x - 1))
    
    if len(S_final) > len(best_S):
        best_S = S_final
        best_list = S_final_list
        best_mask = S_final_mask
    
    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]

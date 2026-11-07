import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s hard limit
    sys.setrecursionlimit(10000)
    
    # Precompute: for each number, count its divisors in 1..256
    # Numbers with many divisors are "bad" - they create more divisibility conflicts
    
    # Key insight: if we pick numbers that are "prime-like" or have few divisors,
    # their subset sums will have fewer divisibility relations.
    
    # Strategy: Simulated Annealing with smart initialization
    # State: set of chosen elements
    # Move: swap one element with another, or add/remove
    
    def compute_sums(elements):
        """Compute all subset sums for a set of elements."""
        if not elements:
            return set()
        elems = sorted(elements)
        n = len(elems)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elems[i]
            sums.add(s)
        return sums
    
    def is_valid(sums):
        """Check if all subset sums are pairwise non-dividing."""
        if len(sums) != len(sums):  # duplicate check (set handles this)
            return False
        sl = sorted(sums)
        n = len(sl)
        for i in range(n):
            for j in range(i + 1, n):
                if sl[j] % sl[i] == 0:
                    return False
        return True
    
    def check_addition(current_sums, candidate):
        """
        Check if adding candidate to a set with current_sums is valid.
        Returns new_sums if valid, else None.
        Also checks that all new sums don't divide or get divided by existing ones.
        """
        new_sums = set()
        # New sums are: candidate, and s + candidate for each s in current_sums
        if candidate in current_sums:
            return None
        new_sums.add(candidate)
        
        for s in current_sums:
            ns = s + candidate
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        
        all_sums = current_sums | new_sums
        
        # Check divisibility: for each new sum, check against all sums
        for s in new_sums:
            for t in all_sums:
                if s == t:
                    continue
                if t % s == 0:
                    return None
                if s % t == 0:
                    return None
        
        return all_sums
    
    def compute_divisibility_score(elements):
        """
        Score a set: count how many divisibility pairs exist among subset sums.
        Lower is better. 0 means valid.
        """
        sums = compute_sums(elements)
        if not sums:
            return 0
        sl = sorted(sums)
        n = len(sl)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if sl[j] % sl[i] == 0:
                    count += 1
        return count
    
    # Candidate pool: numbers 3..256 (avoid 1 and 2 which are too small)
    # Actually, let's think about this more carefully.
    # The subset sums range from min_element to sum(all elements).
    # For a set of size k, we have 2^k - 1 sums.
    # These sums must all be in some range and pairwise non-dividing.
    
    # Better approach: start with a known good structure.
    # Superincreasing sequences have distinct subset sums.
    # Let's try: a, a*3, a*9, a*27, ... (powers of 3)
    # But we need them in 1..256.
    # 3, 9, 27, 81, 243 -> 5 elements. Sums range 3..363.
    # Check divisibility: 3|9, 3|27, etc. - FAILS!
    
    # So powers of 3 don't work because the elements themselves divide each other.
    # We need elements where no element divides another, AND no subset sum divides another.
    
    # Let's try: large primes. If all elements are primes > 128, 
    # then each element is > sum of all smaller elements? No, that's not guaranteed.
    
    # Actually, let's use a different strategy:
    # Pick numbers that are all in a narrow range, say 100-256.
    # Then subset sums range from ~100 to ~256*k.
    # For k=5, sums range 100-1280.
    # Divisibility is less likely when numbers are close together.
    
    # Let me try a completely different approach: 
    # Use numbers that are all odd and large, so their sums are also "spread out".
    
    # Simulated Annealing approach:
    # Start with a random set, then try to improve by swapping elements.
    
    # Phase 1: Find a good initial solution using greedy with multiple orderings
    candidates = list(range(3, 257))
    
    best_set = set()
    best_size = 0
    
    # Try different greedy strategies
    strategies = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        sorted(candidates, key=lambda x: x * (x + 1)),  # composite numbers
        sorted(candidates, key=lambda x: 1 if x > 128 else 0),  # prefer large
    ]
    
    for _ in range(50):
        strategies.append(candidates[:])
        random.shuffle(strategies[-1])
    
    for strat in strategies:
        if time.time() > deadline - 10.0:
            break
        
        current_elements = set()
        current_sums = set()
        
        for c in strat:
            result = check_addition(current_sums, c)
            if result is not None:
                current_elements.add(c)
                current_sums = result
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = set(current_elements)
    
    # Phase 2: Simulated Annealing to improve
    if time.time() < deadline - 5.0:
        # Start from best_set
        current_elements = set(best_set)
        current_sums = compute_sums(current_elements)
        
        # Compute initial score (number of divisibility violations)
        def get_score(elements):
            if not elements:
                return (0, 0)  # (violations, size)
            sums = compute_sums(elements)
            sl = sorted(sums)
            n = len(sl)
            violations = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if sl[j] % sl[i] == 0:
                        violations += 1
            return (violations, len(elements))
        
        current_score = get_score(current_elements)
        best_sa_set = set(current_elements)
        best_sa_score = current_score
        
        # SA parameters
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        # Candidate pool for swaps
        all_candidates = set(range(3, 257))
        
        step = 0
        while T > min_T and time.time() < deadline - 5.0:
            step += 1
            if step % 10000 == 0 and time.time() > deadline - 5.0:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.6 and current_elements:
                # Swap: remove one element, add another
                to_remove = random.choice(list(current_elements))
                pool = [c for c in all_candidates if c not in current_elements]
                if not pool:
                    continue
                to_add = random.choice(pool)
                
                new_elements = current_elements - {to_remove} | {to_add}
                new_sums = compute_sums(new_elements)
                new_score = get_score(new_elements)
                
                # Accept if better or with probability exp(-delta/T)
                delta = new_score[0] - current_score[0]
                if delta <= 0 or random.random() < (1.0 / (1.0 + delta / T if T > 0 else 1e-10)):
                    current_elements = new_elements
                    current_sums = new_sums
                    current_score = new_score
                    if current_score[0] == 0 and current_score[1] > best_sa_score[1]:
                        best_sa_set = set(current_elements)
                        best_sa_score = current_score
            elif move_type < 0.8 and len(current_elements) > 1:
                # Remove one element
                to_remove = random.choice(list(current_elements))
                new_elements = current_elements - {to_remove}
                new_sums = compute_sums(new_elements)
                new_score = get_score(new_elements)
                
                delta = new_score[0] - current_score[0]
                if delta <= 0 or random.random() < (1.0 / (1.0 + abs(delta) / T if T > 0 else 1e-10)):
                    current_elements = new_elements
                    current_sums = new_sums
                    current_score = new_score
                    if current_score[0] == 0 and current_score[1] > best_sa_score[1]:
                        best_sa_set = set(current_elements)
                        best_sa_score = current_score
            else:
                # Add one element
                pool = [c for c in all_candidates if c not in current_elements]
                if not pool:
                    continue
                to_add = random.choice(pool)
                new_elements = current_elements | {to_add}
                new_sums = compute_sums(new_elements)
                new_score = get_score(new_elements)
                
                delta = new_score[0] - current_score[0]
                if delta <= 0 or random.random() < (1.0 / (1.0 + delta / T if T > 0 else 1e-10)):
                    current_elements = new_elements
                    current_sums = new_sums
                    current_score = new_score
                    if current_score[0] == 0 and current_score[1] > best_sa_score[1]:
                        best_sa_set = set(current_elements)
                        best_sa_score = current_score
            
            T *= cooling
        
        if best_sa_score[0] == 0 and best_sa_score[1] > best_size:
            best_set = best_sa_set
            best_size = best_sa_score[1]
    
    # Phase 3: Local search - try all single swaps from best_set
    if time.time() < deadline - 2.0:
        current_elements = set(best_set)
        current_sums = compute_sums(current_elements)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for remove_c in list(current_elements):
                if time.time() > deadline - 2.0:
                    break
                reduced = current_elements - {remove_c}
                reduced_sums = compute_sums(reduced)
                
                for add_c in range(3, 257):
                    if add_c in current_elements:
                        continue
                    new_elements = reduced | {add_c}
                    new_sums = compute_sums(new_elements)
                    if is_valid(new_sums):
                        if len(new_elements) > len(current_elements):
                            current_elements = new_elements
                            current_sums = new_sums
                            improved = True
                            break
                if improved:
                    break
        
        if len(current_elements) > best_size:
            best_set = current_elements
            best_size = len(current_elements)
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        if is_valid(final_sums):
            return [int(x) for x in sorted(best_set)]
    
    # Fallback
    return [3]

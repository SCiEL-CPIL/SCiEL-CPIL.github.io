import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    best_set = []
    
    # Candidate pool: Large numbers are better.
    # Start with 128-256.
    candidates_base = list(range(128, 257))
    
    # We can also include some smaller numbers if they fit, but large numbers
    # have fewer multiples, making the "forbidden" set smaller and checks faster.
    
    max_restarts = 1000
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        candidates = candidates_base.copy()
        random.shuffle(candidates)
        
        current_set = []
        sums = set()
        # forbidden contains all multiples of current sums
        forbidden = set()
        
        max_possible_sum = 256 * len(candidates) # Upper bound for dynamic limit
        
        for c in candidates:
            # Check if c can be added
            # New sums will be: {c} U {s + c for s in sums}
            # We need to check if any new sum is in forbidden
            # And if any new sum is a multiple of any existing sum (covered by forbidden)
            # And if any existing sum is a multiple of any new sum (NOT covered by forbidden currently)
            
            # Wait, the `forbidden` set approach only checks if new sum is a multiple of old sum.
            # We also need to check if old sum is a multiple of new sum.
            # If new sum $s_{new}$ divides old sum $s_{old}$, then $s_{new}$ is a divisor of $s_{old}$.
            # This is harder to precompute.
            
            # However, if we always add elements such that the new sums are "large" relative to old sums?
            # Not necessarily.
            
            # Let's just do the full check for the new sums against all existing sums.
            # Since the set size is small, the number of sums is $2^k$.
            # If $k=10$, $2^{10}=1024$.
            # Checking divisibility: for each new sum, check against all old sums.
            # This is $O(2^k \cdot 2^k) = O(4^k)$.
            # For $k=10$, $4^{10} \approx 10^6$. Fast enough.
            # For $k=15$, $4^{15} \approx 10^9$. Too slow.
            
            # So we need the `forbidden` set optimization for one direction,
            # and a direct check for the other direction, or optimize both.
            
            # Direction 1: New sum $N$ is multiple of Old sum $O$.
            # Check: $N \in \text{forbidden}$.
            
            # Direction 2: Old sum $O$ is multiple of New sum $N$.
            # Check: $O \in \text{multiples}(N)$.
            # We can precompute divisors of all old sums?
            # Or, for each new sum $N$, check if any old sum $O$ is divisible by $N$.
            # This requires iterating old sums.
            
            # Let's try a hybrid.
            
            new_sums = [c] + [s + c for s in sums]
            
            # Check for duplicates in new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            valid = True
            
            # Check 1: New sum divides Old sum?
            # For each new sum n, check if any old sum s is a multiple of n.
            # This is expensive if done naively.
            # But note: if n > max(sums), then no old sum can be a multiple of n (since old sums < n).
            # If we pick large elements, new sums are often larger than old sums?
            # Not necessarily. c is ~128. s is ~128. s+c is ~256.
            # Old sums can be up to sum(current_set).
            
            # Let's just iterate.
            for n in new_sums:
                # Check against forbidden (n is multiple of some old sum)
                if n in forbidden:
                    valid = False
                    break
                
                # Check if n divides some old sum
                # Only need to check if n <= max(sums)
                if n <= max(sums) if sums else float('inf'):
                    for s in sums:
                        if s % n == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check 2: New sums divide each other?
            # Sort new_sums
            ns_sorted = sorted(new_sums)
            for i in range(len(ns_sorted)):
                a = ns_sorted[i]
                for j in range(i+1, len(ns_sorted)):
                    b = ns_sorted[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If valid, add
            current_set.append(c)
            sums.update(new_sums)
            
            # Update forbidden
            for n in new_sums:
                # Add multiples of n
                # Limit to reasonable max
                limit = 256 * 30 # 7680
                for m in range(n, limit + 1, n):
                    forbidden.add(m)
                    
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
                
    return [int(x) for x in best_set]

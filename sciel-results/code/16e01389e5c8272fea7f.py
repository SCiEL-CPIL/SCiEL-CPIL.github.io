import time
import random
import sys

# Deadline guard: 80s wall-clock (10s safety margin under 90s limit)
_DEADLINE = 80.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_validity(sums):
    """
    Check if the set of sums has no two distinct values where one divides the other.
    sums is a set of integers.
    """
    if not sums:
        return True
    
    # Sort to make checking easier
    sorted_sums = sorted(sums)
    n = len(sorted_sums)
    
    # For each sum, check if it divides any larger sum
    for i in range(n):
        s = sorted_sums[i]
        if s == 0: continue # Sums are non-empty, so > 0
        # Check multiples
        # max sum is limited, so this loop is small
        # We can just iterate through the rest of the list
        for j in range(i + 1, n):
            t = sorted_sums[j]
            if t % s == 0:
                return False
    return True

def _add_element(sums, x):
    """
    Returns the new set of sums if x is added, or None if invalid.
    Also checks validity incrementally.
    """
    new_sums = set(sums)
    new_sums.add(x)
    
    # Check x against existing sums
    for s in sums:
        if s % x == 0 or x % s == 0:
            return None
        new_sums.add(s + x)
        
    # Check new sums (s+x) against each other and against old sums
    # Note: s+x > s, so we only need to check if s+x divides some other sum, 
    # or if some other sum divides s+x.
    # Since all sums are positive, if A < B, we only check B % A == 0.
    
    # Let's just collect all new sums and check the whole set for simplicity 
    # given the small size (max ~500 sums for k=9).
    # Actually, incremental check is better.
    
    # Let's refine:
    # We have old sums S.
    # New sums are S' = S U {x} U {s+x | s in S}.
    # We know S is valid.
    # We need to check:
    # 1. x vs S: done above.
    # 2. {s+x} vs S: 
    #    For each s, let t = s+x.
    #    Check if t divides any u in S, or any u in S divides t.
    # 3. {s+x} vs {s'+x}:
    #    Check if s+x divides s'+x, or vice versa.
    
    added_sums = []
    for s in sums:
        t = s + x
        added_sums.append(t)
        
    # Check added_sums against S
    for t in added_sums:
        for u in sums:
            if t > u:
                if t % u == 0:
                    return None
            else:
                if u % t == 0:
                    return None
                    
    # Check added_sums against each other
    # Since added_sums are distinct (because S is distinct and x is fixed), 
    # we just need to check pairs.
    # To optimize, sort added_sums? Or just double loop.
    # Size of added_sums is |S|. |S| <= 2^k. For k=8, |S|=255. 255^2 is ~65k ops. Fast.
    for i in range(len(added_sums)):
        t1 = added_sums[i]
        for j in range(i + 1, len(added_sums)):
            t2 = added_sums[j]
            if t2 > t1:
                if t2 % t1 == 0:
                    return None
            else:
                if t1 % t2 == 0:
                    return None
                    
    # If we get here, it's valid
    return new_sums

def _search(target_size, pool, current_set, current_sums, depth, best_result):
    """
    Recursive backtracking to find a set of size target_size.
    """
    if len(current_set) == target_size:
        return list(current_set)
        
    if _remaining() < 1.0:
        return None
        
    remaining_slots = target_size - len(current_set)
    
    # Heuristic: try candidates in a specific order
    # We want to maximize the chance of finding a solution.
    # Randomization helps escape local minima.
    
    for x in pool:
        if x in current_set:
            continue
            
        # Pruning: if we can't fill the remaining slots with numbers > x, skip?
        # Simple pruning: if not enough numbers left in pool to fill slots, break.
        # Since we iterate through pool, we can count how many valid candidates remain.
        
        new_sums = _add_element(current_sums, x)
        if new_sums is not None:
            current_set.add(x)
            result = _search(target_size, pool, current_set, new_sums, depth + 1, best_result)
            if result:
                return result
            current_set.remove(x)
            
        if _remaining() < 1.0:
            return None
            
    return None

def construct():
    global _START
    _START = time.time()
    
    # We want to maximize size. Try 9, then 8, then 7.
    # The pool is 1..256.
    
    # To improve search efficiency, we can restrict the pool.
    # Small numbers are "dangerous" because they divide many things.
    # Large numbers are "safer".
    # Let's try searching with a pool of larger numbers first, or randomize.
    
    # Strategy:
    # 1. Try to find size 9.
    # 2. Try to find size 8.
    # 3. Return best found.
    
    best = []
    
    # Define a few different pools/orders to increase diversity
    # Pool 1: All numbers 1..256, shuffled
    # Pool 2: Numbers 50..256, shuffled (avoid small divisors)
    # Pool 3: Primes in 1..256
    
    pools = []
    
    # Pool 0: Random full range
    p0 = list(range(1, 257))
    random.seed(1)
    random.shuffle(p0)
    pools.append(p0)
    
    # Pool 1: Larger numbers
    p1 = list(range(30, 257))
    random.seed(2)
    random.shuffle(p1)
    pools.append(p1)
    
    # Pool 2: Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    p2 = [x for x in range(1, 257) if is_prime(x)]
    random.seed(3)
    random.shuffle(p2)
    pools.append(p2)
    
    # Pool 3: Numbers with few divisors (primes and prime powers)
    # Just use p2 for now.
    
    # Try to find size 9
    for pool in pools:
        if _remaining() < 5.0:
            break
        # We need to pick 9 numbers from the pool.
        # Backtracking with full pool 256 is too slow for depth 9.
        # We need to restrict the branching factor.
        
        # Instead of using the full pool at each step, we can sample a subset of candidates.
        # Or, we can sort the pool and only consider the top K candidates at each step.
        
        # Let's try a different approach: 
        # Generate random subsets of size 9 and check them? 
        # C(256, 9) is huge.
        
        # Better: Greedy with backtracking on a reduced candidate list.
        # At each step, pick the best candidate from a list of e.g. 50 random candidates.
        
        # Let's implement a randomized greedy with limited backtracking.
        
        found = False
        for attempt in range(20): # 20 restarts per pool
            if _remaining() < 2.0:
                break
                
            # Create a candidate list
            cand_list = list(pool)
            random.shuffle(cand_list)
            
            # We will try to build a set of size 9.
            # We will use a simple backtracking that only goes back 1 level? 
            # No, full backtracking is needed.
            
            # To make it fast, we limit the number of candidates considered at each depth.
            # E.g., at each step, only try the first 20 candidates from the shuffled list.
            
            current_set = set()
            current_sums = set()
            
            def bt(idx, depth):
                if len(current_set) == 9:
                    return True
                if idx >= len(cand_list):
                    return False
                if _remaining() < 0.5:
                    return False
                
                # Try candidates from idx onwards
                # Limit branching: only try next 10 candidates
                end_idx = min(idx + 10, len(cand_list))
                
                for i in range(idx, end_idx):
                    x = cand_list[i]
                    if x in current_set:
                        continue
                    
                    new_sums = _add_element(current_sums, x)
                    if new_sums is not None:
                        current_set.add(x)
                        if bt(i + 1, depth + 1):
                            return True
                        current_set.remove(x)
                        
                return False
            
            if bt(0, 0):
                best = list(current_set)
                found = True
                break
                
        if found:
            break # Found size 9, we are done searching for larger
            
    if not best:
        # Try size 8
        for pool in pools:
            if _remaining() < 5.0:
                break
            found = False
            for attempt in range(50):
                if _remaining() < 2.0:
                    break
                cand_list = list(pool)
                random.shuffle(cand_list)
                
                current_set = set()
                current_sums = set()
                
                def bt(idx, depth):
                    if len(current_set) == 8:
                        return True
                    if idx >= len(cand_list):
                        return False
                    if _remaining() < 0.5:
                        return False
                    
                    end_idx = min(idx + 15, len(cand_list))
                    
                    for i in range(idx, end_idx):
                        x = cand_list[i]
                        if x in current_set:
                            continue
                        
                        new_sums = _add_element(current_sums, x)
                        if new_sums is not None:
                            current_set.add(x)
                            if bt(i + 1, depth + 1):
                                return True
                            current_set.remove(x)
                            
                    return False
                
                if bt(0, 0):
                    best = list(current_set)
                    found = True
                    break
                    
            if found:
                break

    if not best:
        # Fallback to size 7 (known to be possible)
        # We can just return a hardcoded valid set of size 7 or run the search for 7.
        # Let's run search for 7 just in case.
        for pool in pools:
            if _remaining() < 2.0:
                break
            cand_list = list(pool)
            random.shuffle(cand_list)
            
            current_set = set()
            current_sums = set()
            
            def bt(idx, depth):
                if len(current_set) == 7:
                    return True
                if idx >= len(cand_list):
                    return False
                if _remaining() < 0.5:
                    return False
                
                end_idx = min(idx + 20, len(cand_list))
                
                for i in range(idx, end_idx):
                    x = cand_list[i]
                    if x in current_set:
                        continue
                    
                    new_sums = _add_element(current_sums, x)
                    if new_sums is not None:
                        current_set.add(x)
                        if bt(i + 1, depth + 1):
                            return True
                        current_set.remove(x)
                        
                return False
            
            if bt(0, 0):
                best = list(current_set)
                break

    # Final validation
    if best:
        sums = set()
        n = len(best)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best[i]
            sums.add(s)
        if not _check_validity(sums):
            best = [128, 192, 224] # Fallback

    return [int(x) for x in best]

# EVOLVE-BLOCK-END

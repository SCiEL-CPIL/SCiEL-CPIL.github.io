import time
import random

START = time.time()
DEADLINE = 55.0 # Leave 5s margin for 60s limit (evaluator kills at 90s, but task says 60s finish, usually strict)
# Task says: "Each evaluation must finish within 60 s."
# "The evaluator KILLS the program after 90s wall-clock."
# So we have 90s total, but must return within 60s?
# "Keep an EXPLICIT deadline guard... safety margin before the limit."
# I will use 55s as the internal deadline to be safe.

def _validate(vals):
    n = len(vals)
    # sums_by_k[i] is a bitset of sums for subsets of size i
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    
    # Check disjointness between different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _compute_sums(vals):
    n = len(vals)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    return sums_by_k

def _is_valid_add(vals, sums_by_k, c):
    """
    Check if adding c to vals (with current sums_by_k) results in a valid state.
    Returns True if valid, False otherwise.
    """
    n = len(vals)
    # New sums for k+1 will be old_sums[k+1] | (old_sums[k] << c)
    # We need to ensure that for all i != j, new_sums[i] & new_sums[j] == 0
    
    # The new bits added to class k+1 are: added_k = sums_by_k[k] << c
    # These bits must not intersect with any new_sums[m] for m != k+1.
    
    # Precompute the "added" bits for all k
    # added_bits[k] = sums_by_k[k] << c
    # Note: sums_by_k has length n+1 (indices 0..n).
    # When adding c, the new set has size n+1.
    # new_sums[0] = sums_by_k[0]
    # new_sums[k+1] = sums_by_k[k+1] | (sums_by_k[k] << c)
    
    # Check 1: The new bits in class k+1 must not intersect with existing bits in other classes.
    # The new bits are added_bits[k] = sums_by_k[k] << c.
    # They go into class k+1.
    # They must not intersect with new_sums[m] for m != k+1.
    
    # new_sums[m] for m != k+1:
    # If m == 0: new_sums[0] = sums_by_k[0]
    # If m > 0 and m != k+1: new_sums[m] = sums_by_k[m] | (sums_by_k[m-1] << c)
    #   Note: if m-1 == k, then the term (sums_by_k[m-1] << c) is added_bits[k].
    #   But we are checking intersection with m != k+1.
    
    # Let's just compute the full new_sums and check?
    # n is small (< 20). Computing new_sums is O(n) bit shifts.
    # Checking disjointness is O(n^2) bit ANDs.
    # This is very fast.
    
    new_sums = sums_by_k[:] # copy
    # Update in reverse to avoid using updated values? 
    # Actually, the formula is new_sums[j] = sums_by_k[j] | (sums_by_k[j-1] << c)
    # We can compute this directly.
    
    # new_sums[0] is unchanged
    for j in range(1, n + 2):
        if j <= n:
            new_sums[j] = sums_by_k[j] | (sums_by_k[j-1] << c)
        else: # j == n+1
            new_sums[j] = (sums_by_k[n] << c)
            # Note: sums_by_k list needs to be extended or handled carefully
            # My sums_by_k passed in has length n+1.
            # I need a temporary list of length n+2.
            
    # Let's rewrite _is_valid_add to be self-contained and safe.
    temp_sums = [0] * (n + 2)
    temp_sums[0] = sums_by_k[0]
    for j in range(1, n + 2):
        prev = sums_by_k[j-1] if j-1 <= n else 0
        curr = sums_by_k[j] if j <= n else 0
        temp_sums[j] = curr | (prev << c)
        
    # Check disjointness
    for i in range(n + 2):
        for k in range(i + 1, n + 2):
            if temp_sums[i] & temp_sums[k]:
                return False
    return True

def _try_greedy_construct(target_size, rng, time_limit):
    """
    Try to construct a set of size target_size using a randomized greedy approach 
    with backtracking limited to depth 1 or 2, or just simple greedy with restarts.
    Here we do a simple greedy: pick random candidates, add if valid.
    """
    avail = list(range(1, 257))
    rng.shuffle(avail)
    current = []
    sums_by_k = [1] + [0]*target_size # Pre-allocate worst case
    
    # We need to dynamically update sums_by_k.
    # Let's just maintain the list and recompute sums if we remove? 
    # No, for greedy add, we just check validity.
    
    # To make it faster, we maintain sums_by_k.
    # But if we fail to add, we don't need to undo if we just iterate.
    
    # This simple greedy often fails to reach high sizes because of early bad choices.
    # We will use this as a starting point for Local Search.
    
    for c in avail:
        if len(current) >= target_size:
            break
        if _is_valid_add(current, sums_by_k, c):
            current.append(c)
            # Update sums_by_k
            # We need to update the bitsets.
            # new_sums[j] = old_sums[j] | (old_sums[j-1] << c)
            # We iterate j from len(current) down to 1?
            # Actually, we can just recompute or update in place carefully.
            # Since n is small, let's just update in a new list and replace.
            new_sums = [0] * (len(current) + 1)
            new_sums[0] = 1
            for j in range(1, len(current) + 1):
                # The last element added is c.
                # Sums of size j in current either don't include c (from old sums[j]) 
                # or include c (from old sums[j-1] + c).
                # But we are adding sequentially.
                # It's easier to just recompute from scratch if n is small?
                # Or update:
                pass
            
            # Efficient update:
            # old_sums has length len(current) (before adding c)
            # new_sums has length len(current) + 1
            # new_sums[0] = 1
            # for j in 1..len(current):
            #   new_sums[j] = old_sums[j] (if exists) | (old_sums[j-1] << c)
            
            # Let's implement this update properly.
            old_len = len(current) - 1
            new_sums = [0] * (old_len + 2)
            new_sums[0] = 1
            for j in range(1, old_len + 1):
                new_sums[j] = sums_by_k[j] | (sums_by_k[j-1] << c)
            new_sums[old_len + 1] = (sums_by_k[old_len] << c)
            
            sums_by_k = new_sums

    return current, sums_by_k

def construct():
    global START, DEADLINE
    
    # We want to maximize the size.
    # Let's try to find a set of size 16, 17, etc.
    # Strategy: Iterated Local Search (ILS)
    
    best_set = []
    best_size = 0
    
    # Initialize with a greedy solution
    rng = random.Random(42)
    # Try to get a decent start
    start_set, start_sums = _try_greedy_construct(20, rng, 1.0)
    best_set = start_set
    best_size = len(best_set)
    
    # If greedy didn't find much, try a few random restarts
    for _ in range(5):
        if time.time() - START > DEADLINE - 5:
            break
        s, _ = _try_greedy_construct(20, rng, 0.5)
        if len(s) > best_size:
            best_set = s
            best_size = len(best_set)
            
    # Local Search / Hill Climbing
    # We maintain current_set and current_sums.
    # Move: 
    # 1. Try to add a random unused element.
    # 2. If fails, try to swap: remove random existing, add random unused.
    
    current_set = list(best_set)
    current_sums = _compute_sums(current_set)
    used = [False] * 257
    for v in current_set:
        used[v] = True
    
    unused_pool = [v for v in range(1, 257) if not used[v]]
    
    # We will loop until deadline
    while time.time() - START < DEADLINE:
        # 1. Try to add
        if unused_pool:
            c = rng.choice(unused_pool)
            if _is_valid_add(current_set, current_sums, c):
                # Accept
                current_set.append(c)
                used[c] = True
                unused_pool.remove(c)
                
                # Update sums
                old_len = len(current_set) - 1
                new_sums = [0] * (old_len + 2)
                new_sums[0] = 1
                for j in range(1, old_len + 1):
                    new_sums[j] = current_sums[j] | (current_sums[j-1] << c)
                new_sums[old_len + 1] = (current_sums[old_len] << c)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(best_set)
                    
        # 2. Try to swap (if we can't add, or to improve configuration)
        # We only swap if we are not making progress or to escape local optima.
        # Let's try swapping if we haven't added in the last few steps? 
        # Simpler: Just try swapping with some probability.
        if rng.random() < 0.3 and unused_pool and current_set:
            # Pick random existing e to remove
            e_idx = rng.randint(0, len(current_set) - 1)
            e = current_set[e_idx]
            
            # Pick random unused c to add
            c = rng.choice(unused_pool)
            
            # Check if removing e and adding c is valid.
            # This is tricky to check incrementally without recomputing.
            # Since N is small, we can just recompute the set and check validity.
            
            new_set = [v for i, v in enumerate(current_set) if i != e_idx]
            new_set.append(c)
            
            # Check validity of new_set
            if _validate(new_set):
                # Accept
                current_set = new_set
                current_sums = _compute_sums(current_set)
                
                # Update used/unused
                used[e] = False
                unused_pool.append(e)
                used[c] = True
                unused_pool.remove(c)
                
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(best_set)
        elif rng.random() < 0.1 and unused_pool:
             # Try to add again, maybe different element
             pass # Already tried add at top
            
        # Optional: Perturbation
        # If stuck, remove a few random elements and re-greedy?
        # This might be expensive. Let's stick to Add/Swap.
        
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found size: {len(res)}")
    print(res)
    print("Valid:", _validate(res))

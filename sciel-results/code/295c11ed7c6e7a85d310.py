import time
import random

def construct():
    """
    Find a large subset S of [1, 256] such that for all a < b <= c in S, a does NOT divide b+c.
    Maximize |S|.
    """
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin
    
    # Initial valid set: [171, 256]
    # For any a in this set, 3a >= 513 > 512 = 2*256, so a cannot divide b+c (since b+c <= 512).
    # Size = 86.
    current_set = set(range(171, 257))
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Precompute valid pairs for a given set S
    # We want to add elements x < 171.
    # Conflict if x divides b+c for some b, c in S with x < b <= c.
    
    def check_valid_addition(x, S):
        # Check if x can be added to S
        # Condition: for all b, c in S with x < b <= c, x does NOT divide b+c.
        # Note: b, c must be in S.
        # Also, we need to ensure that the existing elements of S remain valid with respect to each other.
        # Since S is currently valid, we only need to check conflicts involving x.
        # The condition "a must NOT divide b+c" applies to all a < b <= c in the new set.
        # If x is added, x becomes a new 'a' candidate.
        # Also, could x become a 'b' or 'c' for some existing a?
        # If x > a, and x <= c, then a might divide x+c or a might divide x+c?
        # Wait, the condition is: for every a < b <= c in the set, a NOT divide b+c.
        # If we add x, new triples can be formed where x is a, b, or c.
        # 1. x is a: x < b <= c. Check if x divides b+c for any b,c in S.
        # 2. x is b: a < x <= c. Check if a divides x+c for any a,c in S with a < x.
        # 3. x is c: a < b < x (since b <= c=x). Check if a divides b+x for any a,b in S with a < b.
        
        # Optimization:
        # If x is small, many a < x.
        # Let's iterate through S.
        
        S_list = sorted(S)
        
        # Case 1: x is 'a'
        # Need x < b <= c. So b, c must be > x.
        # If x >= 171, this is easy? No, x < 171 here.
        # b, c are in S.
        # We need to check if there exist b, c in S such that x < b <= c and (b+c) % x == 0.
        
        # Case 2: x is 'b'
        # Need a < x <= c.
        # a in S, a < x. c in S, c >= x.
        # Check if a divides x+c.
        
        # Case 3: x is 'c'
        # Need a < b < x (since b <= c=x and b>a, so b<x).
        # a, b in S.
        # Check if a divides b+x.
        
        # Let's combine these checks efficiently.
        # Since S is a subset of [1, 256], and we are adding x < 171, 
        # most elements in S are > x (since S starts at 171).
        
        # Let's just iterate over all pairs in S and check conditions.
        # This might be slow if done for every x in every iteration.
        # Let's try to be smarter.
        
        # For x to be a valid addition:
        # 1. For all b, c in S with x < b <= c: (b+c) % x != 0
        # 2. For all a in S with a < x, and c in S with c >= x: (x+c) % a != 0
        # 3. For all a, b in S with a < b < x: (b+x) % a != 0
        
        # Note: In our current set S = [171, 256] + some smaller ones.
        # Most elements are > x.
        # So case 2 and 3 involve a, b < x. There are few such elements.
        # Case 1 involves b, c > x. There are many such elements.
        
        # Let's precompute for S:
        # For each potential x, we can check conflicts.
        
        # To speed up, let's filter candidates.
        # We will try to add elements from 1 to 170.
        # We can use a greedy approach or local search.
        
        pass # This function is just a placeholder for logic, I will implement the main loop properly.

    def is_valid_set(S):
        """Check if S is a valid set."""
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            # Check b, c such that a < b <= c
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # To speed up checks, we can precompute divisibility or use bitsets?
    # For n=256, O(N^3) is 256^3 ~ 16M. In Python this is too slow for many iterations.
    # We need a faster check.
    
    # Optimization for is_valid:
    # For a given a, we need to check if there exist b, c in S, a < b <= c, such that b+c is a multiple of a.
    # Let M_a be the set of multiples of a in range [2(a+1), 2*256].
    # Actually, min b+c = (a+1) + (a+1) = 2a+2. Max b+c = 256+256=512.
    # So we check if any multiple k*a is in [2a+2, 512] such that it can be written as b+c with b,c in S, a < b <= c.
    
    def can_add(x, S):
        """
        Check if x can be added to S.
        S is a set of ints.
        """
        # x must not be in S
        if x in S:
            return False
            
        # Condition 1: x as 'a'
        # For all b, c in S with x < b <= c, (b+c) % x != 0
        # Since S is sorted, we can find elements > x.
        # Let's just check pairs. To speed up, note that if x is small, many b,c.
        # If x is large, few b,c.
        
        # We'll do a direct check but break early.
        
        # Sort S for easier access? S is a set, let's convert to list once outside if possible, 
        # but S changes. Let's keep S as a set for O(1) lookup, but iterate over values.
        
        S_list = sorted(S)
        
        # Check x as 'a'
        # Find indices where element > x
        # Since x < 171 and most of S is > 171, most of S is > x.
        
        # Optimization: Precompute sum sets?
        # Let's try a heuristic: only check pairs (b,c) where b+c is a multiple of x.
        # Multiples of x in [2*x+2, 512]:
        # m = ceil((2*x+2)/x)*x, ..., floor(512/x)*x
        # For each m, check if m = b+c with b,c in S, x < b <= c.
        
        # This is faster than checking all pairs.
        
        start_m = ((2*x + 2 + x - 1) // x) * x
        end_m = (512 // x) * x
        
        for m in range(start_m, end_m + 1, x):
            # Check if m = b + c with b, c in S, x < b <= c
            # b must be in [max(x+1, m - max(S)), min(m - (x+1), max(S))]
            # Actually, b > x and c >= b => c >= m-b => b >= m-c.
            # Also c in S.
            # Let's iterate b in S such that b > x.
            # For each b, c = m - b.
            # Check if c in S and c >= b.
            
            # To make this faster, we can iterate over the smaller range of b.
            # b < c => b < m/2.
            # b >= m - max(S).
            
            # Let's just iterate b in S_list where b > x and b < m/2 (since b <= c implies b <= m/2).
            # Also b must be such that c = m-b is in S.
            
            # Since S is a set, lookup is O(1).
            # Number of b's to check is roughly |S|/2.
            # Number of multiples m is roughly 512/x.
            # Total checks ~ |S| * 512/x.
            # For x=1, 512 checks * 86 ~ 44k.
            # For x=100, 5 checks * 86 ~ 430.
            # This is acceptable for a single check.
            # Doing this for 170 candidates * many iterations might be slow.
            
            # Let's implement this specific check.
            
            # We need to be careful: b and c are from the CURRENT set S.
            
            # Optimization: If x is small, this is the bottleneck.
            # We can precompute for each x a list of forbidden sums? No, S changes.
            
            for b in S_list:
                if b <= x:
                    continue
                if b > m // 2: 
                    break # Assuming S_list is sorted. If b > m/2, then c < b, so we would have found it when b was smaller?
                    # Wait, we need b <= c. So b <= m/2.
                    # If S_list is sorted, we can break.
                c = m - b
                if c < b:
                    continue
                if c in S:
                    return False
            # Note: The loop above breaks if b > m//2. But we need to ensure we check all b in S that are > x and <= m//2.
            # Since S_list is sorted, we can use bisect.
            
        # Let's redo the check with bisect for efficiency.
        # import bisect
        # idx_start = bisect.bisect_right(S_list, x)
        # idx_end = bisect.bisect_right(S_list, m // 2)
        # for idx in range(idx_start, idx_end):
        #     b = S_list[idx]
        #     c = m - b
        #     if c in S: return False
        
        # I will use this in the final code.
        
        # Condition 2: x as 'b'
        # a < x <= c. a in S, c in S.
        # (x+c) % a == 0.
        # a must be < x.
        # Let's find all a in S with a < x.
        # For each a, check if there exists c in S with c >= x such that (x+c) % a == 0.
        # x+c is a multiple of a.
        # Let y = x+c. y % a == 0.
        # y must be in [x+x, x+256] = [2x, x+256].
        # Also c = y - x. c in S.
        
        # Condition 3: x as 'c'
        # a < b < x. a, b in S.
        # (b+x) % a == 0.
        # a, b < x.
        
        # If S only contains large numbers, conditions 2 and 3 are trivial (no a < x in S).
        # As we add smaller numbers, we need to check these.
        
        # For now, let's assume we only add numbers that are compatible.
        # We will use a greedy approach with backtracking or local search.
        
        # Given the time limit, let's stick to a robust local search.
        # Start with [171, 256].
        # Try to add numbers 1..170 in some order.
        # If adding x fails, skip it or try removing a conflicting element and re-adding.
        
        # A simpler strategy:
        # 1. Start with [171, 256].
        # 2. Iterate x from 1 to 170.
        # 3. If x can be added, add it.
        # 4. After one pass, try to add remaining numbers again (in case order mattered? No, adding x might block y, but adding y might unblock x? No, constraints are monotonic: adding more elements only adds constraints. So if x can't be added now, it can't be added later unless we REMOVE elements from S.)
        # 5. So, if we want to improve, we must REMOVE elements from the current set to allow more elements to be added.
        
        # Let's try a swap-based local search.
        # For each x not in S (x < 171), and each y in S,
        # try swapping y for x.
        # If the new set is valid and size is same (swap) or we can add x and remove y, it's neutral.
        # But we want to INCREASE size.
        # So, if we remove y, can we add x AND another z?
        
        # Alternative:
        # Find a set of elements in S that are "conflicting" with a set of small elements.
        # If we remove 1 element, can we add 2?
        
        # Let's try to just add elements greedily first.
        # Then try to replace 2 elements with 3?
        
        # Given the time, let's implement the greedy add, and then a simple hill climbing:
        # Try to remove a random element y from S, then greedily add any available x that becomes valid.
        # If size increases, keep it.
        
        # Implementation plan:
        # 1. S = [171..256]
        # 2. Greedy add: for x in 1..170: if can_add(x, S): S.add(x)
        # 3. Local search loop:
        #    while time remaining:
        #       pick random y in S.
        #       S.remove(y)
        #       old_size = len(S)
        #       # Greedily add available numbers
        #       for x in 1..256:
        #           if x not in S and can_add(x, S):
        #               S.add(x)
        #       if len(S) > best_size:
        #           best_set = S
        #           best_size = len(S)
        #       else:
        #           # Restore S? Or keep the improved local set even if not global best?
        #           # If we didn't improve, we should probably revert or try a different y.
        #           # Reverting is safer to avoid getting stuck in bad local minima?
        #           # Actually, if we don't improve, the set might still be "better" in terms of potential?
        #           # Let's just keep the current S (which includes the removal and additions) and continue.
        #           # Wait, if size didn't increase, we lost y and maybe gained nothing.
        #           # Let's revert if size didn't increase.
        
        return list(best_set)

# Let's refine the can_add function for speed.

import bisect

def can_add_fast(x, S_set, S_list):
    # S_set: set of elements
    # S_list: sorted list of elements in S_set
    
    # 1. x as 'a'
    # Check if exists b, c in S such that x < b <= c and (b+c) % x == 0.
    start_m = ((2*x + 2 + x - 1) // x) * x
    end_m = (512 // x) * x
    
    for m in range(start_m, end_m + 1, x):
        # b <= c => b <= m/2
        # b > x
        # c = m - b in S
        # b in S
        
        # Find range of b in S_list
        # b > x and b <= m//2
        idx_start = bisect.bisect_right(S_list, x)
        idx_end = bisect.bisect_right(S_list, m // 2)
        
        for i in range(idx_start, idx_end):
            b = S_list[i]
            c = m - b
            if c in S_set:
                return False
    
    # 2. x as 'b'
    # a < x <= c. a in S, c in S.
    # (x+c) % a == 0.
    # a in S, a < x.
    # c in S, c >= x.
    
    # Get a's
    idx_a_end = bisect.bisect_left(S_list, x)
    for i in range(idx_a_end):
        a = S_list[i]
        # Check if exists c in S, c >= x, such that (x+c) % a == 0.
        # x+c = k*a.
        # c = k*a - x.
        # c >= x => k*a - x >= x => k*a >= 2x => k >= 2x/a.
        # c <= 256 => k*a - x <= 256 => k*a <= 256 + x.
        
        k_min = (2*x + a - 1) // a
        k_max = (256 + x) // a
        
        for k in range(k_min, k_max + 1):
            c = k*a - x
            if c in S_set:
                return False
                
    # 3. x as 'c'
    # a < b < x. a, b in S.
    # (b+x) % a == 0.
    
    # a, b < x.
    idx_x = bisect.bisect_left(S_list, x)
    
    # Iterate a < x
    for i in range(idx_x):
        a = S_list[i]
        # Check if exists b in S, a < b < x, such that (b+x) % a == 0.
        # b+x = k*a.
        # b = k*a - x.
        # a < b < x => a < k*a - x < x.
        # a + x < k*a => k > 1 + x/a.
        # k*a - x < x => k*a < 2x => k < 2x/a.
        
        k_min = (a + x) // a + 1
        k_max = (2*x - 1) // a
        
        for k in range(k_min, k_max + 1):
            b = k*a - x
            if a < b < x and b in S_set:
                return False

    return True

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0
    
    # Initial set
    S_set = set(range(171, 257))
    S_list = sorted(S_set)
    best_set = set(S_set)
    best_size = len(S_set)
    
    # Helper to update S_list
    def update_S_list():
        nonlocal S_list
        S_list = sorted(S_set)
        
    # Greedy addition
    for x in range(1, 171):
        if can_add_fast(x, S_set, S_list):
            S_set.add(x)
            update_S_list()
            if len(S_set) > best_size:
                best_set = set(S_set)
                best_size = len(S_set)
    
    # Local Search
    # We will try to remove an element and re-add greedily.
    # To avoid too slow, we limit iterations or check time.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        if attempts % 100 == 0:
            # Check time more frequently if needed
            if time.time() - start_time > time_limit - 5:
                break
        
        # Pick a random element to remove
        # Prefer removing larger elements? Or random?
        # Let's try random.
        y = random.choice(list(S_set))
        S_set.remove(y)
        update_S_list()
        
        old_size = len(S_set)
        
        # Greedily add any x not in S_set
        # Order: maybe try smaller x first? Or random?
        # Let's try adding in increasing order of x.
        for x in range(1, 257):
            if x not in S_set:
                if can_add_fast(x, S_set, S_list):
                    S_set.add(x)
                    update_S_list()
        
        new_size = len(S_set)
        
        if new_size > best_size:
            best_set = set(S_set)
            best_size = new_size
            
        # If we didn't improve, should we revert?
        # If we revert, we might miss a path that requires a temporary decrease.
        # But greedy re-add is deterministic.
        # Let's just keep the current state and continue.
        # However, if we keep making random moves that don't improve, we might wander.
        # Let's add a step: if size didn't increase, revert to best_set?
        # No, best_set is the global best. Current S_set might be different.
        # Let's just continue from current S_set.
        
    return list(best_set)

if __name__ == "__main__":
    # Quick test
    s = construct()
    # Verify validity
    s_set = set(s)
    s_list = sorted(s_set)
    valid = True
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i+1, len(s_list)):
            b = s_list[j]
            for k in range(j, len(s_list)):
                c = s_list[k]
                if (b+c) % a == 0:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Size: {len(s)}, Valid: {valid}")

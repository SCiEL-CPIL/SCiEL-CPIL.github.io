import time
import random
import sys

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    best_set = []
    best_size = 0

    # Precompute divisibility relations for all possible sums 1..30*256=7680
    # We need to check: for any two distinct sums a, b in our set, a does not divide b.
    # This is equivalent to: if a < b, then b % a != 0.
    
    # Since sums are bounded by 30*256 = 7680, we can precompute a list of divisors
    # or just use modulo. Modulo is fast enough in Python if we are smart.
    # However, the number of subset sums for n=30 is 2^30 - 1 ≈ 1e9, which is too many to store.
    # Wait, the task says "Maximize set size".
    # The current program found size 2 or 3.
    # Is it possible to have larger sets?
    # Let's analyze the constraints.
    # 1. All non-empty subset sums must be distinct.
    #    This implies the set is "sum-distinct".
    #    A necessary condition is that the elements are super-increasing?
    #    Not strictly, but it helps. If a_1 < a_2 < ... < a_n, and sum_{i<k} a_i < a_k,
    #    then all subset sums are distinct.
    #    Super-increasing sequences have size limited by the max value.
    #    a_1 >= 1
    #    a_2 > a_1
    #    a_3 > a_1 + a_2
    #    a_k > sum_{i<k} a_i
    #    a_k >= 2^{k-1}
    #    Since a_k <= 256, 2^{k-1} <= 256 => k-1 <= 8 => k <= 9.
    #    So a super-increasing set can have at most 9 elements.
    #    Example: 1, 2, 4, 8, 16, 32, 64, 128. (Size 8). Next would be 256. Size 9.
    #    Let's check if {1, 2, 4, 8, 16, 32, 64, 128, 256} is valid regarding divisibility.
    #    Sums range from 1 to 511.
    #    Divisibility check:
    #    1 divides everything. So 1 cannot be in the set if there is any other sum.
    #    So 1 is forbidden if n > 1.
    #    If we drop 1, start with 2?
    #    {2, 4, 8, 16, 32, 64, 128, 256}.
    #    Sums are even numbers from 2 to 510.
    #    2 divides 4, 6, ... So 2 is forbidden if 4 is in the set (sum 4 exists).
    #    In fact, if a is in the set, and 2a is a subset sum, it's invalid.
    #    In a super-increasing set of powers of 2, every element is a subset sum of smaller ones?
    #    No, in {2,4,...}, 4 is an element. Is 4 a sum of others? No, 2 is the only smaller. 2!=4.
    #    But 2 divides 4. 4 is a sum (the element itself). So {2,4} is invalid.
    #    Generally, if x is in the set, and 2x is in the set, invalid.
    #    If x is in the set, and x+y = 2x (i.e., y=x), but elements are distinct.
    #    So we just need to ensure no sum is a multiple of another sum.
    
    # Let's try to construct a set manually or via search.
    # Key insight: Large numbers are better because their multiples are less likely to be in the sum set.
    # Also, avoid small numbers that divide many things.
    
    # Let's use a backtracking search with strong pruning.
    # State: current set S, current set of sums Sums.
    # Candidate x:
    #   New sums = Sums U {x + s for s in Sums}
    #   Check:
    #     1. No duplicates in New sums. (Implies x not in Sums and x+s1 != x+s2 for s1!=s2, which is true).
    #        Actually, x+s1 == x+s2 => s1==s2. So only need to check if any new sum is in old Sums.
    #     2. No divisibility between any two sums in New sums.
    
    # Optimization for Divisibility Check:
    # Instead of checking all pairs, we can maintain a list of sums.
    # When adding new sums, check each new sum against all existing sums.
    # Also check new sums against each other.
    
    # To speed up, we can precompute for each possible sum value v (1..7680)
    # a bitset of its multiples? Or just iterate.
    # Since n is small (max ~10), the number of sums is 2^n.
    # For n=10, 1024 sums. Checking pairs is 1024^2 / 2 ≈ 500k operations.
    # This is fast in Python if done carefully? Maybe.
    # But we do this for each step of backtracking.
    # Depth 10, branching factor 256. 256^10 is huge. We need pruning.
    
    # Pruning:
    # If we add x, the new sums are x and x+s.
    # We can check divisibility incrementally.
    
    # Let's try a randomized local search or greedy with backtracking.
    # Given the time limit, a simple greedy with random restarts might work if we are smart.
    
    # Let's try to find a set of size 5 or 6.
    # Known small valid sets:
    # [20, 21, 23] size 3.
    # Can we add more?
    # Sums of [20, 21, 23]: 20, 21, 23, 41, 43, 44, 64.
    # Try adding 25?
    # New sums: 25, 25+20=45, 25+21=46, 25+23=48, 25+41=66, 25+43=68, 25+44=69, 25+64=89.
    # Check duplicates: None in old sums.
    # Check divisibility:
    # Old: 20, 21, 23, 41, 43, 44, 64
    # New: 25, 45, 46, 48, 66, 68, 69, 89
    # 20 divides 40, 60, 80... 45 no, 46 no, 48 no, 66 no, 68 no, 69 no, 89 no.
    # 21 divides 42, 63, 84... 45 no, 46 no, 48 no, 66 no, 68 no, 69 no, 89 no.
    # 23 divides 46. 46 is in new sums! Invalid.
    
    # Try adding 26?
    # New sums: 26, 46, 47, 49, 67, 69, 70, 90.
    # 23 divides 46. Invalid.
    
    # It seems hard to add to [20, 21, 23].
    
    # Let's try a different base.
    # Avoid small numbers.
    # Try [100, 101, 103]?
    # Sums: 100, 101, 103, 201, 203, 204, 304.
    # Divisibility:
    # 100 div 200, 300... 201 no, 203 no, 204 no, 304 no.
    # 101 div 202, 303... 201 no, 203 no, 204 no, 304 no.
    # 103 div 206, 309... 201 no, 203 no, 204 no, 304 no.
    # Valid. Size 3.
    
    # Can we add 107?
    # New sums: 107, 207, 208, 210, 308, 310, 311, 411.
    # Check against old:
    # 100 div 200, 300... 207 no, 208 no, 210 no, 308 no, 310 no, 311 no, 411 no.
    # 101 div 202, 303... 207 no, 208 no, 210 no, 308 no, 310 no, 311 no, 411 no.
    # 103 div 206, 309... 207 no, 208 no, 210 no, 308 no, 310 no, 311 no, 411 no.
    # 201 div 402... 411 no.
    # 203 div 406... 411 no.
    # 204 div 408... 411 no.
    # 304 div 608...
    # Check new sums against each other:
    # 107 div 214... 207 no, 208 no, 210 no, 308 no, 310 no, 311 no, 411 no.
    # 207 div 414...
    # 208 div 416...
    # 210 div 420...
    # 308 div 616...
    # 310 div 620...
    # 311 div 622...
    # Seems valid!
    # Set: [100, 101, 103, 107]. Size 4.
    
    # Let's verify [100, 101, 103, 107] in code.
    # Sums:
    # 100, 101, 103, 107
    # 201, 203, 207
    # 204, 210, 214
    # 304, 308, 311
    # 412
    # Total 15 sums.
    # Sorted: 100, 101, 103, 107, 201, 203, 204, 207, 210, 214, 304, 308, 311, 412.
    # Wait, 100+101+103+107 = 411?
    # 100+101=201. 201+103=304. 304+107=411.
    # My manual list had 412. 100+101+103+107 = 411.
    # Let's re-calculate sums.
    # Subsets of size 1: 100, 101, 103, 107
    # Size 2: 201, 203, 207, 204, 210, 214
    # Size 3: 304, 308, 311, 411 (100+101+103=304, 100+101+107=308, 100+103+107=310, 101+103+107=311)
    # Wait.
    # 100+103+107 = 310.
    # 101+103+107 = 311.
    # Size 4: 411.
    # Sums: 100, 101, 103, 107, 201, 203, 204, 207, 210, 214, 304, 308, 310, 311, 411.
    # Check 100 divides 200, 300, 400... 201 no, 203 no, 204 no, 207 no, 210 no, 214 no, 304 no, 308 no, 310 no, 311 no, 411 no.
    # Check 101 divides 202, 303, 404... 201 no, 203 no, 204 no, 207 no, 210 no, 214 no, 304 no, 308 no, 310 no, 311 no, 411 no.
    # Check 103 divides 206, 309, 412... 201 no, 203 no, 204 no, 207 no, 210 no, 214 no, 304 no, 308 no, 310 no, 311 no, 411 no.
    # Check 107 divides 214. 214 is in the list! (107+107? No, 107+107 not allowed. 107+101=208? No. 107+100=207. 107+103=210. 107+101+103=311. 107+100+103=310. 107+100+101=308. 107+100+101+103=411.
    # Where did 214 come from? 107+107? No.
    # Ah, 107+101=208. 107+103=210. 107+100=207.
    # Did I make a mistake in size 2 sums?
    # 100+101=201
    # 100+103=203
    # 100+107=207
    # 101+103=204
    # 101+107=208
    # 103+107=210
    # So 214 is NOT a sum. 214 = 107+107. Not possible.
    # So 107 does not divide any sum?
    # Multiples of 107: 214, 321, 428.
    # 214 not in sums.
    # 321 not in sums.
    # 428 not in sums.
    # So [100, 101, 103, 107] is VALID. Size 4.
    
    # Can we add more?
    # Try 109?
    # New sums involving 109:
    # 109
    # 109+100=209, 109+101=210 (Dup with 103+107=210!). Invalid.
    
    # Try 113?
    # 113
    # 113+100=213
    # 113+101=214
    # 113+103=216
    # 113+107=220
    # 113+201=314
    # 113+203=316
    # 113+204=317
    # 113+207=320
    # 113+208=321
    # 113+210=323
    # 113+304=417
    # 113+308=421
    # 113+310=423
    # 113+311=424
    # 113+411=524
    # Check divisibility with old sums:
    # 100 div 200, 300, 400, 500... 213 no, 214 no, 216 no, 220 no, 314 no, 316 no, 317 no, 320 no, 321 no, 323 no, 417 no, 421 no, 423 no, 424 no, 524 no.
    # 101 div 202, 303, 404, 505... 213 no, 214 no, 216 no, 220 no, 314 no, 316 no, 317 no, 320 no, 321 no, 323 no, 417 no, 421 no, 423 no, 424 no, 524 no.
    # 103 div 206, 309, 412, 515... 213 no, 214 no, 216 no, 220 no, 314 no, 316 no, 317 no, 320 no, 321 no, 323 no, 417 no, 421 no, 423 no, 424 no, 524 no.
    # 107 div 214. 214 is in new sums! (113+101=214). Invalid.
    
    # It seems adding to this set is hard.
    # Let's try a different strategy: Randomized search for size 5.
    
    def is_valid_set(elements):
        if not elements:
            return True
        sums = set()
        sums.add(0)
        for e in elements:
            new_sums = set()
            for s in sums:
                ns = s + e
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        # Check divisibility
        sums_list = sorted(s for s in sums if s != 0)
        for i in range(len(sums_list)):
            a = sums_list[i]
            for j in range(i + 1, len(sums_list)):
                b = sums_list[j]
                if b % a == 0:
                    return False
        return True

    # Known valid set of size 4
    candidate = [100, 101, 103, 107]
    if is_valid_set(candidate):
        best_set = candidate
        best_size = 4
    
    # Try to extend or find better via random search
    # We will try to build sets of size 5, 6, ...
    # Use a greedy approach with randomization and backtracking.
    
    def try_build(target_size):
        # Start with a random base of size 2 or 3
        # Or start empty and add one by one
        current = []
        current_sums = {0}
        
        def add(x):
            nonlocal current, current_sums
            new_sums = set()
            for s in current_sums:
                ns = s + x
                if ns in current_sums:
                    return False
                new_sums.add(ns)
            
            # Check divisibility
            all_sums = list(current_sums - {0}) + list(new_sums)
            # This is inefficient. Let's do it smarter.
            # Check new_sums against current_sums
            for ns in new_sums:
                for cs in current_sums:
                    if cs == 0: continue
                    if cs > ns:
                        if cs % ns == 0: return False
                    else:
                        if ns % cs == 0: return False
            # Check new_sums against each other
            ns_list = sorted(new_sums)
            for i in range(len(ns_list)):
                for j in range(i+1, len(ns_list)):
                    if ns_list[j] % ns_list[i] == 0:
                        return False
            
            current.append(x)
            current_sums.update(new_sums)
            return True
        
        def remove(x):
            nonlocal current, current_sums
            current.remove(x)
            # Recalculate sums
            current_sums = {0}
            for e in current:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + e)
                current_sums.update(new_sums)
        
        # Randomized backtracking
        # This is a simple heuristic, might not be optimal but fast.
        # We'll just try random permutations of a subset of candidates.
        # Candidates: 50..256
        candidates = list(range(50, 257))
        random.shuffle(candidates)
        
        # Try to pick target_size elements
        # Greedy: pick first valid, then next, etc.
        # If stuck, backtrack? 
        # For simplicity, just do random restarts with greedy.
        
        for restart in range(1000):
            if time.time() > deadline:
                return best_set
            random.shuffle(candidates)
            current = []
            current_sums = {0}
            for x in candidates:
                if len(current) >= target_size:
                    break
                # Try to add x
                new_sums = set()
                valid = True
                for s in current_sums:
                    ns = s + x
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if not valid:
                    continue
                
                # Check divisibility
                for ns in new_sums:
                    for cs in current_sums:
                        if cs == 0: continue
                        if cs > ns:
                            if cs % ns == 0:
                                valid = False
                                break
                        else:
                            if ns % cs == 0:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    continue
                
                ns_list = sorted(new_sums)
                for i in range(len(ns_list)):
                    for j in range(i+1, len(ns_list)):
                        if ns_list[j] % ns_list[i] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                current.append(x)
                current_sums.update(new_sums)
            
            if len(current) > best_size:
                best_set = current[:]
                best_size = len(current)
                if best_size >= 6:
                    return best_set

    # Try to find size 5
    try_build(5)
    
    # Try to find size 6
    if best_size < 6:
        try_build(6)
        
    # Try to find size 7
    if best_size < 7:
        try_build(7)

    # Final validation
    if best_set and not is_valid_set(best_set):
        # Fallback to known valid set
        best_set = [100, 101, 103, 107]
        best_size = 4

    return list(best_set)

if __name__ == "__main__":
    print(construct())

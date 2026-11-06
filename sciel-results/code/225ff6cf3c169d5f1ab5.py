import time
import random
import bisect
from collections import defaultdict

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Precompute divisors for each number in [1, N]
    # divisors[i] contains all j in [1, N] such that i divides j
    divisors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divisors[i].append(j)
    
    # Precompute pairs (b, c) with b <= c and their sum
    # We want to know, for a given set S, if there exist a, b, c in S 
    # with a < b <= c and a | (b+c).
    # This is equivalent to: for each a in S, does there exist b, c in S 
    # with a < b <= c and b+c is a multiple of a?
    
    # To speed up checks, we can precompute for each a, the set of sums b+c
    # that are multiples of a.
    # Actually, it's easier to check on the fly with bitmasks or sets.
    
    # Let's represent the set S as a Python set for O(1) membership.
    # And a sorted list for iteration.
    
    def is_valid_addition(s_set, s_sorted, candidate):
        """
        Check if adding 'candidate' to 's_set'/'s_sorted' keeps validity.
        Returns True if valid.
        """
        # 1. candidate as 'a' (smallest)
        # Need a < b <= c in S U {a} such that a | (b+c).
        # Here a = candidate.
        # b, c must be in S U {a} and > a.
        # Since a is the smallest, b, c are from S elements > a, or a itself?
        # No, a < b, so b cannot be a.
        # So b, c must be in S and > a.
        # Condition: exists b, c in S such that a < b <= c and (b+c) % a == 0.
        
        # To optimize, we can group elements of S > a by their residue mod a.
        # If there are two elements b, c in S > a with b <= c and (b+c) % a == 0, invalid.
        # Note: b and c can be the same value? No, set has unique values.
        # But b <= c allows b=c? The task says "a<b<=c". Since S is a set, b and c are distinct elements?
        # "subset of [1,n]". Sets don't have duplicates. So b != c.
        # Wait, "Repeated larger summands b=c are included." in the task description.
        # "For every a<b<=c in the set".
        # If the set is a set of integers, elements are distinct.
        # Does "in the set" mean values from the set? Yes.
        # If b=c, then b must be in the set.
        # So if b=c, we need 2b to be divisible by a.
        
        # Let's re-read carefully: "construct() -> a nonempty subset of [1,n=256]".
        # "For every a<b<=c in the set".
        # This implies a, b, c are elements of the set.
        # Since it's a set, a != b != c.
        # But the condition is a < b <= c.
        # So a, b, c are distinct?
        # If a < b, then a != b.
        # If b <= c, and they are from a set, b can equal c only if the set allows duplicates?
        # No, a set does not allow duplicates.
        # So b < c strictly?
        # Or does "b<=c" imply we consider the case where the value b appears twice?
        # "Repeated larger summands b=c are included."
        # This suggests that if we pick b from the set, we can use it twice as b and c.
        # So the constraint is: For all a, b, c in S such that a < b and b <= c (where b,c are values, and if b=c, the value b is in S), a must NOT divide b+c.
        # Since S is a set, if b=c, then b is in S.
        # So we check:
        # 1. Distinct a, b, c with a < b < c in S: a | (b+c)?
        # 2. a, b in S with a < b: a | (2b)? (case b=c)
        
        # Let's implement this logic.
        
        # Case 1: candidate is 'a'
        # a = candidate
        # We need to check if there exist b, c in S such that a < b <= c and a | (b+c).
        # This splits into:
        # 1a. b, c distinct in S, a < b < c, a | (b+c)
        # 1b. b in S, a < b, a | (2b)
        
        # To check 1a and 1b efficiently:
        # For each b in S with b > a:
        #   Check 1b: if (2*b) % a == 0: return False
        #   Check 1a: Is there c in S such that c > b (or c >= b? if c=b it's 1b) and (b+c) % a == 0?
        #   Actually, if we iterate b, and check for c in S with c > b and (b+c) % a == 0.
        
        # Optimization:
        # Precompute residues of elements in S greater than a?
        # Since 'candidate' is being added, S is the current set.
        # We can just iterate through S.
        
        # Let's try a simpler check first, then optimize if too slow.
        
        # Check candidate as 'a':
        for b in s_set:
            if b <= candidate:
                continue
            # Check b=c case
            if (2 * b) % candidate == 0:
                return False
            # Check b<c case
            # We need c in s_set such that c > b and (b+c) % candidate == 0
            # Target residue: target = (-b) % candidate
            # We need c in s_set, c > b, c % candidate == target
            # To make this fast, we could maintain a count of residues for s_set.
            # But s_set changes.
            # For now, linear scan of s_set for c?
            # This makes is_valid_addition O(|S|^2) in worst case.
            # |S| is around 100-150. 150^2 = 22500 ops.
            # If we do this for 256 candidates, and repeat in local search... might be slow.
            
            # Let's use a helper to check if any c in s_set with c > b has residue target.
            # We can precompute a list of (value, residue) for s_set?
            # Or just iterate.
            
            target = (-b) % candidate
            found_c = False
            for c in s_set:
                if c > b and c % candidate == target:
                    found_c = True
                    break
            if found_c:
                return False

        # Case 2: candidate is 'b' or 'c'
        # Subcase 2a: candidate is 'b'
        # a < candidate <= c
        # a in S, a < candidate
        # c in S U {candidate} such that candidate <= c
        # If c = candidate: a | (2*candidate)
        # If c in S, c > candidate: a | (candidate + c)
        
        for a in s_set:
            if a >= candidate:
                continue
            # Check c = candidate
            if (2 * candidate) % a == 0:
                return False
            # Check c in S, c > candidate
            target = (-candidate) % a
            for c in s_set:
                if c > candidate and c % a == target:
                    return False
                    # break inner, but return False immediately

        # Subcase 2b: candidate is 'c'
        # a < b <= candidate
        # a, b in S, a < b
        # candidate is c.
        # Check if a | (b + candidate)
        
        # We need to check all pairs (a, b) in S with a < b.
        # This is O(|S|^2).
        # Is there a faster way?
        # For each a in S:
        #   For each b in S with a < b:
        #      if (b + candidate) % a == 0: return False
        
        # We can optimize by iterating a, and checking if there exists b in S 
        # such that a < b and b % a == (-candidate) % a.
        
        for a in s_set:
            target = (-candidate) % a
            # We need b in S such that a < b and b % a == target
            for b in s_set:
                if b > a and b % a == target:
                    return False

        return True

    def try_add(s_set, s_sorted, candidate):
        if is_valid_addition(s_set, s_sorted, candidate):
            new_set = s_set | {candidate}
            new_sorted = sorted(new_set)
            return True, new_set, new_sorted
        return False, s_set, s_sorted

    def greedy_with_order(order):
        s_set = set()
        s_sorted = []
        for candidate in order:
            ok, new_set, new_sorted = try_add(s_set, s_sorted, candidate)
            if ok:
                s_set = new_set
                s_sorted = new_sorted
        return s_set, s_sorted

    # Multiple greedy strategies
    best_set = set()
    best_sorted = []
    best_len = 0
    
    # Strategy 1: High to low
    s1_set, s1_sorted = greedy_with_order(range(N, 0, -1))
    if len(s1_set) > best_len:
        best_set = s1_set
        best_sorted = s1_sorted
        best_len = len(s1_set)
    
    # Strategy 2: Randomized
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        order = list(range(N, 0, -1))
        random.shuffle(order)
        # Bias toward larger numbers
        biased_order = sorted(order, key=lambda x: x * random.uniform(0.8, 1.2), reverse=True)
        s_set, s_sorted = greedy_with_order(biased_order)
        if len(s_set) > best_len:
            best_set = s_set
            best_sorted = s_sorted
            best_len = len(s_set)
            
    current_set = best_set
    current_sorted = best_sorted
    
    # Local search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # 1. Try single add
        missing = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(missing)
        for candidate in missing:
            ok, new_set, new_sorted = try_add(current_set, current_sorted, candidate)
            if ok:
                current_set = new_set
                current_sorted = new_sorted
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set
                    best_sorted = current_sorted
                    best_len = len(current_set)
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # 2. Try swap: remove one, try to add two
        if len(current_set) > 2:
            remove_val = random.choice(list(current_set))
            temp_set = current_set - {remove_val}
            temp_sorted = [x for x in current_sorted if x != remove_val]
            
            missing = [x for x in range(1, N + 1) if x not in temp_set]
            random.shuffle(missing)
            
            added = 0
            new_set = temp_set
            new_sorted = temp_sorted
            for candidate in missing:
                if added >= 2:
                    break
                ok, ns, nso = try_add(new_set, new_sorted, candidate)
                if ok:
                    new_set = ns
                    new_sorted = nso
                    added += 1
            
            if added >= 2:
                current_set = new_set
                current_sorted = new_sorted
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set
                    best_sorted = current_sorted
                    best_len = len(current_set)
        
        if not improved:
            # 3. Perturbation: remove k random, rebuild
            k = random.randint(3, 7)
            if len(current_set) <= k:
                k = max(1, len(current_set) // 3)
            
            remove_vals = random.sample(list(current_set), k)
            remove_set = set(remove_vals)
            perturbed_set = current_set - remove_set
            perturbed_sorted = [x for x in current_sorted if x not in remove_set]
            
            all_missing = [x for x in range(1, N + 1) if x not in perturbed_set]
            random.shuffle(all_missing)
            
            for candidate in all_missing:
                if time.time() - start_time > deadline:
                    break
                ok, ns, nso = try_add(perturbed_set, perturbed_sorted, candidate)
                if ok:
                    perturbed_set = ns
                    perturbed_sorted = nso
            
            if len(perturbed_set) > len(current_set):
                current_set = perturbed_set
                current_sorted = perturbed_sorted
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set
                    best_sorted = current_sorted
                    best_len = len(current_set)
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return list(best_set)

import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    limit = P['n']
    all_nums = list(range(2, limit + 1))
    n = len(all_nums)
    
    # Precompute for each x: the set of pairs (b, c) with b < c, b != x, c != x,
    # such that x | b*c.
    # We'll store this as a list of frozensets for each x.
    # But that's too much memory. Instead, for each x, we'll store a set of
    # "conflict indicators". 
    #
    # Better approach: For each x, precompute the set of numbers y such that
    # there exists z (z != x, z != y) with x | y*z.
    # No, that's not quite right either.
    #
    # Let me think about this differently. The constraint is:
    # For any three distinct a, b, c in S: a does not divide b*c.
    #
    # For a fixed set S, this is violated if there exist distinct a, b, c in S
    # such that a | b*c.
    #
    # For a greedy approach: when we want to add x to S, we need to check:
    # 1. For all pairs (b, c) in S: x does not divide b*c
    # 2. For all b in S: x does not divide b*c for any c in S (c != x, c != b)
    #    But wait, this is the same as 1 if we consider all pairs.
    # 3. For all pairs (b, c) in S: b does not divide x*c and c does not divide x*b
    #
    # So the check when adding x to S is:
    # For all distinct b, c in S:
    #   (b*c) % x != 0
    #   (x*c) % b != 0
    #   (x*b) % c != 0
    #
    # This is O(|S|^2) per addition.
    #
    # To speed this up, we can precompute for each x a "signature" that helps
    # with the divisibility checks. But it's hard to avoid the O(|S|^2) check.
    #
    # Alternative: use local search / simulated annealing.
    # Start with a random set, then try to swap elements or add/remove to improve.
    #
    # Let me try a different approach: use the fact that if x is prime and x > limit/2,
    # then x cannot divide b*c for any b, c <= limit (since b*c < x^2 <= limit^2,
    # but more precisely, if x > limit/2 and x is prime, then x | b*c implies
    # x | b or x | c, but b, c < x so this is impossible).
    # Wait, that's not right. If x > limit/2 and x is prime, and b, c are in [2, limit],
    # then x | b*c implies x | b or x | c (by Euclid's lemma). Since b, c <= limit < 2x,
    # x | b implies b = x (since b >= 2 and b < 2x, so b = x). But b != x (distinct),
    # so this is impossible. Similarly for c.
    # So all primes in (limit/2, limit] can always be included!
    
    # Let me verify: primes in (128, 256]
    # These are: 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251
    # That's 23 primes.
    
    # What about primes in (limit/3, limit/2]? Say x in (85, 128].
    # x | b*c. If x is prime, x | b or x | c.
    # b can be x or 2x (if 2x <= limit). For x > 85, 2x > 170, so 2x <= 256 means x <= 128.
    # So for x in (85, 128], 2x is in (170, 256], which is in our range.
    # So if we include x and 2x, and some c, then x | (2x)*c. But we need three DISTINCT elements.
    # So if we have x, 2x, and c (c != x, c != 2x), then x | 2x*c. This violates the constraint.
    # So we cannot have both x and 2x in S (if there's a third element).
    # But if S = {x, 2x} only, that's fine (no three distinct elements).
    # But we want to maximize |S|, so we'll have many elements.
    
    # So the "free" elements are primes > 128. Let me count them.
    
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        for i in range(3, int(num**0.5) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    free_primes = [x for x in range(129, limit + 1) if is_prime(x)]
    # These can always be included.
    
    # For the remaining numbers (2 to 128, plus primes in 2..128), we need to be careful.
    # Let me use a greedy + local search approach.
    
    # Precompute: for each x in [2, 128], and for each pair (b, c) in [2, 256],
    # whether x | b*c. But that's too much.
    
    # Let me just use the direct check with optimization.
    
    def can_add(x, S):
        """Check if x can be added to set S."""
        S_list = list(S)
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i + 1, len(S_list)):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True
    
    # Start with all free primes
    base_set = set(free_primes)
    
    # Now try to add other numbers greedily
    # Order: try numbers in various orders
    
    random.seed(42)
    
    best_set = base_set.copy()
    best_size = len(base_set)
    
    # Try multiple random orderings
    remaining = [x for x in all_nums if x not in base_set]
    
    num_restarts = 300
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(remaining)
        S = base_set.copy()
        
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search: try to swap elements
    if time.time() < deadline - 3:
        S = list(best_set)
        S_set = set(S)
        not_in_S = [x for x in all_nums if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            random.shuffle(not_in_S)
            for new_x in not_in_S:
                if time.time() > deadline - 2:
                    break
                for i, old_x in enumerate(S):
                    if time.time() > deadline - 2:
                        break
                    # Try swapping old_x with new_x
                    S_test = S.copy()
                    S_test.remove(old_x)
                    S_test.add(new_x)
                    S_test_list = list(S_test)
                    valid = True
                    for a in S_test_list:
                        for b_idx in range(len(S_test_list)):
                            b = S_test_list[b_idx]
                            if b == a:
                                continue
                            for c_idx in range(b_idx + 1, len(S_test_list)):
                                c = S_test_list[c_idx]
                                if c == a or c == b:
                                    continue
                                if (b * c) % a == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        S = S_test
                        S_set = set(S)
                        not_in_S = [x for x in all_nums if x not in S_set]
                        improved = True
                        break
                if improved:
                    break
    
    result = sorted(best_set)
    
    # Final validation
    valid = True
    for i in range(len(result)):
        a = result[i]
        for j in range(i + 1, len(result)):
            b = result[j]
            for k in range(j + 1, len(result)):
                c = result[k]
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: just the free primes
        return sorted(free_primes)
    
    return result

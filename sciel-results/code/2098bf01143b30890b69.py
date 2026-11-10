"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 82  # 82s safety margin (90s kill limit)
    N = 512
    
    # Precompute pair conflicts: pair_conflict[x] is a bitset where bit y is set if
    # there exists some z in [1,512] (z != x, z != y) such that x is the mean of {y, z}
    # i.e., 2*x = y + z, so z = 2*x - y.
    # If we have both x and y in our set, and 2*x-y is also in our set (and distinct from x,y),
    # then x is the mean of {y, 2*x-y}.
    # More generally, for any element a in our set, a is bad if there exist b,c in set\{a}
    # with a = (b+c)/k for some k>=2.
    # For k=2: a = (b+c)/2, i.e., 2a = b+c.
    
    # Precompute for each element x, the set of pairs (y,z) such that 2*x = y+z
    # This is the "pair-mean" conflict.
    
    # Actually, let's think about this differently.
    # The constraint is: no element a can be the arithmetic mean of any subset of >= 2 OTHER elements.
    # For subset size 2: a = (b+c)/2, i.e., 2a = b+c, b != a, c != a, b != c.
    # For subset size 3: a = (b+c+d)/3, i.e., 3a = b+c+d.
    # etc.
    
    # The pair check (k=2) is the most restrictive and most common.
    # Let's precompute: for each x, which y values would create a pair-mean conflict?
    # If x is in the set, and y is in the set, then 2*x - y must NOT be in the set (unless 2*x-y == x or 2*x-y == y, which means y=x or y=x, impossible since distinct).
    # Wait: 2*x = y + z where z = 2*x - y. We need z != x and z != y.
    # z != x means 2*x - y != x means x != y (true since distinct).
    # z != y means 2*x - y != y means x != y (true).
    # So if x, y, and 2*x-y are all in the set and all distinct, we have a violation.
    
    # Precompute pair_conflict[x]: bitset of y values where having x and y together requires 2*x-y NOT to be in set.
    # Actually simpler: conflict[x][y] = True means if x and y are both in set, then 2*x-y must not be.
    # But that's a 3-way constraint. Let me use a different approach.
    
    # For the k=2 case: if a, b, c are all in set and 2a = b+c (with a,b,c distinct), violation.
    # So for a given set S, a is "bad" if there exist b,c in S\{a} with b+c = 2a.
    # This is equivalent to: 2a is in the set of all pair sums of S\{a}.
    
    # Let me precompute for each a, the list of (b,c) pairs with b+c=2a, b!=a, c!=a, b<c.
    # Then checking if a is bad = checking if any (b,c) pair has both b and c in S\{a}.
    
    # For efficiency, I'll use bitsets.
    # Let me index elements 1..512 as bits 0..511.
    
    # Precompute: for each a in 1..512, a bitset pair_mask[a] where bit j is set if element (j+1)
    # forms a pair with some other element to average to a.
    # Actually, let me precompute: pair_partners[a] = list of (b,c) pairs where b+c=2a, b<a, c<a or b<a, c>a, etc.
    # Simpler: for each a, precompute the set of b values (b != a) such that 2a-b is a valid element (1..512, != a, != b).
    # Then a is bad iff there exist b,c in S\{a} with b+c=2a.
    # Using bitset: if I have a bitset S of elements in the set, I can check if there's a pair summing to 2a.
    
    # Even better: precompute for each a, a bitset "partner_mask[a]" where bit i is set if element (i+1) is a valid partner.
    # Then a is bad iff (S & partner_mask[a]) has at least 2 bits... no, that's not right either.
    # a is bad iff there exist b,c in S\{a} with b+c=2a.
    # Let T = S without a. Then I need to check if 2a is a pair sum of T.
    # Using bitset convolution or just iterating.
    
    # For speed, let me precompute for each target sum s (2..1024), the list of pairs (b,c) with b+c=s, b<c, 1<=b,c<=512.
    # Then for checking if a is bad: s=2a, check if any pair in pairs[s] has both elements in S\{a}.
    
    # Precompute all pair decompositions
    pair_decomp = {}  # s -> list of (b,c) with b+c=s, 1<=b<c<=512
    for s in range(4, 1025):  # min sum is 1+3=4 (since b<c and b>=1, c>=2, but actually b>=1,c>=b+1)
        pairs = []
        for b in range(1, min(s-1, 512) + 1):
            c = s - b
            if c > b and c <= 512:
                pairs.append((b, c))
        if pairs:
            pair_decomp[s] = pairs
    
    # Precompute for each a, the relevant pairs (those where a is not b or c)
    # a is bad iff there exists (b,c) in pair_decomp[2a] with b!=a and c!=a, and b,c in S.
    # Since b+c=2a and b!=a, c!=a automatically if b!=c (which is guaranteed).
    # Wait: if b=a, then c=2a-a=a, so b=c=a, but we require b<c, so this can't happen.
    # So all pairs in pair_decomp[2a] have b!=a and c!=a automatically!
    
    # Great. So for each a, I just need to check if any pair in pair_decomp[2a] has both elements in S.
    
    # For k=3: a is bad iff there exist b,c,d in S\{a} with b+c+d=3a.
    # This is harder to precompute efficiently. Let me handle k=2 first and see how far we get,
    # then add k=3,4,... checks incrementally.
    
    # Let me build a fast check function.
    
    def can_add_fast(element, S_set):
        """Check if element can be added to S_set (a set of ints) without violating constraint.
        Returns True if safe to add."""
        # Check 1: is element the mean of any subset of S_set?
        # For k=2: check if 2*element is a pair sum of S_set
        s2 = 2 * element
        if s2 in pair_decomp:
            for b, c in pair_decomp[s2]:
                if b in S_set and c in S_set:
                    return False
        
        # For k=3: check if 3*element is a triple sum of S_set
        s3 = 3 * element
        if s3 <= 3 * 512:
            # Check if there exist b,c,d in S_set with b+c+d=s3, all != element
            # Use a bitset approach or iterate
            elems = list(S_set)
            n = len(elems)
            if n >= 3:
                # For each pair (b,c), check if s3-b-c is in S_set and != element, != b, != c
                # This is O(n^2) which might be slow for large n
                # Use a set for O(1) lookup
                for i in range(n):
                    b = elems[i]
                    if b == element:
                        continue
                    for j in range(i+1, n):
                        c = elems[j]
                        if c == element:
                            continue
                        d = s3 - b - c
                        if d != element and d != b and d != c and 1 <= d <= 512 and d in S_set:
                            return False
        
        # For k=4: check if 4*element is a quadruple sum of S_set
        s4 = 4 * element
        if s4 <= 4 * 512:
            elems = list(S_set)
            n = len(elems)
            if n >= 4:
                # Check if there exist b,c,d,e in S_set with b+c+d+e=s4, all != element
                # O(n^3) is too slow. Use meet-in-the-middle: split into pairs.
                # Pair sums of S_set (excluding element)
                pair_sums = {}
                for i in range(n):
                    b = elems[i]
                    if b == element:
                        continue
                    for j in range(i+1, n):
                        c = elems[j]
                        if c == element:
                            continue
                        ps = b + c
                        if ps not in pair_sums:
                            pair_sums[ps] = []
                        pair_sums[ps].append((b, c))
                
                for ps, pairs in pair_sums.items():
                    complement = s4 - ps
                    if complement in pair_sums:
                        for b, c in pairs:
                            for d, e in pair_sums[complement]:
                                # Check all distinct and != element
                                if len({b, c, d, e}) == 4 and element not in {b, c, d, e}:
                                    return False
        
        # Check 2: for each a in S_set, does adding element create a new mean violation for a?
        # a is bad if there exist b,c in (S_set ∪ {element}) \ {a} with b+c=2a
        # Previously a was fine, so the new violation must involve element.
        # Case 2a: element is one of b,c. So element + c = 2a for some c in S_set\{a}, c != element.
        #   i.e., c = 2a - element. Check if 2a-element is in S_set and != a and != element.
        # Case 2b: element is the mean itself. Already checked in Check 1.
        # Case 2c: for k=3: element + c + d = 3a for some c,d in S_set\{a}.
        #   i.e., c+d = 3a - element. Check if 3a-element is a pair sum of S_set\{a,element}.
        
        for a in S_set:
            # k=2: check if element + c = 2a for some c in S_set\{a}
            c = 2 * a - element
            if c != a and c != element and 1 <= c <= 512 and c in S_set:
                return False
            
            # k=3: check if element + c + d = 3a for some c,d in S_set\{a}
            s3a = 3 * a - element
            if s3a >= 2:  # min sum of two distinct positive ints is 1+2=3
                # Check if s3a is a pair sum of S_set without a and without element
                if s3a in pair_decomp:
                    for b, c in pair_decomp[s3a]:
                        if b == a or c == a or b == element or c == element:
                            continue
                        if b in S_set and c in S_set:
                            return False
            
            # k=4: check if element + c + d + e = 4a for some c,d,e in S_set\{a}
            s4a = 4 * a - element
            if s4a >= 6:  # min sum of three distinct positive ints is 1+2+3=6
                # Check if s4a is a triple sum of S_set without a and without element
                # This is expensive. Skip for now or use a quick check.
                # For now, skip k=4 in the incremental check and rely on final verification.
                pass
        
        return True
    
    # Final verification of a complete set
    def verify_set(S):
        """Verify that no element in S is the mean of any subset of >= 2 other elements."""
        S_set = set(S)
        n = len(S)
        for a in S:
            others = [x for x in S if x != a]
            n_others = len(others)
            if n_others < 2:
                continue
            for k in range(2, n_others + 1):
                target = k * a
                # Check if target is a sum of exactly k elements from others
                # Use DP with bitsets
                # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
                max_sum = target
                dp = [0] * (k + 1)
                dp[0] = 1  # sum 0 achievable with 0 elements
                for val in others:
                    if val > max_sum:
                        break
                    for j in range(k, 0, -1):
                        if dp[j-1]:
                            shifted = (dp[j-1] << val) & ((1 << (max_sum + 1)) - 1)
                            dp[j] |= shifted
                if (dp[k] >> target) & 1:
                    return False
        return True
    
    # Greedy construction with fast check
    def greedy_build(ordering, S_set=None, S_list=None):
        if S_list is None:
            S_list = []
        if S_set is None:
            S_set = set()
        
        for x in ordering:
            if time.time() > deadline - 3:
                break
            if x in S_set:
                continue
            if can_add_fast(x, S_set):
                S_list.append(x)
                S_set.add(x)
        
        return S_list, S_set
    
    # Multiple greedy strategies
    best_list = []
    best_set = set()
    
    # Ordering 1: ascending
    l, s = greedy_build(list(range(1, 513)))
    if len(l) > len(best_list):
        best_list, best_set = l, s
    
    # Ordering 2: descending
    l, s = greedy_build(list(range(512, 0, -1)))
    if len(l) > len(best_list):
        best_list, best_set = l, s
    
    # Ordering 3: alternating small/large
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)
    l, s = greedy_build(alt)
    if len(l) > len(best_list):
        best_list, best_set = l, s
    
    # Ordering 4: primes first
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    l, s = greedy_build(primes + non_primes)
    if len(l) > len(best_list):
        best_list, best_set = l, s
    
    # Ordering 5: random
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        random.seed(42 + trial)
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        l, s = greedy_build(rand_order)
        if len(l) > len(best_list):
            best_list, best_set = l, s
    
    # Local improvement: try adding more elements
    if time.time() < deadline - 3:
        improved = best_list[:]
        improved_set = best_set.copy()
        remaining = [x for x in range(1, 513) if x not in improved_set]
        for x in remaining:
            if time.time() > deadline - 3:
                break
            if can_add_fast(x, improved_set):
                improved.append(x)
                improved_set.add(x)
        if len(improved) > len(best_list):
            best_list, best_set = improved, improved_set
    
    # Swap-based local search: remove one element, try to add others
    if time.time() < deadline - 3:
        improved = best_list[:]
        improved_set = best_set.copy()
        improved_size = len(improved)
        
        for i in range(len(improved)):
            if time.time() > deadline - 3:
                break
            
            # Remove element i
            removed = improved[i]
            new_list = improved[:i] + improved[i+1:]
            new_set = set(new_list)
            
            # Try to add elements
            remaining = [x for x in range(1, 513) if x not in new_set]
            for x in remaining:
                if time.time() > deadline - 3:
                    break
                if can_add_fast(x, new_set):
                    new_list.append(x)
                    new_set.add(x)
            
            if len(new_list) > improved_size:
                improved = new_list
                improved_set = new_set
                improved_size = len(improved)
                break  # restart the swap loop with the improved set
        
        if len(improved) > len(best_list):
            best_list, best_set = improved, improved_set
    
    # Verify the best set
    if best_list and not verify_set(best_list):
        # Fallback: rebuild with full verification
        best_list = []
        best_set = set()
        for x in range(1, 513):
            if time.time() > deadline - 2:
                break
            if x in best_set:
                continue
            # Full check
            if can_add_fast(x, best_set):
                # Double-check with full verification
                test_list = best_list + [x]
                if verify_set(test_list):
                    best_list.append(x)
                    best_set.add(x)
    
    return best_list
# EVOLVE-BLOCK-END

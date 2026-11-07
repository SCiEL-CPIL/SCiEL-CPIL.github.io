import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    def get_subset_sums(S):
        """Get all non-empty subset sums of S using iterative DP."""
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums |= new_sums
        return sums

    def is_valid_fast(S):
        """Check if all non-empty subset sums are divisibility-incomparable and unique."""
        sums = get_subset_sums(S)
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    def can_add_fast(current_sums, x):
        """Check if adding x to a set (with current_sums) keeps divisibility-incomparability and uniqueness."""
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        # Check uniqueness: no new sum equals an existing sum
        if new_sums & current_sums:
            return False
        all_sums = current_sums | new_sums
        sums_list = sorted(all_sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    # Focus on large numbers [128, 256] where divisibility conflicts are rare
    # Key insight: for elements in [128,256], subset sums of k elements are in [128k, 256k]
    # Divisibility s1|s2 requires s2 >= 2*s1, which is harder to satisfy for large elements
    
    large_candidates = list(range(128, 257))
    
    # Precompute pairwise conflict: can elements a and b coexist?
    # Two elements a,b are compatible if no subset sum divides another
    # For 2 elements: sums are {a, b, a+b}. Need a∤b, a∤(a+b), b∤(a+b)
    # a | (a+b) iff a | b, so just need a∤b and b∤a (i.e., a≠b and neither divides the other)
    # Actually for distinct a,b in [128,256]: a∤b since b<2a (b≤256, a≥128, so b/a ≤ 2)
    # b∤a similarly. And a∤(a+b) iff a∤b. So any two distinct elements in [128,256] are pairwise compatible!
    
    # The constraint is on 3+ elements. Let's check triple compatibility.
    # For {a,b,c} with a<b<c all in [128,256]:
    # Sums: a, b, c, a+b, a+c, b+c, a+b+c
    # Need no sum divides another.
    # Since all single elements are in [128,256] and all 2-sums are in [257,512],
    # and 3-sum is in [387,768]:
    # - Single x divides 2-sum y+z only if y+z=2x (since y+z<3x for x≥128)
    # - Single x divides 3-sum y+z+w only if y+z+w=2x or 3x
    #   3x ≥ 384, and y+z+w ≥ 3*129=387, so possible
    # - 2-sum divides 3-sum: (a+b) | (a+b+c) iff (a+b)|c, but c < 2(a+b) since c≤256 and a+b≥257, so (a+b)|c impossible unless c=0
    #   Wait, a+b ≥ 257 > 256 ≥ c, so a+b > c, so (a+b) cannot divide c.
    #   So 2-sums never divide 3-sums.
    # - 2-sum divides another 2-sum: (a+b)|(a+c) iff (a+b)|(c-b). Since |c-b| < 256 < a+b (for a+b≥257), impossible.
    #   Similarly for other pairs of 2-sums.
    
    # So the only constraints are:
    # 1. No element divides a 2-sum: x | (y+z) for distinct x,y,z means y+z = 2x
    # 2. No element divides a 3-sum: x | (y+z+w) means y+z+w = 2x or 3x
    # 3. Uniqueness: all subset sums distinct (guaranteed if all elements are distinct and positive? No, need to verify)
    
    # Let's use a smarter approach: build the set incrementally, checking only the new constraints
    
    def build_set_greedy(candidates, deadline):
        """Greedy build with smart ordering."""
        S = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add_fast(current_sums, x):
                S.append(x)
                new_sums = {x}
                for s in current_sums:
                    new_sums.add(x + s)
                current_sums = current_sums | new_sums
        
        return S, current_sums

    def local_improve(S, candidates_pool, deadline):
        """Try to improve by swapping elements."""
        best_S = list(S)
        best_sums = get_subset_sums(best_S)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_S)):
                if time.time() > deadline:
                    break
                # Try removing best_S[i] and adding a different element
                temp_S = best_S[:i] + best_S[i+1:]
                temp_sums = get_subset_sums(temp_S)
                
                for x in candidates_pool:
                    if x in best_S:
                        continue
                    if can_add_fast(temp_sums, x):
                        new_S = temp_S + [x]
                        # Check if this is actually better (it's same size, but might enable further growth)
                        # For now, just accept any swap and try to extend
                        new_sums = get_subset_sums(new_S)
                        # Try to extend
                        extend_S = list(new_S)
                        extend_sums = new_sums
                        for y in candidates_pool:
                            if y in extend_S:
                                continue
                            if can_add_fast(extend_sums, y):
                                extend_S.append(y)
                                new_ext = {y}
                                for s in extend_sums:
                                    new_ext.add(s + y)
                                extend_sums = extend_sums | new_ext
                        
                        if len(extend_S) > len(best_S):
                            best_S = extend_S
                            best_sums = extend_sums
                            improved = True
                            break
                if improved:
                    break
        
        return best_S

    # Strategy 1: Greedy from different orderings
    best_S = []
    best_size = 0
    
    # Order 1: Descending (largest first)
    S1, _ = build_set_greedy(sorted(large_candidates, reverse=True), deadline - 5.0)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Order 2: Ascending
    S2, _ = build_set_greedy(sorted(large_candidates), deadline - 5.0)
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Order 3: Randomized multiple times
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        shuffled = large_candidates[:]
        random.shuffle(shuffled)
        S_trial, _ = build_set_greedy(shuffled, deadline - 10.0)
        if len(S_trial) > best_size:
            best_S = S_trial
            best_size = len(S_trial)
    
    # Local improvement
    if time.time() < deadline - 5.0:
        improved = local_improve(best_S, large_candidates, deadline - 3.0)
        if len(improved) > best_size:
            best_S = improved
            best_size = len(best_S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 2.0:
        current_sums = get_subset_sums(best_S)
        for x in large_candidates:
            if x in best_S:
                continue
            if can_add_fast(current_sums, x):
                best_S.append(x)
                new_sums = {x}
                for s in current_sums:
                    new_sums.add(s + x)
                current_sums = current_sums | new_sums
    
    # Verify and return
    if is_valid_fast(best_S):
        return [int(x) for x in sorted(best_S)]
    
    # Fallback: return a known valid small set
    fallback = [128, 129, 131]
    if is_valid_fast(fallback):
        return fallback
    
    return [128]

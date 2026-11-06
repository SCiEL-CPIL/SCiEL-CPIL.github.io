"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def _full_check_violation(S):
        """
        Returns True if S is valid (no element is mean of subset of others).
        Uses bitset DP for exact subset sum checks for all k >= 2.
        """
        n = len(S)
        if n < 3:
            return True
        
        for i, a in enumerate(S):
            T = S[:i] + S[i+1:]
            n_t = len(T)
            
            # DP for subset sums of T
            # dp[k] is bitset of sums achievable with k elements.
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            # Check if any k*a is in dp[k] for k >= 2
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * a
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def _remove_until_valid(S):
        """
        Iteratively removes violating elements until S is valid.
        Returns the valid set.
        """
        S = list(S)
        while True:
            if not _full_check_violation(S):
                # Find a violating element to remove
                # We can just pick the first one found or a random one
                # To be efficient, we can just remove the first one that causes violation
                # But _full_check_violation doesn't return which one.
                # Let's modify _full_check_violation to return the violating element.
                
                # For now, let's just try removing each element and checking?
                # That's O(N^2 * check). N=50. 2500 * 50 * bitops. Might be slow.
                
                # Better: Modify check to find one violation.
                viol_elem = _find_one_violation(S)
                if viol_elem is not None:
                    S.remove(viol_elem)
                else:
                    break
            else:
                break
        return S

    def _find_one_violation(S):
        n = len(S)
        for i, a in enumerate(S):
            T = S[:i] + S[i+1:]
            n_t = len(T)
            dp = [0] * (n_t + 1)
            dp[0] = 1
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * a
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    return a
        return None

    def _fast_conflict_check(S, x, max_k=4):
        """
        Checks if adding x to S creates a violation for k up to max_k.
        1. Is x the mean of a subset of S of size k (2 <= k <= max_k)?
        2. Is any a in S the mean of a subset of S U {x} of size k (2 <= k <= max_k) involving x?
        
        Returns True if NO conflict (safe to add), False if conflict.
        """
        # Check 1: x is mean of subset of S
        # Sum(T) = k * x, T subset of S, |T|=k
        # We can use a simple DP or just check small k.
        # For k=2: 2x = u+v.
        # For k=3: 3x = u+v+w.
        
        # Let's use a helper to check if target sum is achievable with k elements from S.
        def can_sum_k_elements(S, target, k):
            if k == 0:
                return target == 0
            if k > len(S):
                return False
            # Simple DP for small k
            # dp[j][s] -> too big.
            # Use set of sums for each j?
            sums = {0}
            for _ in range(k):
                new_sums = set()
                for s in sums:
                    for y in S:
                        new_sums.add(s + y)
                sums = new_sums
            return target in sums

        # This is O(N^k). For N=50, k=4, 50^4 = 6M. Too slow for 512 candidates * 50 checks.
        # We need a faster check.
        
        # Let's stick to k=2 and k=3 for the fast check, and rely on cleanup for higher k.
        # Or optimize k=2,3.
        
        # k=2 check for x:
        # 2x = u+v.
        # u = 2x - v.
        # Check if 2x - v in S for any v in S.
        # O(N).
        
        # k=3 check for x:
        # 3x = u+v+w.
        # u+v = 3x - w.
        # Check if 3x - w is sum of 2 elements in S.
        # O(N^2).
        
        # Check 2: a in S is mean of subset involving x.
        # Subset size k. Contains x.
        # Sum(T) + x = k * a.
        # Sum(T) = k * a - x.
        # T is subset of S \ {a} of size k-1.
        
        # k=2: T size 1. u = 2a - x. Check if u in S \ {a}.
        # k=3: T size 2. u+v = 3a - x. Check if 3a - x is sum of 2 elements in S \ {a}.
        
        # Let's implement these specific checks.
        
        S_set = set(S)
        
        # 1. x is mean of k=2 in S
        for v in S:
            if 2*x - v in S_set and 2*x - v != v: # distinct elements?
                # The problem says "subset of at least two OTHER elements".
                # Subset elements must be distinct from a, but can they be equal to each other?
                # "distinct positive integers" in the set S. So elements in S are distinct.
                # So u != v is guaranteed if u,v in S and u != v.
                # But we need 2 distinct elements from S.
                # If 2x - v == v, then 2x = 2v -> x=v. But x is not in S yet (we are checking add).
                # So u != v is always true if x not in S.
                # Wait, if x is not in S, then u=2x-v. If u=v, then x=v, contradiction.
                # So just check if 2x - v in S_set.
                return False
        
        # 2. x is mean of k=3 in S
        # 3x = u+v+w
        # Iterate w in S. Need u+v = 3x - w.
        # Check if 3x - w is sum of 2 distinct elements in S \ {w}.
        for w in S:
            target = 3*x - w
            # Check if target is sum of 2 elements in S, distinct from w.
            # u+v = target.
            # u = target - v.
            # Iterate v in S.
            # If v == w, skip? No, u and v are distinct from each other and from w?
            # The subset is {u,v,w}. They must be distinct elements of S.
            # So u != v, u != w, v != w.
            found = False
            for v in S:
                if v == w: continue
                u = target - v
                if u == v: continue
                if u == w: continue
                if u in S_set:
                    found = True
                    break
            if found:
                return False
        
        # 3. a in S is mean of k=2 involving x
        # a = (u+x)/2 => u = 2a - x
        for a in S:
            u = 2*a - x
            if u in S_set and u != a:
                return False
        
        # 4. a in S is mean of k=3 involving x
        # a = (u+v+x)/3 => u+v = 3a - x
        for a in S:
            target = 3*a - x
            # Check if target is sum of 2 distinct elements in S \ {a}.
            found = False
            for v in S:
                if v == a: continue
                u = target - v
                if u == v: continue
                if u == a: continue
                if u in S_set:
                    found = True
                    break
            if found:
                return False
        
        return True

    best_set = []
    
    # Parameters
    num_attempts = 100
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate candidate order
        # Bias towards edges?
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if len(S) >= 48:
                break
            # Fast check
            if _fast_conflict_check(S, x):
                S.append(x)
        
        # Cleanup with full check
        if S:
            S_clean = _remove_until_valid(S)
            if len(S_clean) > len(best_set):
                best_set = S_clean
                if len(best_set) == 48:
                    break
                    
    # Fallback
    if len(best_set) < 40:
        # Simple greedy descending
        S = []
        for x in range(512, 0, -1):
            if time.time() > deadline:
                break
            if _fast_conflict_check(S, x):
                S.append(x)
            if len(S) == 48:
                break
        S_clean = _remove_until_valid(S)
        if len(S_clean) > len(best_set):
            best_set = S_clean

    return best_set

# EVOLVE-BLOCK-END

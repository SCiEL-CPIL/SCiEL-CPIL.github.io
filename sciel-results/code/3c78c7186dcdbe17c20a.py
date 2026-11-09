import time
import random

def construct():
    deadline = time.time() + 85.0  # Safety margin from 90s limit
    
    def is_valid_addition(S, x):
        """
        Check if x can be added to valid set S.
        S is assumed to be valid.
        """
        n = len(S)
        if n < 2:
            return True
        
        # Check 1: Is x the mean of any subset of S of size k >= 2?
        # DP on S
        # dp[k] is bitset of sums using k elements from S
        max_sum_S = sum(S)
        dp_S = [0] * (n + 1)
        dp_S[0] = 1
        for y in S:
            for k in range(n - 1, -1, -1):
                if dp_S[k]:
                    dp_S[k+1] |= (dp_S[k] << y)
        
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        # Check 2: Is any a in S the mean of a subset of (S \ {a}) U {x} of size >= 2?
        # As derived, this implies the subset must contain x.
        # Subset T = {x} U U, U subset of S \ {a}, |U| >= 1.
        # Sum(T) = x + Sum(U) = k * a for some k >= 2.
        # Sum(U) = k * a - x.
        # We need to check if k * a - x is a sum of a subset U of S \ {a} with size >= 1.
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m == 0:
                continue
            
            # DP on others
            # dp_others[j] is bitset of sums using j elements from others
            max_sum_others = sum(others)
            dp_others = [0] * (m + 1)
            dp_others[0] = 1
            for y in others:
                for j in range(m - 1, -1, -1):
                    if dp_others[j]:
                        dp_others[j+1] |= (dp_others[j] << y)
            
            # We need to check if there exists j >= 1 and k >= 2 such that:
            # k * a - x is a sum of a subset of size j from others.
            # Let val = k * a - x.
            # val must be in [min_sum_j, max_sum_j] for some j.
            # Also val > 0.
            
            # Iterate k? k can be up to (max_sum_others + x) / a.
            # But k is also bounded by m + 1 (since |T| = j+1 <= m+1, and k=|T|? No, k is the count in the mean definition? 
            # Wait, the definition is "arithmetic mean of any subset".
            # Mean = Sum(T) / |T|.
            # So a = Sum(T) / |T| => Sum(T) = a * |T|.
            # Here |T| = j + 1. Let K = j + 1.
            # So we need Sum(T) = K * a.
            # Sum(T) = x + Sum(U).
            # x + Sum(U) = K * a => Sum(U) = K * a - x.
            # K ranges from 2 to m+1 (since |U| >= 1, |T| >= 2).
            
            for K in range(2, m + 2):
                target_sum_U = K * a - x
                if target_sum_U <= 0:
                    continue
                if target_sum_U > max_sum_others:
                    break # K is increasing, so target_sum_U increases
                
                # Check if target_sum_U is a sum of a subset of 'others' of size j = K - 1.
                # Note: j must be >= 1, which is true since K >= 2.
                # j must be <= m, which is true since K <= m+1.
                j = K - 1
                if (dp_others[j] >> target_sum_U) & 1:
                    return False
        
        return True

    best_set = []
    best_size = 0
    
    # Phase 1: Greedy with various orders
    orders = []
    base = list(range(1, 513))
    orders.append(base)
    orders.append(base[::-1])
    
    random.seed(42)
    for _ in range(20):
        random.shuffle(base)
        orders.append(base[:])
        
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in order:
            if len(current_set) >= 48:
                break
            if is_valid_addition(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        if best_size == 48:
            return best_set

    # Phase 2: Local Search / Swapping to improve size
    if best_size < 48 and time.time() < deadline - 5:
        S = list(best_set)
        S_set = set(S)
        
        unused = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add
            for u in unused:
                if len(S) >= 48:
                    break
                if is_valid_addition(S, u):
                    S.append(u)
                    S_set.add(u)
                    unused.remove(u)
                    improved = True
                    break # Restart add loop
            
            if len(S) >= 48:
                break
                
            # Try to swap
            random.shuffle(unused)
            for u in unused[:50]:
                if time.time() > deadline - 2:
                    break
                for i, s in enumerate(S):
                    # Swap s with u
                    # We need to check if S \ {s} U {u} is valid.
                    # This is not just is_valid_addition because removing s might break validity of others?
                    # No, removing an element never breaks validity.
                    # So S \ {s} is valid.
                    # We just need to check if u can be added to S \ {s}.
                    S_minus_s = S[:i] + S[i+1:]
                    if is_valid_addition(S_minus_s, u):
                        # Accept swap
                        S[i] = u
                        S_set.remove(s)
                        S_set.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break

        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set

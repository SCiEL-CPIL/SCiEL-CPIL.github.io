"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def _can_add(S, candidate):
        """
        Check if adding 'candidate' to set S would violate the constraint for 'candidate'.
        I.e., is 'candidate' the mean of any subset of S of size >= 2?
        Uses subset-sum DP with bitsets on S.
        Returns True if candidate can be added (no violation).
        """
        n = len(S)
        if n < 2:
            return True
        
        # DP: dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements from S
        # We only need to check up to k where k*candidate <= k*max(S), but we can just run all.
        # Actually, we need to check if k*candidate is in dp[k] for any k>=2.
        
        # Optimization: The max sum for k elements is k * max(S).
        # If k * candidate > k * max(S), i.e., candidate > max(S), it's impossible.
        # But candidate might be larger than some elements.
        
        # Let's use a dictionary of bitsets or a list of integers.
        # Since n can be up to ~48, and values up to 512, max sum is 48*512 = 24576.
        # Bitsets can be Python integers.
        
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in S:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        # Now check for each k >= 2 if k*candidate is in dp[k]
        # Max k is n.
        # We can break early if k*candidate > k*max(S) -> candidate > max(S)
        # But we don't know max(S) easily without computing it. Let's just iterate.
        
        # Actually, we can be smarter.
        # For a fixed k, target = k * candidate.
        # If target > k * max(S), then (dp[k] >> target) & 1 is 0.
        
        max_S = max(S) if S else 0
        for k in range(2, n + 1):
            target = k * candidate
            if target > k * max_S:
                # Since max_S is fixed, if candidate > max_S, this holds for all k.
                # But candidate might be <= max_S.
                # However, if target > k * max_S, then it's impossible.
                # And since k increases, target increases linearly.
                # Wait, if candidate <= max_S, then k*candidate <= k*max_S is always true.
                # So we only break if candidate > max_S.
                if candidate > max_S:
                    break
                # If candidate <= max_S, we must check all k.
            
            if (dp[k] >> target) & 1:
                return False
                
        return True

    def _is_valid_set(S):
        """
        Check if S is valid.
        Returns True if no element is the mean of any subset of other elements of size >= 2.
        """
        n = len(S)
        if n < 3:
            return True
        
        S_list = list(S)
        for i, a in enumerate(S_list):
            n_t = n - 1
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            max_val = max(S_list)
            for k in range(2, n_t + 1):
                target = k * a
                if target > k * max_val:
                    break
                if (dp[k] >> target) & 1:
                    return False
                    
        return True

    def _construct_greedy(target_size=48):
        """
        Construct a set of size target_size using greedy addition.
        """
        S = []
        # Start with a small random set to avoid trivial conflicts?
        # Or start empty.
        
        # To make it faster, we can start with a few random elements that are far apart.
        # Let's just start empty and add.
        
        # We want to maximize size. So we try to add as many as possible.
        # We should try candidates in an order that is likely to succeed.
        # Random order? Or sorted?
        # If we add small numbers first, large numbers might be means of small ones?
        # If we add large numbers first, small numbers might be means of large ones?
        
        # Let's try a random permutation of 1..512.
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if _can_add(S, c):
                S.append(c)
        
        return S

    def _local_search(S, max_iters=500):
        """
        Try to improve S by swapping elements or adding new ones.
        """
        S = list(S)
        current_size = len(S)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding any element from 1..512 not in S
            S_set = set(S)
            for c in range(1, 513):
                if c in S_set:
                    continue
                if _can_add(S, c):
                    S.append(c)
                    S_set.add(c)
                    improved = True
                    # Don't break, try to add more
                    if len(S) >= 48:
                        return S
            
            # If no additions, try swaps
            if not improved:
                for i, s_elem in enumerate(S):
                    if time.time() > deadline:
                        return S
                    for c in range(1, 513):
                        if c in S_set:
                            continue
                        # Check if S - {s_elem} + {c} is valid
                        # This requires checking:
                        # 1. c is not mean of subset of S\{s_elem}
                        # 2. s_elem is not mean of subset of S\{s_elem, c} + {c} ? No, s_elem is removed.
                        # 3. Other elements in S\{s_elem} are not means of subsets involving c?
                        #    Actually, if we remove s_elem and add c, we need to check:
                        #    - c is valid wrt S\{s_elem}
                        #    - All elements in S\{s_elem} are valid wrt (S\{s_elem})\{x} U {c}
                        #    This is expensive.
                        
                        # Simpler: just check if the new set is valid.
                        new_S = S[:i] + S[i+1:] + [c]
                        if _is_valid_set(new_S):
                            S = new_S
                            S_set.remove(s_elem)
                            S_set.add(c)
                            improved = True
                            break
                    if improved:
                        break
                        
        return S

    best_set = []
    best_size = 0
    
    # Phase 1: Multiple greedy restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        S = _construct_greedy(48)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        if best_size == 48:
            break
    
    # Phase 2: Local search on best set
    if best_size < 48 and best_set:
        if time.time() < deadline - 10:
            improved_set = _local_search(best_set, max_iters=200)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
    
    # Phase 3: If still short, try more restarts
    if best_size < 48:
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            S = _construct_greedy(48)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            if best_size == 48:
                break

    return best_set

# EVOLVE-BLOCK-END

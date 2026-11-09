import time
import random

def construct():
    deadline = time.time() + 58.0
    n_max = 512
    target_size = 48

    def is_valid_set(S):
        """
        Checks if S is a valid set using full DP reconstruction.
        Returns True if valid, False otherwise.
        """
        n = len(S)
        if n < 3:
            return True
        
        # 1. For each element a in S, check if a is the mean of a subset of S \ {a}
        for i, a in enumerate(S):
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            
            # DP for others
            # dp[k] is bitset of sums of k elements from others
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            
            # Check if a is mean of subset of size k >= 2
            # sum(T) = k * a
            for k in range(2, m + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def greedy_build_fast(order):
        """
        Greedy construction using maintained DP states for speed.
        """
        S = []
        S_set = set()
        
        # DP_all[k] : bitset of sums of k elements from S
        # DP_leave[i][k] : bitset of sums of k elements from S \ {S[i]}
        dp_all = [1]  # dp_all[0] = 1
        dp_leave = [] # List of DPs for leave-one-out sets
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            # Check 1: Is x the mean of a subset of S?
            # x is mean of subset of size k if bit k*x is set in dp_all[k]
            valid = True
            for k in range(2, len(S) + 1):
                target = k * x
                # Optimization: if target > max possible sum, break
                # But we don't track max sum explicitly, so just check bit
                if k < len(dp_all):
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: Does adding x make any existing element a_i the mean of a subset of S U {x} \ {a_i}?
            # For each a_i in S (index i):
            # We need to check if there is a subset T of S \ {a_i} such that
            # {x} U T has mean a_i.
            # sum(T) + x = (|T| + 1) * a_i
            # sum(T) = (|T| +  1) * a_i - x
            # Let j = |T|. We need to check if bit ((j+1)*a_i - x) is set in dp_leave[i][j].
            # Note: dp_leave[i] corresponds to S \ {a_i}.
            
            for i, a_i in enumerate(S):
                # dp_leave[i] is the DP for S \ {a_i}
                # The size of S \ {a_i} is len(S) - 1
                # j can range from 1 to len(S) - 1
                # (j=0 implies T is empty, subset is {x}, size 1, but condition requires size >= 2)
                # So j >= 1.
                
                # We need to access dp_leave[i][j]
                # dp_leave[i] has length (len(S)-1) + 1 = len(S)
                # So indices 0 .. len(S)-1 are valid.
                
                for j in range(1, len(S)):
                    target_sum = (j + 1) * a_i - x
                    if target_sum < 0:
                        continue
                    # Check if target_sum is achievable with j elements from S \ {a_i}
                    # dp_leave[i][j] contains sums of j elements
                    if j < len(dp_leave[i]):
                        if (dp_leave[i][j] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Add x to S
                S.append(x)
                S_set.add(x)
                
                # Update DP_all
                # New DP_all will have length len(S) + 1
                old_dp_all = dp_all
                new_dp_all = [0] * (len(S) + 1)
                new_dp_all[0] = 1
                for k in range(len(S) - 1, -1, -1):
                    if old_dp_all[k]:
                        new_dp_all[k] |= old_dp_all[k]
                        new_dp_all[k+1] |= (old_dp_all[k] << x)
                dp_all = new_dp_all
                
                # Update DP_leave
                new_dp_leave = []
                # For existing elements, their leave-one-out set includes x
                for i in range(len(S) - 1):
                    old_dp_leave_i = dp_leave[i]
                    # The set is (S \ {S[i]}) U {x}
                    # Size increases by 1
                    new_dp_leave_i = [0] * (len(old_dp_leave_i) + 1)
                    new_dp_leave_i[0] = 1
                    for k in range(len(old_dp_leave_i) - 1, -1, -1):
                        if old_dp_leave_i[k]:
                            new_dp_leave_i[k] |= old_dp_leave_i[k]
                            new_dp_leave_i[k+1] |= (old_dp_leave_i[k] << x)
                    new_dp_leave.append(new_dp_leave_i)
                
                # For the new element x, its leave-one-out set is the old S
                # So dp_leave_new.append(old_dp_all)
                new_dp_leave.append(old_dp_all)
                
                dp_leave = new_dp_leave
        
        return S

    def local_search(S):
        """
        Perform local search swaps to improve the set.
        """
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng = random.Random()
            rng.shuffle(available)
            rng.shuffle(S)
            
            # Try to add if size < target
            if len(S) < target_size:
                for x in available:
                    if time.time() > deadline:
                        break
                    S_test = S + [x]
                    if is_valid_set(S_test):
                        S = S_test
                        S_set.add(x)
                        available.remove(x)
                        improved = True
                        break
            
            if len(S) >= target_size:
                break
            
            # If size is target or cannot add, try swaps
            # We want to maximize cardinality, so if we are at target, we don't need to swap for size.
            # But maybe we are stuck at size < target. Swaps might help find a valid set of larger size?
            # Actually, if we are stuck at size K < target, it means no element from 'available' can be added.
            # Swapping an element a in S with b in available might allow further additions.
            
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_without_old = [x for x in S if x != old]
                    
                    for x in available:
                        if time.time() > deadline:
                            break
                        S_test = S_without_old + [x]
                        if is_valid_set(S_test):
                            # Check if this swap allows us to add more elements?
                            # For now, just accept if valid.
                            # We might want to prioritize swaps that allow growth, but is_valid_set is the gate.
                            S = S_test
                            S_set.discard(old)
                            S_set.add(x)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    # Main logic
    best_set = []
    best_size = 0
    
    # 1. Greedy with different orders
    # Try descending
    S = greedy_build_fast(range(n_max, 0, -1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Try ascending
    S = greedy_build_fast(range(1, n_max + 1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Try random orders
    rng = random.Random(42)
    for i in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S = greedy_build_fast(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        elif len(S) == best_size and len(S) > 0:
            # Maybe keep multiple candidates? For now just keep the first one found of max size
            pass

    # 2. Local Search
    if best_size < target_size and time.time() < deadline:
        S = local_search(best_set)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set

# Quick self-test
if __name__ == "__main__":
    s = construct()
    print(len(s))
    print(s[:10])
    # Verify
    def verify(S):
        n = len(S)
        for i, a in enumerate(S):
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2: continue
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            for k in range(2, m + 1):
                target = k * a
                if (dp[k] >> target) & 1:
                    print(f"Invalid: {a} is mean of subset size {k}")
                    return False
        return True
    print(verify(s))

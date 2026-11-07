import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def try_greedy_build(order):
        S = []
        n = 0
        # DP[k] is a set of sums
        dp = [set() for _ in range(1)] # Start with n=0, dp[0]={0}
        dp[0].add(0)
        
        # Union of all sums
        all_sums = {0}
        
        for x in order:
            if x in S:
                continue
            
            # Check if x can be added
            # New cardinality will be n+1
            # New sums for cardinality k (1 <= k <= n+1) come from old dp[k-1]
            
            # We need to check collisions for each k
            valid = True
            new_dp = [set(s) for s in dp] # Copy current dp
            # Actually, we don't need to copy everything if we just check.
            # But we need to update dp if valid.
            
            # Let's compute the new sums for each k
            # new_sums[k] = {s + x for s in dp[k-1]}
            # We need to check if new_sums[k] intersects with (all_sums - dp[k])
            
            # Optimization: 
            # For each k, if new_sums[k] has any element in all_sums that is NOT in dp[k], invalid.
            
            # We can precompute for each k: forbidden_sums[k] = all_sums - dp[k]
            # But all_sums changes as we add elements? No, all_sums is the union of current dp.
            # If we add x, the new all_sums will be all_sums U (union of new_sums).
            # But for the check, we only care about collisions with EXISTING sums.
            
            # So:
            # For k in 1..n+1:
            #   candidates = {s+x for s in dp[k-1]}
            #   conflict = candidates & (all_sums - dp[k])
            #   if conflict: valid = False; break
            
            # Note: dp[k] for k=n+1 is empty initially.
            
            if n + 1 > len(dp):
                # Extend dp if needed? 
                # We are adding one element, so max cardinality increases by 1.
                # We can just handle it in the loop.
                pass

            for k in range(1, n + 2):
                if k - 1 < len(dp):
                    prev_sums = dp[k-1]
                    if not prev_sums:
                        continue
                    new_sums = set()
                    for s in prev_sums:
                        new_sums.add(s + x)
                    
                    # Get existing sums for this cardinality k
                    if k < len(dp):
                        existing_k = dp[k]
                    else:
                        existing_k = set()
                    
                    # Forbidden sums are those in all_sums but not in existing_k
                    # However, all_sums includes sums from ALL cardinalities.
                    # So we check:
                    # Is there any y in new_sums such that y in all_sums AND y not in existing_k?
                    
                    # Efficient check:
                    # Iterate new_sums.
                    # If y in all_sums and y not in existing_k: conflict.
                    
                    # To speed up, we can just check:
                    # conflict = new_sums & all_sums
                    # if conflict:
                    #    # Check if any of these conflicts are actually in existing_k (which is allowed)
                    #    for y in conflict:
                    #        if y not in existing_k:
                    #            valid = False
                    #            break
                    #    if not valid: break
                    
                    # Even faster:
                    # forbidden = all_sums - existing_k
                    # if new_sums & forbidden:
                    #    valid = False
                    #    break
                    
                    # Creating `forbidden` set every time might be slow if all_sums is large.
                    # But all_sums size is bounded by max_sum (~256*n).
                    # For n=15, max_sum=3840. Set of 3840 ints is small.
                    
                    forbidden = all_sums - existing_k
                    if new_sums & forbidden:
                        valid = False
                        break
            
            if valid:
                # Commit x
                S.append(x)
                n += 1
                # Update dp
                # new_dp[k] = dp[k] U {s+x for s in dp[k-1]}
                # We can update in place if we go from high k to low k?
                # No, dp[k] depends on old dp[k-1].
                # So we need a copy or a new list.
                old_dp = dp
                dp = [set() for _ in range(n + 1)]
                for k in range(n + 1):
                    dp[k] = set(old_dp[k]) if k < len(old_dp) else set()
                    if k > 0 and k-1 < len(old_dp):
                        for s in old_dp[k-1]:
                            dp[k].add(s + x)
                
                # Update all_sums
                for k in range(n + 1):
                    all_sums.update(dp[k])

        return S

    best_set = []
    
    # Generate diverse orders
    orders = []
    # 1. Sorted
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Random
    for _ in range(100):
        if time.time() > deadline:
            break
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)
        
    # 3. Biased: Small numbers first (to keep sums small?)
    #    Or Large numbers first?
    #    Let's try a mix.
    
    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            
    return [int(x) for x in best_set]

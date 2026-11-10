<<<<<<< SEARCH
    def local_search(S, S_set, dp, max_sum=24576):
        """
        Improve the set by adding elements or swapping.
        """
        if not S:
            return S, S_set, dp
        
        S = S[:]
        S_set = set(S_set)
        best_len = len(S)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        
        # Rebuild dp if not provided correctly or to be safe
        dp = [0] * (MAX_SIZE + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)

        improved = True
        while improved and len(S) < MAX_SIZE:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add
            # Sort candidates by some heuristic? Random is fine.
            random.shuffle(candidates)
            for x in candidates:
                if len(S) >= MAX_SIZE:
                    break
                if check_3term_ap_conflict(x, S_set):
                    continue
                if is_valid_candidate(x, S, dp, max_sum):
                    S.append(x)
                    S_set.add(x)
                    for k in range(len(S) - 1, 0, -1):
                        dp[k+1] |= (dp[k] << x)
                    dp[1] |= (dp[0] << x)
                    candidates.remove(x)
                    improved = True
                    best_len = len(S)
                    if best_len == MAX_SIZE:
                        return S, S_set, dp
                    break # Restart adding loop with new state
            
            # 2. Try to swap if no add found
            if len(S) < MAX_SIZE:
                continue
                
            # If we are at max size, we can still try swaps to potentially allow more?
            # No, max size is 48. If we are at 48, we are done.
            # The task says "Maximize cardinality. At most 48 integers".
            # So if we reach 48, we stop.
            
            if len(S) == MAX_SIZE:
                return S, S_set, dp

            # If we couldn't add, try swaps to unblock?
            # Swapping doesn't increase size. It only changes composition.
            # We only care about size. So if we can't add, we are stuck in this local optimum.
            # We should restart with a different ordering.
            
        return S, S_set, dp
=======
    def local_search(S, S_set, dp, max_sum=24576):
        """
        Improve the set by adding elements or swapping.
        """
        if not S:
            return S, S_set, dp
        
        S = S[:]
        S_set = set(S_set)
        best_len = len(S)
        
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        
        # Rebuild dp if not provided correctly or to be safe
        dp = [0] * (MAX_SIZE + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, 0, -1):
                dp[k+1] |= (dp[k] << x)
            dp[1] |= (dp[0] << x)

        improved = True
        swap_attempts = 0
        max_swap_attempts = 50
        
        while improved and len(S) < MAX_SIZE:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add
            random.shuffle(candidates)
            for x in candidates:
                if len(S) >= MAX_SIZE:
                    break
                if check_3term_ap_conflict(x, S_set):
                    continue
                if is_valid_candidate(x, S, dp, max_sum):
                    S.append(x)
                    S_set.add(x)
                    for k in range(len(S) - 1, 0, -1):
                        dp[k+1] |= (dp[k] << x)
                    dp[1] |= (dp[0] << x)
                    candidates.remove(x)
                    improved = True
                    best_len = len(S)
                    if best_len == MAX_SIZE:
                        return S, S_set, dp
                    break # Restart adding loop with new state
            
            # 2. If we couldn't add, try swaps to unblock
            if not improved and len(S) < MAX_SIZE and swap_attempts < max_swap_attempts:
                swap_attempts += 1
                # Try a random swap
                if S and candidates:
                    e = random.choice(S)
                    c = random.choice(candidates)
                    
                    # Check if (S - {e}) U {c} is valid
                    S_temp = S[:]
                    S_temp.remove(e)
                    S_set_temp = S_set - {e}
                    dp_temp = [0] * (MAX_SIZE + 1)
                    dp_temp[0] = 1
                    for y in S_temp:
                        for k in range(len(S_temp) - 1, 0, -1):
                            dp_temp[k+1] |= (dp_temp[k] << y)
                        dp_temp[1] |= (dp_temp[0] << y)
                    
                    if not check_3term_ap_conflict(c, S_set_temp) and is_valid_candidate(c, S_temp, dp_temp, max_sum):
                        # Perform swap
                        S = S_temp + [c]
                        S_set = S_set_temp | {c}
                        candidates.remove(c)
                        if e not in candidates:
                            candidates.append(e)
                        dp = rebuild_dp(S)
                        improved = True # Force another loop iteration to try adding
            
        return S, S_set, dp
>>>>>>> REPLACE

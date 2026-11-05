import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Target: find 7 distinct denominators >= 5 summing to 1
    # Use DFS
    target_num, target_den = 1, 1
    
    def dfs(rem_num, rem_den, min_d, depth, path):
        if time.time() > deadline:
            return None
        if depth == 7:
            if rem_num == 0:
                return path
            return None
        
        # Pruning: max possible sum of remaining terms
        # If sum of 1/min_d + 1/(min_d+1) + ... < rem, fail
        # Simple bound: (7-depth) * (1/min_d) < rem -> fail
        # rem = rem_num/rem_den
        if (7 - depth) * (1.0 / min_d) < rem_num / rem_den:
            return None
            
        # Also, if rem_num/rem_den > (7-depth)/min_d, it's the same check.
        
        # Iterate possible next denominators
        # d must be >= min_d
        # 1/d <= rem -> d >= 1/rem
        # Also 1/d > rem / (7-depth) -> d < (7-depth)/rem
        max_d = int((7 - depth) * rem_den / rem_num) + 1
        if max_d < min_d:
            return None
            
        for d in range(min_d, max_d + 1):
            if d < 5:
                continue
            if rem_num * d == rem_den:
                # Exact match, but we need more terms? No, if depth < 7 and rem becomes 0, we need 0 for rest.
                # But we need distinct denominators. If rem becomes 0, remaining terms must be 0, impossible for unit fractions.
                # So we only accept if depth == 7 and rem == 0.
                # If rem becomes 0 before depth 7, it's invalid unless we can add 0s, which we can't.
                pass
            
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break
            if new_num == 0:
                if depth == 6: # One term left, must be 0? No.
                    # If rem becomes 0, we have used 'depth' terms.
                    # We need 7 terms. If we have 6 terms and sum is 1, the 7th term must be 0. Impossible.
                    # So we only accept if depth == 7 and rem == 0.
                    # Wait, if depth == 7, we check rem==0 at start of loop or after adding?
                    # My logic: check rem==0 at depth==7.
                    # So if new_num == 0 and depth < 6, we can't fill remaining slots with 0.
                    # So we skip new_num == 0 unless depth == 6 (then next depth is 7, rem is 0, accepted).
                    if depth == 6:
                         # Next call will be depth 7, rem 0.
                         res = dfs(0, 1, d + 1, depth + 1, path + [d])
                         if res:
                             return res
                continue
            
            # Reduce fraction? Optional for speed, but numbers can get big.
            # For now, keep as is.
            res = dfs(new_num, rem_den * d, d + 1, depth + 1, path + [d])
            if res:
                return res
        return None

    result = dfs(1, 1, 5, 0, [])
    if result:
        return result
    
    # Fallback to seed-like behavior if DFS fails (unlikely for k=7)
    # But we can just return a known k=11 solution or similar.
    # Actually, if DFS fails, it might be timeout.
    # Let's add a hardcoded k=7 solution if we can find one.
    # Since I can't verify a hardcoded one easily here, I rely on DFS.
    # If DFS times out, we return the seed solution.
    
    # Seed solution logic (fast)
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no representation")

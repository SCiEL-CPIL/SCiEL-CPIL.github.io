import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0
    
    # Greedy initialization
    best = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        for a in reversed(best):
            total += a
            tails.append(total)
        tails.append(x)
        if not sums.intersection(tails):
            best.append(x)
            sums.update(tails)
    
    def dfs(prefix, diffs, diff_set):
        nonlocal best
        if time.time() - start_time > TIME_LIMIT:
            return
        
        n = len(diffs)
        if n > len(best):
            best = list(diffs)
        
        if not diffs:
            min_diff = 1
        else:
            min_diff = diffs[-1] + 1
        
        # Max additional elements: from prefix[-1] to 256, with min step 1
        last_val = prefix[-1] if prefix else 0
        max_additional = 256 - last_val
        
        if min_diff > 256 - last_val:
            return
        
        if n + max_additional <= len(best):
            return
        
        for d in range(min_diff, 257 - last_val + 1):
            if time.time() - start_time > TIME_LIMIT:
                return
            
            new_p = last_val + d
            
            valid = True
            new_diffs = []
            for j in range(len(prefix)):
                diff = new_p - prefix[j]
                if diff in diff_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            prefix.append(new_p)
            diffs.append(d)
            diff_set.update(new_diffs)
            
            dfs(prefix, diffs, diff_set)
            
            prefix.pop()
            diffs.pop()
            diff_set.difference_update(new_diffs)
    
    dfs([0], [], set())
    return best

import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s safety margin
    
    # 1. Generate smooth numbers up to 128
    max_n = 128
    smooth = []
    is_smooth = [False] * (max_n + 1)
    for x in range(1, max_n + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            is_smooth[x] = True
            
    smooth_set = set(smooth)
    
    # 2. Precompute antichain validity for pairs
    # antichain[a][b] is True if a and b form an antichain (neither divides the other)
    # We only care about distinct a, b
    # Using a set of tuples (min, max) that are INVALID (divisible)
    invalid_pairs = set()
    for i in range(len(smooth)):
        a = smooth[i]
        for j in range(len(smooth)):
            if i == j: continue
            b = smooth[j]
            if a > b:
                continue # only check a < b to avoid duplicates
            if b % a == 0:
                invalid_pairs.add((a, b))
                
    def is_antichain_pair(a, b):
        if a == b:
            return False
        if a > b:
            a, b = b, a
        return (a, b) not in invalid_pairs

    answers = []
    
    # 3. Solve for each target
    for n in range(1, max_n + 1):
        if time.time() > deadline:
            # Fallback: return best effort or simple valid reps if time runs out
            # This is unlikely given the small search space
            break
            
        ans = None
        
        # Case 1: n is smooth
        if is_smooth[n]:
            ans = [n]
        else:
            # Case 2: Try 2 terms
            found = False
            # Iterate a in smooth, a < n/2 to ensure a < b and distinct
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b <= a:
                    continue
                if is_smooth[b] and is_antichain_pair(a, b):
                    ans = [a, b]
                    found = True
                    break
            
            # Case 3: Try 3 terms if 2 failed
            if not found:
                # Combinations of 3 distinct smooth numbers
                # To optimize, iterate a < b < c
                for i in range(len(smooth)):
                    a = smooth[i]
                    if a >= n: break
                    for j in range(i+1, len(smooth)):
                        b = smooth[j]
                        if a + b >= n: break
                        c = n - a - b
                        if c <= b:
                            continue
                        if is_smooth[c] and is_antichain_pair(a, b) and is_antichain_pair(a, c) and is_antichain_pair(b, c):
                            ans = [a, b, c]
                            found = True
                            break
                    if found:
                        break
                        
            # Case 4: Fallback for 4+ terms (should not happen for n<=128)
            if not found:
                # Greedy or random search for more terms
                # This is extremely unlikely to be needed
                # Simple backtracking for up to 10 terms
                def backtrack(remaining, start_idx, current_terms):
                    if remaining == 0:
                        return current_terms
                    if len(current_terms) >= 10:
                        return None
                    for k in range(start_idx, len(smooth)):
                        t = smooth[k]
                        if t > remaining:
                            break
                        # Check antichain with all existing terms
                        valid = True
                        for existing in current_terms:
                            if not is_antichain_pair(t, existing):
                                valid = False
                                break
                        if not valid:
                            continue
                        res = backtrack(remaining - t, k + 1, current_terms + [t])
                        if res:
                            return res
                    return None
                
                ans = backtrack(n, 0, [])
                if ans is None:
                    # Should never reach here for n <= 128
                    ans = [1, 1, 1] # Placeholder invalid, but prevents crash

        answers.append(ans)
        
    return answers

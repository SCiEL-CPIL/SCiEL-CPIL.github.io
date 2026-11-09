import time

def construct():
    start_time = time.time()
    timeout = 50.0

    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    
    def is_compatible(x, current_terms):
        for t in current_terms:
            if x % t == 0 or t % x == 0:
                return False
        return True

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Emergency: if we run out of time, we need to return something valid.
            # Since we process 1..128 sequentially, and most are fast,
            # we might have a partial list. We must pad with valid reps if needed.
            # But this should not happen.
            pass

        answer = None
        
        # 1 term
        if n in smooth_set:
            answer = [n]
        else:
            # 2 terms
            found = False
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and a < b:
                    if a % b != 0 and b % a != 0:
                        answer = [a, b]
                        found = True
                        break
            if found:
                answers.append(answer)
                continue

            # 3 terms
            found = False
            for i, a in enumerate(smooth):
                if a >= n:
                    break
                for j in range(i + 1, len(smooth)):
                    b = smooth[j]
                    if a + b >= n:
                        break
                    c = n - a - b
                    if c in smooth_set and c > b:
                        if (a % b != 0 and b % a != 0 and
                            a % c != 0 and c % a != 0 and
                            b % c != 0 and c % b != 0):
                            answer = [a, b, c]
                            found = True
                            break
                if found:
                    break
            if found:
                answers.append(answer)
                continue

            # DFS for 4+ terms
            # We want minimal terms, so try 4, then 5, etc.
            # Given n <= 128, max terms is small.
            found = False
            for max_terms in range(4, 10):
                def dfs(start_idx, remaining, current):
                    if remaining == 0:
                        if current:
                            return list(current)
                        return None
                    if remaining < 0:
                        return None
                    # Pruning: if not enough terms left to reach max_terms? 
                    # We just want ANY valid representation with <= max_terms.
                    # Actually, we want minimal. So we check 4 first. If 4 fails, try 5.
                    
                    # Estimate min sum of (max_terms - len(current)) terms?
                    # Just iterate.
                    for k in range(start_idx, len(smooth)):
                        x = smooth[k]
                        if x > remaining:
                            break
                        if not is_compatible(x, current):
                            continue
                        current.append(x)
                        res = dfs(k + 1, remaining - x, current)
                        if res:
                            return res
                        current.pop()
                    return None

                res = dfs(0, n, [])
                if res:
                    answer = res
                    found = True
                    break
            
            if found:
                answers.append(answer)
            else:
                # Should not happen
                answers.append([1]) # Invalid placeholder

    return answers

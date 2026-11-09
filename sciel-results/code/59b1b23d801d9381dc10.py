import time
import bisect

def construct():
        start_time = time.time()
        deadline = start_time + 50.0  # 10s margin
        
        n = 5040
        target = 4979
        
        divisors = [d for d in range(1, n + 1) if n % d == 0]
        odd_divs = [d for d in divisors if d % 2 != 0]
        even_divs = [d for d in divisors if d % 2 == 0]
        even_divs.sort()
        
        # Search for k=4
        for d1 in odd_divs:
            if time.time() > deadline:
                break
            rem_target = target - d1
            if rem_target < 0:
                continue
            # Find 3 distinct even divisors summing to rem_target
            # e1 < e2 < e3
            le = len(even_divs)
            for i in range(le):
                if time.time() > deadline:
                    break
                if even_divs[i] >= rem_target:
                    break
                rem2 = rem_target - even_divs[i]
                for j in range(i + 1, le):
                    if time.time() > deadline:
                        break
                    if even_divs[j] >= rem2:
                        break
                    rem3 = rem2 - even_divs[j]
                    # e3 must be > e2 and in even_divs
                    k = bisect.bisect_left(even_divs, rem3, j + 1)
                    if k < le and even_divs[k] == rem3:
                        return [d1, even_divs[i], even_divs[j], even_divs[k]]
        
        # Fallback to greedy k=5 (or whatever greedy produces)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    return chosen
        
        # If greedy didn't work (should not happen), try brute force for k=5?
        # Greedy is known to work for this specific instance.
        return chosen

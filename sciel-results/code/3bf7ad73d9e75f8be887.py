import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    div_set = set(divisors)
    
    # Try increasing number of divisors
    for k in range(1, len(divisors) + 1):
        if time.time() > deadline:
            break
        # For small k, use combinations with pruning
        if k == 1:
            if target in div_set:
                return [target]
        elif k == 2:
            for d in divisors:
                if d >= target:
                    continue
                rem = target - d
                if rem in div_set and rem != d:
                    return [d, rem]
        elif k == 3:
            # Meet in the middle or triple loop with pruning
            # Sort ascending for better pruning
            divs_asc = sorted(divisors)
            for i in range(len(divs_asc)):
                d1 = divs_asc[i]
                if d1 >= target:
                    break
                rem2 = target - d1
                # Find two distinct divisors (not d1) summing to rem2
                for j in range(i + 1, len(divs_asc)):
                    d2 = divs_asc[j]
                    if d2 >= rem2:
                        break
                    d3 = rem2 - d2
                    if d3 <= d2:
                        continue
                    if d3 in div_set and d3 != d1 and d3 != d2:
                        return [d1, d2, d3]
        elif k == 4:
            # Use combinations from itertools for k=4
            # 60 choose 4 = 487635, very fast
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    return list(combo)
                if time.time() > deadline:
                    return None  # fallback, shouldn't happen
        else:
            # For k >= 5, use backtracking with pruning
            # Try combinations but with early termination
            found = None
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    found = list(combo)
                    break
                if time.time() > deadline:
                    break
            if found:
                return found
    
    # Fallback: greedy (should not reach here)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    assert remainder == 0, f"Could not reach target, remainder={remainder}"
    return chosen

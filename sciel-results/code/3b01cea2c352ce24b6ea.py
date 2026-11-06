"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    num_divs = len(divisors)  # 60 divisors
    div_set = set(divisors)
    
    # k=1: check if target is a divisor
    if target in div_set:
        return [target]
    
    # k=2: check all pairs
    for i in range(num_divs):
        complement = target - divisors[i]
        if complement > divisors[i] and complement in div_set:
            return [divisors[i], complement]
    
    # k=3: check all triples (60 choose 3 = 34220, fast)
    for i in range(num_divs):
        for j in range(i+1, num_divs):
            s2 = divisors[i] + divisors[j]
            if s2 >= target:
                break
            complement = target - s2
            if complement > divisors[j] and complement in div_set:
                if complement != divisors[i] and complement != divisors[j]:
                    return [divisors[i], divisors[j], complement]
    
    # k=4: 60 choose 4 = 487635, still manageable
    for i in range(num_divs):
        if time.time() > deadline - 15:
            break
        for j in range(i+1, num_divs):
            s2 = divisors[i] + divisors[j]
            if s2 >= target:
                break
            for k in range(j+1, num_divs):
                s3 = s2 + divisors[k]
                if s3 >= target:
                    break
                complement = target - s3
                if complement > divisors[k] and complement in div_set:
                    if complement not in (divisors[i], divisors[j], divisors[k]):
                        return [divisors[i], divisors[j], divisors[k], complement]
    
    # k=5: Use meet-in-the-middle approach for efficiency
    # Split divisors into two halves
    half = num_divs // 2
    first_half = divisors[:half]
    second_half = divisors[half:]
    
    # Generate all sums of 2 elements from first half with their indices
    sums_2_first = {}
    for i in range(len(first_half)):
        for j in range(i+1, len(first_half)):
            s = first_half[i] + first_half[j]
            if s < target:
                if s not in sums_2_first:
                    sums_2_first[s] = []
                sums_2_first[s].append((i, j))
    
    # Generate all sums of 3 elements from second half with their indices
    sums_3_second = {}
    for i in range(len(second_half)):
        for j in range(i+1, len(second_half)):
            s2 = second_half[i] + second_half[j]
            if s2 >= target:
                break
            for k in range(j+1, len(second_half)):
                s3 = s2 + second_half[k]
                if s3 >= target:
                    break
                if s3 not in sums_3_second:
                    sums_3_second[s3] = []
                sums_3_second[s3].append((i, j, k))
    
    # Try to combine: 2 from first + 3 from second = 5 total
    for s1, indices1 in sums_2_first.items():
        if time.time() > deadline - 5:
            break
        complement = target - s1
        if complement in sums_3_second:
            indices2 = sums_3_second[complement]
            # Check for no overlap (shouldn't happen since halves are disjoint)
            return [first_half[indices1[0][0]], first_half[indices1[0][1]],
                    second_half[indices2[0][0]], second_half[indices2[0][1]], second_half[indices2[0][2]]]
    
    # Also try 3 from first + 2 from second
    sums_3_first = {}
    for i in range(len(first_half)):
        for j in range(i+1, len(first_half)):
            s2 = first_half[i] + first_half[j]
            if s2 >= target:
                break
            for k in range(j+1, len(first_half)):
                s3 = s2 + first_half[k]
                if s3 >= target:
                    break
                if s3 not in sums_3_first:
                    sums_3_first[s3] = []
                sums_3_first[s3].append((i, j, k))
    
    sums_2_second = {}
    for i in range(len(second_half)):
        for j in range(i+1, len(second_half)):
            s = second_half[i] + second_half[j]
            if s < target:
                if s not in sums_2_second:
                    sums_2_second[s] = []
                sums_2_second[s].append((i, j))
    
    for s1, indices1 in sums_3_first.items():
        if time.time() > deadline - 5:
            break
        complement = target - s1
        if complement in sums_2_second:
            indices2 = sums_2_second[complement]
            return [first_half[indices1[0][0]], first_half[indices1[0][1]], first_half[indices1[0][2]],
                    second_half[indices2[0][0]], second_half[indices2[0][1]]]
    
    # k=6: Use greedy + local search
    # Greedy: take largest possible divisors
    def greedy_solve():
        rem = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
            if rem == 0:
                break
        if rem == 0:
            return chosen
        return None
    
    greedy_result = greedy_solve()
    if greedy_result:
        return greedy_result
    
    # Fallback: try combinations of small divisors for the difference
    diff = n - target  # 61
    small_divs = [d for d in divisors if d <= diff]
    for r in range(1, len(small_divs)+1):
        for comb in combinations(small_divs, r):
            if sum(comb) == diff:
                result = [d for d in divisors if d not in set(comb)]
                return result
        if time.time() > deadline:
            break
    
    # Ultimate fallback
    return greedy_result if greedy_result else divisors
# EVOLVE-BLOCK-END

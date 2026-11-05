# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Generate all 2^a * 3^b * 5^c smooth numbers up to 128
    smooth = []
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.append(c)
                c *= 5
            b *= 3
        a *= 2
    smooth = sorted(set(smooth))
    
    # Precompute divisibility relations
    # divs[x] = set of smooth numbers that divide x
    divs = {}
    for x in smooth:
        divs[x] = set()
        for y in smooth:
            if y < x and x % y == 0:
                divs[x].add(y)
    
    # For each target n, find minimal antichain representation
    # Search from k=1 upward
    
    def find_min_antichain(n, max_k=30):
        """Find a minimal antichain of smooth numbers summing to n."""
        # Candidate smooth numbers <= n
        candidates = [x for x in smooth if x <= n]
        
        for k in range(1, max_k + 1):
            if time.time() > DEADLINE:
                return None
            # Try to find k distinct smooth numbers from candidates
            # that sum to n and form an antichain
            # Use DFS with pruning
            
            # Sort candidates to help pruning
            # We need to pick k numbers from candidates that sum to n
            # and no one divides another
            
            def dfs(start_idx, remaining_sum, chosen, k_left):
                if time.time() > DEADLINE:
                    return None
                if k_left == 0:
                    if remaining_sum == 0:
                        return chosen
                    return None
                if remaining_sum < 0:
                    return None
                if start_idx >= len(candidates):
                    return None
                if k_left > len(candidates) - start_idx:
                    return None
                
                # Pruning: even if we take the largest k_left numbers,
                # can we reach remaining_sum?
                # And even if we take the smallest k_left numbers,
                # can we not exceed remaining_sum?
                
                # Try each candidate from start_idx
                for i in range(start_idx, len(candidates)):
                    val = candidates[i]
                    if val > remaining_sum:
                        break
                    # Check antichain: val should not divide any chosen,
                    # and no chosen should divide val
                    valid = True
                    for c in chosen:
                        if val % c == 0 or c % val == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    result = dfs(i + 1, remaining_sum - val, chosen + [val], k_left - 1)
                    if result is not None:
                        return result
                return None
            
            result = dfs(0, n, [], k)
            if result is not None:
                return result
        return None
    
    answers = []
    for n in range(1, 129):
        if time.time() > DEADLINE:
            # Fill remaining with any valid answer (use seed-like approach)
            # Try simple approach
            for count in (3, 2, 1):
                found = False
                for terms in combinations([x for x in smooth if x <= n], count):
                    if sum(terms) == n and all(b % a != 0 for a, b in combinations(terms, 2)):
                        answers.append(list(terms))
                        found = True
                        break
                if found:
                    break
            if not found:
                answers.append([n] if n in smooth else [1, n-1] if n-1 > 0 else [1])
            continue
        
        result = find_min_antichain(n)
        if result is None:
            # Fallback: try seed approach
            found = False
            for count in (3, 2, 1):
                for terms in combinations([x for x in smooth if x <= n], count):
                    if sum(terms) == n and all(b % a != 0 for a, b in combinations(terms, 2)):
                        answers.append(list(terms))
                        found = True
                        break
                if found:
                    break
            if not found:
                # Last resort: just use [n] if smooth, else something
                if n in smooth:
                    answers.append([n])
                else:
                    # Try to find any valid representation
                    for count in range(1, 31):
                        for terms in combinations([x for x in smooth if x <= n], count):
                            if sum(terms) == n and all(b % a != 0 for a, b in combinations(terms, 2)):
                                answers.append(list(terms))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        answers.append([1] * min(n, 30) if n <= 30 else [1]*30)
        else:
            answers.append(result)
    
    return answers
# EVOLVE-BLOCK-END

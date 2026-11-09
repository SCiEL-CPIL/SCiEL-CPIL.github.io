# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit

    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_numbers.sort()
    n_smooth = len(smooth_numbers)
    
    # 2. Precompute divisibility relationships
    # divides[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(n_smooth):
            if i != j:
                vj = smooth_numbers[j]
                if vj % vi == 0:
                    divides[i][j] = True
    
    # 3. Precompute best (minimum size) antichain for each sum 1..128
    # best[s] = (size, list_of_terms)
    best = {}
    
    # Use backtracking to generate all antichains with sum <= 128
    # We maintain the current set and check antichain property incrementally
    
    def backtrack(idx, current_sum, current_set, current_indices):
        if current_sum > 128:
            return
        
        # Record this antichain if non-empty
        if current_set:
            s = current_sum
            if s not in best or len(current_set) < best[s][0]:
                best[s] = (len(current_set), list(current_set))
        
        # Try adding more elements
        for i in range(idx, n_smooth):
            val = smooth_numbers[i]
            new_sum = current_sum + val
            if new_sum > 128:
                break  # Since sorted, further values will be larger
            
            # Check if val can be added: must not divide or be divided by any in current_set
            # Since smooth_numbers is sorted and we pick increasing indices,
            # current_set elements are <= val. So we only need to check if any divides val.
            valid = True
            for j in current_indices:
                if divides[j][i]:  # smooth_numbers[j] divides smooth_numbers[i]
                    valid = False
                    break
                if divides[i][j]:  # smooth_numbers[i] divides smooth_numbers[j] (shouldn't happen if sorted and distinct)
                    valid = False
                    break
            
            if valid:
                current_set.append(val)
                current_indices.append(i)
                backtrack(i + 1, new_sum, current_set, current_indices)
                current_set.pop()
                current_indices.pop()
    
    backtrack(0, 0, [], [])
    
    # 4. Build answers for targets 1..128
    answers = []
    for n in range(1, 129):
        if n in best:
            answers.append(best[n][1])
        else:
            # Fallback: should not happen, but find any valid antichain
            found = None
            for k in range(1, 8):
                candidates = [x for x in smooth_numbers if x <= n]
                for terms in combinations(candidates, k):
                    if sum(terms) == n:
                        # Check antichain
                        is_anti = True
                        for a in range(k):
                            for b in range(a+1, k):
                                if terms[b] % terms[a] == 0 or terms[a] % terms[b] == 0:
                                    is_anti = False
                                    break
                            if not is_anti:
                                break
                        if is_anti:
                            found = list(terms)
                            break
                if found:
                    break
            if found:
                answers.append(found)
            else:
                # Last resort: this should never happen for 1..128
                answers.append([n] if n in smooth_numbers else [])
    
    return answers

# EVOLVE-BLOCK-END

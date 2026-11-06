"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Generate all 5-smooth numbers <= 128
    smooth = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(i)
    
    smooth_set = set(smooth)
    M = len(smooth)
    
    # Precompute divisibility: divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j:
                a, b = smooth[i], smooth[j]
                if a > b:
                    divs[i][j] = (b % a == 0)  # a divides b? No, b < a so a can't divide b
                    # Actually we want: does smooth[i] divide smooth[j]?
                    divs[i][j] = (b % a == 0) if b >= a else False
                else:
                    divs[i][j] = (b % a == 0)
    
    # More carefully: divs[i][j] = True if smooth[i] divides smooth[j]
    for i in range(M):
        for j in range(M):
            if i == j:
                divs[i][j] = False
            elif smooth[j] >= smooth[i]:
                divs[i][j] = (smooth[j] % smooth[i] == 0)
            else:
                divs[i][j] = False
    
    # For antichain check: no pair (i,j) with i!=j such that divs[i][j] or divs[j][i]
    # i.e., neither divides the other.
    
    def is_compatible(idx1, idx2):
        return not (divs[idx1][idx2] or divs[idx2][idx1])
    
    answers = []
    
    for n in range(1, 129):
        found = None
        
        # Try k = 1, 2, 3, ... up to some limit
        for k in range(1, 11):
            if k == 1:
                if n in smooth_set:
                    found = [n]
                    break
                continue
            
            if k == 2:
                found2 = None
                for i in range(M):
                    a = smooth[i]
                    if a >= n:
                        break
                    b = n - a
                    if b in smooth_set and b > a:
                        # Find index of b
                        # Use binary search or precomputed dict
                        # We'll just check divisibility
                        if b % a != 0:  # a doesn't divide b; b > a so b can't divide a
                            found2 = [a, b]
                            break
                if found2:
                    found = found2
                    break
                continue
            
            if k == 3:
                found3 = None
                for i in range(M):
                    a = smooth[i]
                    if a >= n:
                        break
                    for j in range(i + 1, M):
                        b = smooth[j]
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c in smooth_set and c > b:
                            # Check pairwise antichain
                            if (b % a != 0) and (c % a != 0) and (c % b != 0):
                                found3 = [a, b, c]
                                break
                    if found3:
                        break
                if found3:
                    found = found3
                    break
                continue
            
            # For k >= 4, use backtracking with pruning
            def backtrack(start_idx, current_terms, current_sum):
                if len(current_terms) == k:
                    if current_sum == n:
                        return current_terms[:]
                    return None
                if current_sum > n:
                    return None
                
                rem = n - current_sum
                terms_left = k - len(current_terms)
                
                # Pruning: minimum possible sum of remaining terms
                # The smallest terms we can still pick (in order)
                if terms_left > 0:
                    # We need at least terms_left terms, each >= 1, but more precisely
                    # the smallest available terms from start_idx onward
                    min_possible = 0
                    for idx in range(start_idx, min(start_idx + terms_left, M)):
                        min_possible += smooth[idx]
                    if min_possible > rem:
                        return None
                    # Max possible: we need rem, each term <= rem
                    # Also, the largest term we can pick is rem (if it's smooth and >= current)
                
                for idx in range(start_idx, M):
                    term = smooth[idx]
                    if term > rem:
                        break
                    
                    # Check compatibility with all current terms
                    compatible = True
                    for ci, ct in enumerate(current_terms):
                        if term % ct == 0 or ct % term == 0:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    new_terms = current_terms + [term]
                    result = backtrack(idx + 1, new_terms, current_sum + term)
                    if result is not None:
                        return result
                return None
            
            result = backtrack(0, [], 0)
            if result is not None:
                found = result
                break
        
        if found is None:
            # Should not happen for 1..128, but as a safety net
            # Try a more exhaustive search with higher k
            for k in range(11, 16):
                def backtrack2(start_idx, current_terms, current_sum):
                    if len(current_terms) == k:
                        if current_sum == n:
                            return current_terms[:]
                        return None
                    if current_sum > n:
                        return None
                    
                    rem = n - current_sum
                    terms_left = k - len(current_terms)
                    if terms_left > 0:
                        min_possible = 0
                        for idx in range(start_idx, min(start_idx + terms_left, M)):
                            min_possible += smooth[idx]
                        if min_possible > rem:
                            return None
                    
                    for idx in range(start_idx, M):
                        term = smooth[idx]
                        if term > rem:
                            break
                        compatible = True
                        for ct in current_terms:
                            if term % ct == 0 or ct % term == 0:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        result = backtrack2(idx + 1, current_terms + [term], current_sum + term)
                        if result is not None:
                            return result
                    return None
                
                result = backtrack2(0, [], 0)
                if result is not None:
                    found = result
                    break
        
        if found is None:
            # Last resort: this should never happen
            found = []
        
        answers.append(found)
    
    return answers

# EVOLVE-BLOCK-END

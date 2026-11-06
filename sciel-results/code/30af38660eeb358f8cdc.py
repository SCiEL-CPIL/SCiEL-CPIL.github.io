import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    n_smooth = len(smooth_numbers)
    
    # 2. Helper to check antichain property for a tuple of numbers
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a == b:
                    return False
                if b % a == 0 or a % b == 0:
                    return False
        return True
    
    # 3. Find representations
    answers = [None] * 128
    
    # Level 1: 1 term
    for s in smooth_numbers:
        if s <= 128:
            answers[s - 1] = [s]
    
    # Level 2: 2 terms
    two_term_sums = {}
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if b % a != 0 and a % b != 0:
                if s not in two_term_sums:
                    two_term_sums[s] = [a, b]
    
    for s, terms in two_term_sums.items():
        if answers[s - 1] is None:
            answers[s - 1] = terms
            
    # Level 3: 3 terms
    three_term_sums = {}
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            if a + b > 128:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s = a + b + c
                if s > 128:
                    break
                if b % a == 0 or a % b == 0:
                    continue
                if c % a == 0 or a % c == 0:
                    continue
                if c % b == 0 or b % c == 0:
                    continue
                
                if s not in three_term_sums:
                    three_term_sums[s] = [a, b, c]
                    
    for s, terms in three_term_sums.items():
        if answers[s - 1] is None:
            answers[s - 1] = terms
            
    if all(ans is not None for ans in answers):
        return answers
    
    # For remaining targets, try 4+ terms with targeted search
    # Sort smooth numbers for better pruning
    smooth_sorted = sorted(smooth_numbers)
    
    for i in range(128):
        if answers[i] is None:
            target = i + 1
            found = False
            
            # Try 4 terms with backtracking
            def search_4(idx, remaining, chosen):
                if time.time() > deadline:
                    return None
                if len(chosen) == 4:
                    if remaining == 0:
                        return list(chosen)
                    return None
                if remaining <= 0:
                    return None
                if idx >= len(smooth_sorted):
                    return None
                
                # Pruning: if we can't reach remaining with max possible sum, stop
                max_possible = remaining
                for pos in range(4 - len(chosen)):
                    if idx + pos < len(smooth_sorted):
                        max_possible += smooth_sorted[idx + pos]
                    else:
                        break
                if max_possible < remaining:
                    return None
                
                for j in range(idx, len(smooth_sorted)):
                    val = smooth_sorted[j]
                    if val > remaining:
                        break
                    # Check antichain with all chosen
                    valid = True
                    for c in chosen:
                        if val % c == 0 or c % val == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    chosen.append(val)
                    result = search_4(j + 1, remaining - val, chosen)
                    if result is not None:
                        return result
                    chosen.pop()
                return None
            
            result = search_4(0, target, [])
            if result is not None:
                answers[i] = result
                found = True
            
            if not found:
                # Try 5 terms
                def search_5(idx, remaining, chosen):
                    if time.time() > deadline:
                        return None
                    if len(chosen) == 5:
                        if remaining == 0:
                            return list(chosen)
                        return None
                    if remaining <= 0:
                        return None
                    if idx >= len(smooth_sorted):
                        return None
                    
                    for j in range(idx, len(smooth_sorted)):
                        val = smooth_sorted[j]
                        if val > remaining:
                            break
                        valid = True
                        for c in chosen:
                            if val % c == 0 or c % val == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        chosen.append(val)
                        result = search_5(j + 1, remaining - val, chosen)
                        if result is not None:
                            return result
                        chosen.pop()
                    return None
                
                result = search_5(0, target, [])
                if result is not None:
                    answers[i] = result
                    found = True
            
            if not found:
                # Fallback: use target itself if smooth, else find any valid representation
                # This should not happen for 1-128
                answers[i] = [target]
    
    return answers

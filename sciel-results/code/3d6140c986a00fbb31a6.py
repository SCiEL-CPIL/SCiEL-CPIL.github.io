import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin
    
    # 1. Greedy Initialization
    # Pick largest numbers first if coprime to all previously picked
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)
    
    current_score = get_score(chosen)
    best_set = set(chosen)
    best_score = current_score
    
    # Helper to check if a number is coprime to a set
    def is_coprime_to_set(a, s):
        return all(gcd(a, b) == 1 for b in s)
        
    # Local Search
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # 1. Try adding a single element
        best_add = None
        best_add_gain = 0
        for a in range(1, 512):
            if a not in best_set:
                if is_coprime_to_set(a, best_set):
                    gain = 1.0 / (512 - a)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = a
        
        if best_add is not None:
            best_set.add(best_add)
            current_score += best_add_gain
            if current_score > best_score:
                best_score = current_score
            improved = True
            continue # Restart search with new set
            
        # 2. Try swapping 1 element with 1 element
        best_swap = None
        best_swap_gain = 0
        for a in list(best_set):
            for b in range(1, 512):
                if b not in best_set:
                    # b must be coprime to all in best_set except a
                    rest = best_set - {a}
                    if is_coprime_to_set(b, rest):
                        gain = (1.0 / (512 - b)) - (1.0 / (512 - a))
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (a, b)
        
        if best_swap is not None:
            a, b = best_swap
            best_set.remove(a)
            best_set.add(b)
            current_score += best_swap_gain
            if current_score > best_score:
                best_score = current_score
            improved = True
            continue # Restart
            
        # 3. Try swapping 1 element with 2 elements
        # Remove a, add b, c
        # b, c must be coprime to each other and to rest
        best_1_for_2 = None
        best_1_for_2_gain = 0
        for a in list(best_set):
            rest = best_set - {a}
            # Candidate b, c from numbers not in best_set
            # To optimize, we can iterate b, then c
            # But 511^2 is 260k, times |S| (~100) is 26M. Might be slow in Python.
            # Let's restrict b, c to be > a? Or just all.
            # Given time limit, let's try a limited version or just skip if too slow.
            # Actually, 26M gcd checks might take a few seconds.
            # Let's try it, but with a time check inside the loop if possible, or just rely on the outer loop.
            # To be safe, let's only try b, c that are "large" (high value).
            # Sort candidates by value desc.
            candidates = [x for x in range(1, 512) if x not in best_set]
            candidates.sort(key=lambda x: 1.0/(512-x), reverse=True)
            
            # Only consider top 50 candidates for b and c to keep it fast
            top_candidates = candidates[:50]
            
            for i, b in enumerate(top_candidates):
                if not is_coprime_to_set(b, rest):
                    continue
                for c in top_candidates[i+1:]:
                    if b == c: continue
                    if not is_coprime_to_set(c, rest):
                        continue
                    if not is_coprime_to_set(c, {b}):
                        continue
                    gain = (1.0 / (512 - b) + 1.0 / (512 - c)) - (1.0 / (512 - a))
                    if gain > best_1_for_2_gain:
                        best_1_for_2_gain = gain
                        best_1_for_2 = (a, b, c)
                        
        if best_1_for_2 is not None:
            a, b, c = best_1_for_2
            best_set.remove(a)
            best_set.add(b)
            best_set.add(c)
            current_score += best_1_for_2_gain
            if current_score > best_score:
                best_score = current_score
            improved = True
            continue

        # If no improvement found in any step, break
        break
        
    return list(best_set)

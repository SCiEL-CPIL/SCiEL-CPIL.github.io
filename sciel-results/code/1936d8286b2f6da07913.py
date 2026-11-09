"""B_3[1] set construction: maximize distinct integers in [1,4096] with unique unordered triple sums."""
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def check_valid(selected):
        """Check if all unordered triple sums (with repetition) are unique."""
        sums = set()
        n = len(selected)
        # Triple sums: a<=b<=c from selected
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = selected[i] + selected[j] + selected[k]
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def greedy_with_order(order, start_selected=None):
        """Greedy: try to add elements in given order, maintaining validity."""
        if start_selected is None:
            selected = []
            triple_sums = set()
        else:
            selected = list(start_selected)
            # Build triple sums from current selected
            triple_sums = set()
            n = len(selected)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        triple_sums.add(selected[i] + selected[j] + selected[k])
        
        for x in order:
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 5:
                break
            if x in selected:
                continue
            
            # Check if adding x creates any duplicate triple sum
            n = len(selected)
            valid = True
            # New triple sums involving x:
            # (x, x, x), (x, x, sel[i]), (x, sel[i], sel[j]) for i<=j
            new_sums = set()
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                for i in range(n):
                    s = 2 * x + selected[i]
                    if s in triple_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                for i in range(n):
                    for j in range(i, n):
                        s = x + selected[i] + selected[j]
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    # Strategy 1: Simple ascending greedy
    best = greedy_with_order(range(1, N + 1))
    
    # Strategy 2: Try different starting points and orderings
    # Use a more spread-out order to avoid clustering
    for start in range(1, 100):
        if time.time() > DEADLINE - 10:
            break
        # Order: start, start+1, start+2, ... but skip to spread out
        order = list(range(start, N + 1))
        result = greedy_with_order(order)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Randomized greedy with different seeds
    np.random.seed(42)
    for trial in range(200):
        if time.time() > DEADLINE - 10:
            break
        # Random permutation of a subset
        subset_size = min(N, 2000)
        candidate_pool = np.random.choice(range(1, N + 1), size=subset_size, replace=False)
        np.random.shuffle(candidate_pool)
        result = greedy_with_order(candidate_pool.tolist())
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Local search / improvement
    # Try to swap elements to improve the set
    if len(best) < MAX_SIZE:
        if time.time() < DEADLINE - 15:
            # Try removing one element and adding a better one
            improved = True
            while improved and time.time() < DEADLINE - 10:
                improved = False
                for i in range(len(best)):
                    if time.time() > DEADLINE - 10:
                        break
                    # Try removing best[i] and see if we can add something else
                    without_i = best[:i] + best[i+1:]
                    # Try adding candidates not in without_i
                    for x in range(1, N + 1):
                        if x in without_i:
                            continue
                        # Quick check: can we add x to without_i?
                        n = len(without_i)
                        triple_sums = set()
                        for a in range(n):
                            for b in range(a, n):
                                for c in range(b, n):
                                    triple_sums.add(without_i[a] + without_i[b] + without_i[c])
                        
                        valid = True
                        new_sums = set()
                        s = 3 * x
                        if s in triple_sums:
                            valid = False
                        else:
                            new_sums.add(s)
                        
                        if valid:
                            for a in range(n):
                                s = 2 * x + without_i[a]
                                if s in triple_sums or s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                        
                        if valid:
                            for a in range(n):
                                for b in range(a, n):
                                    s = x + without_i[a] + without_i[b]
                                    if s in triple_sums or s in new_sums:
                                        valid = False
                                        break
                                    new_sums.add(s)
                                if not valid:
                                    break
                        
                        if valid:
                            new_set = without_i + [x]
                            new_set.sort()
                            if new_set != best:
                                best = new_set
                                improved = True
                                break
                    if improved:
                        break
    
    # Final verification
    if not check_valid(best):
        # Fallback to simple greedy
        best = greedy_with_order(range(1, N + 1))
    
    return best

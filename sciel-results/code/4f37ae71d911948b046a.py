"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80
    
    def can_add(x, s):
        """Check if x can be added without creating a 3-term AP."""
        for a in s:
            b = 2 * x - a
            if b != x and b in s:
                return False
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b != a and b != x and b in s:
                    return False
        return True
    
    def is_valid(s):
        s = set(s)
        for a in s:
            for b in s:
                if a < b:
                    c = 2 * b - a
                    if c in s:
                        return False
        return True
    
    def score_of(s):
        return sum(1.0 / x for x in s)
    
    # Greedy initialization with various orders
    best_set = None
    best_score = 0.0
    
    random.seed(12345)
    
    for trial in range(500):
        if trial == 0:
            order = list(range(1, 257))
        elif trial == 1:
            order = list(range(256, 0, -1))
        else:
            order = list(range(1, 257))
            random.shuffle(order)
        
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        
        if len(s) > 0:
            sc = score_of(s)
            if sc > best_score:
                best_score = sc
                best_set = s.copy()
    
    if best_set is None:
        return [1]
    
    # Local search
    current_set = best_set.copy()
    current_score = best_score
    recip = {x: 1.0 / x for x in range(1, 257)}
    
    # Phase 1: Try to add elements
    for _ in range(10000):
        if time.time() > deadline - 10:
            break
        candidates = [x for x in range(1, 257) if x not in current_set]
        if not candidates:
            break
        random.shuffle(candidates)
        added = False
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
                current_score += recip[x]
                added = True
                break
        if not added:
            break
    
    # Phase 2: Try swaps to improve
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        improved = False
        elems = list(current_set)
        random.shuffle(elems)
        for x in elems:
            current_set.remove(x)
            base = current_score - recip[x]
            for y in range(1, 257):
                if y in current_set:
                    continue
                if can_add(y, current_set):
                    ns = base + recip[y]
                    if ns > current_score:
                        current_set.add(y)
                        current_score = ns
                        improved = True
                        break
            if not improved:
                current_set.add(x)
                current_score = base + recip[x]
            else:
                break
    
    if not is_valid(current_set):
        return sorted(best_set)
    
    return sorted(current_set)
# EVOLVE-BLOCK-END

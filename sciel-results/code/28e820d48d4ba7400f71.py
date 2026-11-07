import random
import time

def construct():
    n = 12
    m = 40
    start_time = time.time()
    deadline = start_time + 85
    
    all_masks = list(range(1 << n))
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                a, b = family[i], family[j]
                if (a | b) == b and a != b:
                    count += 1
                elif (a | b) == a and b != a:
                    count += 1
        return count
    
    # Greedy initialization
    family = [0, (1 << n) - 1]
    remaining = [mask for mask in all_masks if mask not in family]
    
    while len(family) < m and time.time() < deadline:
        best_gain = -1
        best_idx = -1
        for idx, mask in enumerate(remaining):
            gain = 0
            for f in family:
                if (mask | f) == f and mask != f:
                    gain += 1
                elif (mask | f) == mask and f != mask:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx >= 0:
            family.append(remaining.pop(best_idx))
    
    while len(family) < m:
        for mask in all_masks:
            if mask not in family:
                family.append(mask)
                break
    
    best_family = family[:]
    best_score = count_comparable(best_family)
    
    # Local search
    random.seed(42)
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, m - 1)
            other = random.choice(all_masks)
            if other in best_family:
                continue
            new_family = best_family[:]
            new_family[i] = other
            new_score = count_comparable(new_family)
            if new_score > best_score:
                best_score = new_score
                best_family = new_family
                improved = True
        if not improved:
            break
    
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result

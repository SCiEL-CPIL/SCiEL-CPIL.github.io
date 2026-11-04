import time
import random
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # 1. Exhaustive search for small k
    # Try to find best set of size k in range [0, M]
    # k=3, M=10
    for k in range(3, 7):
        max_m = 10 + k * 5
        # Sample random subsets if too many
        # For k=3, C(13,3) is small.
        # For k=6, C(40,6) is large. Use random sampling.
        num_samples = 5000
        for _ in range(num_samples):
            if time.time() > deadline - 10:
                break
            # Random set of size k
            # Ensure 0 is in U
            # Pick k-1 from [1, max_m]
            try:
                U = [0] + random.sample(range(1, max_m + 1), k - 1)
            except ValueError:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # 2. Local Search
    current = best_set[:]
    current_score = best_score
    
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False

    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        moved = False
        
        # 1. Add element
        if len(current) < 15: # Limit size to keep D small?
            max_val = current[-1]
            # Try adding small values or values near existing
            candidates = []
            for x in current:
                candidates.extend([x+1, x-1, x*2, x+random.randint(1, 10)])
            candidates.append(max_val + random.randint(1, 20))
            candidates.append(random.randint(1, max_val + 10))
            
            for new_val in random.sample(candidates, min(10, len(candidates))):
                if new_val > 0 and new_val not in current:
                    if try_move(current + [new_val]):
                        moved = True
                        break
        if moved: continue
        
        # 2. Remove element
        if len(current) > 3:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                continue
                
        # 3. Shift element
        i = random.randint(1, len(current) - 1)
        old_val = current[i]
        for delta in random.sample([-5, -3, -2, -1, 1, 2, 3, 5, 10], 3):
            new_val = old_val + delta
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                if try_move(new_set):
                    moved = True
                    break
        if moved: continue
        
        # 4. Restart if stuck (simple heuristic)
        if iterations % 1000 == 0:
             # Try a random restart from best
             current = best_set[:]
             current_score = best_score

    return best_set

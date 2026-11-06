import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0
    
    N = 256
    
    # Precompute conflict matrix: conflict[a][b] = True if (a+b) divides a*b
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    def can_add(s, num):
        for x in s:
            if conflict[num][x]:
                return False
        return True
    
    def local_search(initial):
        best = set(initial)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try to add any number
            for num in range(1, N + 1):
                if num not in best and can_add(best, num):
                    best.add(num)
                    improved = True
                    break
            if time.time() > DEADLINE:
                break
            # Try swap: remove one blocker, add the number
            if not improved:
                for num in range(1, N + 1):
                    if num in best:
                        continue
                    blockers = [x for x in best if conflict[num][x]]
                    if len(blockers) == 1:
                        rem = blockers[0]
                        best.discard(rem)
                        best.add(num)
                        improved = True
                        break
                    if time.time() > DEADLINE:
                        break
        return best
    
    # Strategy 1: all odds, then greedily add compatible evens
    s1 = set(range(1, N + 1, 2))
    for num in range(2, N + 1, 2):
        if can_add(s1, num):
            s1.add(num)
    best = local_search(s1)
    best_size = len(best)
    
    # Strategy 2: random restarts with local search
    rng = random.Random(42)
    while time.time() < DEADLINE - 5:
        init_size = rng.randint(100, 150)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        init = set(candidates[:init_size])
        # Clean: keep only a valid subset
        cleaned = set()
        for num in sorted(init):
            if can_add(cleaned, num):
                cleaned.add(num)
        result = local_search(cleaned)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return sorted(best)

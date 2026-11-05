import random
import itertools
from collections import Counter

def construct():
    def count_collisions(s):
        """Count the number of duplicate unordered pair sums (including diagonals) in set s."""
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        c = Counter(sums)
        return sum(v - 1 for v in c.values() if v > 1)
    
    def largest_sidon_size(s):
        """Find the largest Sidon subset size in s."""
        s = sorted(s)
        n = len(s)
        best = 1
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(s, size):
                subset = list(subset)
                sums = set()
                valid = True
                for i in range(len(subset)):
                    for j in range(i, len(subset)):
                        sm = subset[i] + subset[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                    if not valid:
                        break
                if valid:
                    best = size
                    found = True
                    break
            if found:
                break
        return best

    def optimize(s):
        """Stochastic Hill Climbing to maximize collisions."""
        s = list(s)
        current_collisions = count_collisions(s)
        
        # Run for a fixed number of steps
        for step in range(2000):
            i = random.randint(0, 11)
            old_val = s[i]
            
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            if new_val in s:
                continue
                
            s[i] = new_val
            new_collisions = count_collisions(s)
            
            if new_collisions > current_collisions:
                current_collisions = new_collisions
            else:
                s[i] = old_val
        
        return s

    random.seed(42)
    best_set = None
    best_collision = -1
    
    # Run multiple random trials
    for trial in range(150):
        s = random.sample(range(129), 12)
        s = optimize(s)
        col = count_collisions(s)
        if col > best_collision:
            best_collision = col
            best_set = s
    
    # Also try structured initializations
    structured_starts = [
        list(range(12)),
        list(range(0, 129, 11))[:12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 124],
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128],
    ]
    for start in structured_starts:
        if len(set(start)) == 12 and all(0 <= x <= 128 for x in start):
            s = optimize(list(start))
            col = count_collisions(s)
            if col > best_collision:
                best_collision = col
                best_set = s
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return list(best_set)

import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 50.0 # 10s margin
        
        def can_add(x, selected, sums):
            # Check sums involving x
            # 3x
            if 3*x in sums:
                return False
            # 2x + s
            for s in selected:
                if 2*x + s in sums:
                    return False
            # x + s + t
            # We need to check all pairs (s, t) from selected
            # To be efficient, we can iterate
            n = len(selected)
            for i in range(n):
                s = selected[i]
                for j in range(i, n):
                    t = selected[j]
                    if x + s + t in sums:
                        return False
            return True

        def add_x(x, selected, sums):
            selected.append(x)
            sums.add(3*x)
            for s in selected[:-1]:
                sums.add(2*x + s)
            n = len(selected)
            # selected[-1] is x
            for i in range(n-1):
                s = selected[i]
                for j in range(i, n-1):
                    t = selected[j]
                    sums.add(x + s + t)

        # Standard Greedy
        selected = []
        sums = set()
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if can_add(x, selected, sums):
                add_x(x, selected, sums)
        
        best_set = selected[:]
        best_size = len(selected)

        # Try a few randomized restarts if time permits
        # Strategy: Pick a random starting point, then greedily add?
        # Or: Perturb the greedy set?
        
        # A common trick for B-sets: The greedy algorithm is deterministic.
        # We can try to "swap" elements in the found set to see if we can fit more.
        # But that's complex.
        
        # Let's try running greedy with a random "skip" probability? No.
        
        # Let's just return the best found so far.
        return best_set

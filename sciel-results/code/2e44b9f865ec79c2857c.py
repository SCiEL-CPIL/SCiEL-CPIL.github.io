import time
import random

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    def get_diffs_list(s):
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]

    def is_valid(s):
        diffs_list = get_diffs_list(s)
        diffs_set = set(diffs_list)
        if len(diffs_set) != len(diffs_list):
            return False
        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs_set:
                return False
        return True

    # Step 1: Greedy construction
    a = [0]
    for d in range(1, 31):
        if time.time() > deadline:
            break
        if d in set(get_diffs_list(a)):
            continue
        found = False
        # Try to add x such that x - v = d
        for v in a:
            x = v + d
            if x not in a:
                trial = sorted(a + [x])
                if is_valid(trial):
                    a = trial
                    found = True
                    break
        if not found:
            # Try x such that v - x = d
            for v in a:
                x = v - d
                if x >= 0 and x not in a:
                    trial = sorted(a + [x])
                    if is_valid(trial):
                        a = trial
                        found = True
                        break
        if not found:
            # Fallback: try random additions
            for _ in range(100):
                x = random.randint(1, 1000)
                if x not in a:
                    trial = sorted(a + [x])
                    if is_valid(trial):
                        a = trial
                        found = True
                        break
            if not found:
                # Should not happen, but if it does, we have a problem.
                # We can try to start over with a random set.
                pass

    # If greedy failed completely (unlikely), do a random search
    if not is_valid(a):
        for _ in range(10000):
            if time.time() > deadline:
                break
            k = random.randint(10, 15)
            s = sorted([0] + random.sample(range(1, 200), k-1))
            if is_valid(s):
                a = s
                break

    # Step 2: Shrink the set to minimize max element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        a = sorted(a)
        
        # 1. Try to remove elements (starting from largest)
        for i in range(len(a) - 1, 0, -1):
            trial = a[:i] + a[i+1:]
            if is_valid(trial):
                a = trial
                improved = True
                break
        
        # 2. Try to replace the largest element with a smaller one
        if len(a) > 1:
            largest_idx = len(a) - 1
            largest_val = a[largest_idx]
            # Try values from largest_val - 1 down to 1
            # Optimization: if largest_val is large, we can skip some ranges?
            # No, just scan down.
            for new_val in range(largest_val - 1, 0, -1):
                if new_val in a:
                    continue
                trial = a[:largest_idx] + a[largest_idx+1:] + [new_val]
                trial.sort()
                if is_valid(trial):
                    a = trial
                    improved = True
                    break
        
        # 3. Try to shift any non-zero element down by 1
        if not improved:
            for i in range(len(a) - 1, 0, -1):
                if a[i] > 1:
                    new_val = a[i] - 1
                    if new_val not in a:
                        trial = list(a)
                        trial[i] = new_val
                        trial.sort()
                        if is_valid(trial):
                            a = trial
                            improved = True
                            break

    return a

import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s
    
    def all_products(seq):
        """Compute all subarray products of seq. Return set and count."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return None  # Conflict
                prods.add(p)
        return prods
    
    def is_valid(seq):
        return all_products(seq) is not None
    
    # Greedy initialization
    seq = []
    for x in range(2, 129):
        candidate = seq + [x]
        if is_valid(candidate):
            seq = candidate
    
    best_seq = list(seq)
    best_len = len(seq)
    
    # Try to improve: iteratively add excluded numbers
    excluded = [x for x in range(2, 129) if x not in set(seq)]
    
    # Fixed point: try adding each excluded number
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        excluded = [x for x in range(2, 129) if x not in set(seq)]
        for x in excluded:
            # Find position
            pos = 0
            for i in range(len(seq)):
                if seq[i] < x:
                    pos = i + 1
                else:
                    break
            candidate = seq[:pos] + [x] + seq[pos:]
            if is_valid(candidate):
                seq = candidate
                changed = True
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = list(seq)
                break  # Restart with new excluded set
    
    # Local search: try removing one element and adding others
    # Try all single removals, then see if we can add more
    for _ in range(100):  # Limit iterations
        if time.time() > DEADLINE:
            break
        current_set = set(seq)
        for rem in list(seq):
            if time.time() > DEADLINE:
                break
            trial = [x for x in seq if x != rem]
            # Try to add excluded numbers to trial
            trial_changed = True
            while trial_changed and time.time() < DEADLINE:
                trial_changed = False
                trial_set = set(trial)
                for x in range(2, 129):
                    if x in trial_set:
                        continue
                    pos = 0
                    for i in range(len(trial)):
                        if trial[i] < x:
                            pos = i + 1
                        else:
                            break
                    candidate = trial[:pos] + [x] + trial[pos:]
                    if is_valid(candidate):
                        trial = candidate
                        trial_set = set(trial)
                        trial_changed = True
                        break
            if len(trial) > best_len:
                best_len = len(trial)
                best_seq = list(trial)
    
    return best_seq

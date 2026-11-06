"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    TARGETS = frozenset(range(1, 31))
    
    def get_diffs(s):
        diffs = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        d = get_diffs(s)
        return len(d) == len(set(d))
    
    def covered(s):
        return set(get_diffs(s)) & TARGETS
    
    def energy(s):
        d = get_diffs(s)
        collisions = len(d) - len(set(d))
        missing = len(TARGETS - set(d))
        max_elem = max(s) if s else 0
        return max_elem + 10000.0 * (missing + collisions)
    
    def valid(s):
        if not s or 0 not in s:
            return False
        if len(s) > 32 or max(s) > 10**6:
            return False
        d = get_diffs(s)
        if len(d) != len(set(d)):
            return False
        if not TARGETS.issubset(set(d)):
            return False
        return True
    
    def greedy_construct():
        a = [0]
        for d in range(1, 31):
            if d in set(get_diffs(a)):
                continue
            found = False
            for x in sorted(a):
                for cand in (x + d, x - d):
                    if cand < 0 or cand in a:
                        continue
                    trial = sorted(a + [cand])
                    if is_sidon(trial) and d in set(get_diffs(trial)):
                        a = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        break
                    y += 1
        return a
    
    def local_search(initial, time_limit):
        best = list(initial)
        best_e = energy(best)
        current = list(best)
        current_e = best_e
        T = 2000.0
        T_min = 1.0
        cooling = 0.9997
        
        while time.time() < time_limit:
            T = max(T_min, T * cooling)
            r = random.random()
            
            if r < 0.45:
                # Replace element with smaller
                if len(current) <= 1:
                    continue
                idx = random.randint(1, len(current) - 1)
                old = current[idx]
                if old <= 1:
                    continue
                new_val = random.randint(1, old - 1)
                if new_val in current:
                    continue
                trial = current[:idx] + [new_val] + current[idx+1:]
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                te = energy(trial)
                if te < current_e or random.random() < (2.718281828 ** (-(te - current_e) / max(T, 1e-10))):
                    current = trial
                    current_e = te
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif r < 0.65:
                # Remove element
                if len(current) <= 8:
                    continue
                idx = random.randint(1, len(current) - 1)
                trial = current[:idx] + current[idx+1:]
                trial = sorted(trial)
                if len(trial) < 2:
                    continue
                te = energy(trial)
                if te < current_e or random.random() < (2.718281828 ** (-(te - current_e) / max(T, 1e-10))):
                    current = trial
                    current_e = te
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif r < 0.80:
                # Add small element
                if len(current) >= 32:
                    continue
                mx = max(current)
                new_val = random.randint(1, mx + 100)
                if new_val in current:
                    continue
                trial = sorted(current + [new_val])
                if len(trial) > 32:
                    continue
                te = energy(trial)
                if te < current_e or random.random() < (2.718281828 ** (-(te - current_e) / max(T, 1e-10))):
                    current = trial
                    current_e = te
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif r < 0.90:
                # Perturbation: move element to random position
                if len(current) <= 1:
                    continue
                idx = random.randint(1, len(current) - 1)
                old = current[idx]
                new_val = random.randint(1, max(old + 200, 300))
                if new_val in current:
                    continue
                trial = current[:idx] + [new_val] + current[idx+1:]
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                te = energy(trial)
                if te < current_e or random.random() < (2.718281828 ** (-(te - current_e) / max(T, 1e-10))):
                    current = trial
                    current_e = te
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            else:
                # Compact: shift all non-zero down by 1
                if len(current) <= 2:
                    continue
                trial = [0] + [x - 1 for x in current if x > 0]
                trial = sorted(set(trial))
                if len(trial) != len(current) or min(trial) < 0:
                    continue
                te = energy(trial)
                if te < current_e or random.random() < (2.718281828 ** (-(te - current_e) / max(T, 1e-10))):
                    current = trial
                    current_e = te
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
        
        return best
    
    random.seed(12345)
    
    initial = greedy_construct()
    if not valid(initial):
        for _ in range(20):
            initial = greedy_construct()
            if valid(initial):
                break
    
    time_limit = deadline - 5.0
    result = local_search(initial, time_limit)
    
    if not valid(result):
        result = initial
    
    result = sorted(set(result))
    if 0 not in result:
        result = sorted([0] + result)
    
    return result
# EVOLVE-BLOCK-END

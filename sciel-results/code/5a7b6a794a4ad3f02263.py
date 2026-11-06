# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _positive_diffs(s):
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]

def _is_sidon(s):
    d = _positive_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    d = set(_positive_diffs(s))
    return all(x in d for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _greedy_construct():
    a = [0]
    covered = set()
    for _ in range(31):
        if time.time() > _DEADLINE:
            break
        missing = [d for d in range(1, 31) if d not in covered]
        if not missing:
            break
        
        candidates = set()
        for v in a:
            for d in missing:
                x1 = v + d
                x2 = v - d
                if x1 > 0:
                    candidates.add(x1)
                if x2 > 0:
                    candidates.add(x2)
        for i in range(1, 10):
            candidates.add(i)
            
        best_x = None
        best_gain = -1
        best_trial = None
        best_new_covered = None
        
        for x in sorted(candidates):
            if x in a:
                continue
            trial = sorted(a + [x])
            if not _is_sidon(trial):
                continue
            new_diffs = set(_positive_diffs(trial))
            new_covered = covered.union(new_diffs.intersection(range(1, 31)))
            gain = len(new_covered - covered)
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_trial = trial
                best_new_covered = new_covered
                
        if best_x is not None:
            a = best_trial
            covered = best_new_covered
        else:
            break
    return a

def _dfs_csp(bound):
    def dfs(S, covered):
        if time.time() > _DEADLINE:
            return None
            
        if all(d in covered for d in range(1, 31)):
            return S[:]
            
        if len(S) == 32:
            return None
            
        slots = 32 - len(S)
        max_new_diffs = 0
        current_len = len(S)
        for i in range(slots):
            max_new_diffs += current_len + i
        missing_count = 30 - len(covered)
        if max_new_diffs < missing_count:
            return None

        missing_diffs = [d for d in range(1, 31) if d not in covered]
        candidates = set()
        for v in S:
            for d in missing_diffs:
                x1 = v + d
                x2 = v - d
                if 0 < x1 <= bound:
                    candidates.add(x1)
                if 0 < x2 <= bound:
                    candidates.add(x2)
        
        for x in sorted(candidates):
            if x in S:
                continue
            if x > bound:
                continue
                
            trial = S + [x]
            trial.sort()
            
            if not _is_sidon(trial):
                continue
                
            new_diffs = set(_positive_diffs(trial))
            new_covered = covered.union(new_diffs.intersection(range(1, 31)))
            
            if len(new_covered) == len(covered):
                continue
                
            res = dfs(trial, new_covered)
            if res is not None:
                return res
                
        return None

    return dfs([0], set())

def construct():
    t0 = time.time()
    
    g = _greedy_construct()
    
    if not _valid(g):
        rng = random.Random(42)
        found = False
        for _ in range(1000):
            if time.time() > _DEADLINE - 5:
                break
            s = [0]
            for _ in range(10):
                s.append(rng.randint(1, 100))
            s = sorted(set(s))
            if _valid(s):
                g = s
                found = True
                break
        if not found:
            return g # Return best effort, might be invalid, but better than crash? No, score 0.
                     # But we must return something.

    best_set = g
    best_val = max(g)
    
    current_bound = best_val - 1
    
    while current_bound >= 1 and time.time() < _DEADLINE - 10:
        sol = _dfs_csp(current_bound)
        if sol is not None:
            best_set = sol
            best_val = max(sol)
            if best_val < current_bound:
                current_bound = best_val - 1
            else:
                current_bound -= 1
        else:
            current_bound -= 1
            
    return best_set
# EVOLVE-BLOCK-END

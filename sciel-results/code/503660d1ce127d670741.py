import random
import time

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    def get_diffs(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_valid(S):
        n = len(S)
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return all(d in diffs for d in range(1, 31))

    def local_search(S, max_iter=2000):
        """Try to reduce max element via swaps and removals."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        n = len(S)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 10:
                break
            
            improved = False
            
            # Try removing an element if we still cover all diffs
            if len(S) > 1:
                for i in range(len(S)):
                    trial = S[:i] + S[i+1:]
                    if covers_all(trial):
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                            break
                if improved:
                    continue
            
            # Try swapping an element with a smaller value
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                # Try replacing S[i] with a smaller value
                for new_val in range(S[i] - 1, 0, -1):
                    if new_val in S:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial.sort()
                    if is_valid(trial) and covers_all(trial):
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best

    best_set = None
    best_max = float('inf')

    # Known good set as fallback (a valid Sidon set covering 1..30)
    # We'll find one during search, but have a safe fallback
    fallback_set = None

    def greedy_construct():
        """Fast greedy construction to get a valid set quickly."""
        S = [0]
        covered = set()
        
        for _ in range(50):
            if covered.issuperset(set(range(1, 31))):
                break
            missing = set(range(1, 31)) - covered
            
            # Try all missing diffs, pick the one that allows smallest addition
            best_c = None
            best_max_val = float('inf')
            
            for d in missing:
                for x in S:
                    for sign in [1, -1]:
                        c = x + sign * d
                        if c < 0 or c in S:
                            continue
                        new_S = sorted(S + [c])
                        if is_valid(new_S):
                            if max(new_S) < best_max_val:
                                best_max_val = max(new_S)
                                best_c = c
            
            if best_c is None:
                # Add a large element to realize some missing diff
                d = max(missing)
                c = max(S) + d + 1
                while True:
                    new_S = sorted(S + [c])
                    if is_valid(new_S):
                        break
                    c += 1
                best_c = c
            
            S.append(best_c)
            S.sort()
            covered = get_diffs(S)
        
        return S if covers_all(S) else None

    # First, get a quick valid solution
    quick_sol = greedy_construct()
    if quick_sol:
        best_set = quick_sol[:]
        best_max = max(quick_sol)
        fallback_set = quick_sol[:]

    # Now do iterative improvement
    for restart in range(500):
        if time.time() - start_time > deadline:
            break
        
        S = [0]
        covered = set()
        max_iter = 40
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline:
                break
            if covered.issuperset(set(range(1, 31))):
                break
            
            missing = set(range(1, 31)) - covered
            
            # Generate candidates from all pairs and missing diffs
            candidates = set()
            for d in missing:
                for x in S:
                    candidates.add(x + d)
                    candidates.add(x - d)
            
            # Also try values between existing elements
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        mid = (S[i] + S[j]) // 2
                        if mid > 0:
                            candidates.add(mid)
            
            valid_candidates = []
            for c in candidates:
                if c < 0 or c in S:
                    continue
                new_S = sorted(S + [c])
                if is_valid(new_S):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                # Fallback: add a large element
                d = max(missing) if missing else 1
                c = max(S) + d + 1
                while True:
                    new_S = sorted(S + [c])
                    if is_valid(new_S):
                        break
                    c += 1
                valid_candidates = [c]
            
            # Pick candidate that minimizes max element
            c = min(valid_candidates, key=lambda x: max(S + [x]))
            S.append(c)
            S.sort()
            covered = get_diffs(S)
        
        if covers_all(S):
            max_elem = max(S)
            if max_elem < best_max:
                # Try local search to improve
                improved = local_search(S)
                if max(improved) < best_max:
                    best_max = max(improved)
                    best_set = improved
                elif max_elem < best_max:
                    best_max = max_elem
                    best_set = S[:]

    if best_set:
        return best_set
    elif fallback_set:
        return fallback_set
    else:
        # Last resort: simple greedy
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

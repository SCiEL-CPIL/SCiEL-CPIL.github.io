import random
import time

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    def get_diffs(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
    
    def is_valid(S):
        n = len(S)
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all(S):
        diffs = get_diffs(S)
        return all(d in diffs for d in range(1, 31))
    
    def local_search(S, max_iter=5000):
        """Try to reduce max element via swaps and removals."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            
            # Try removing an element if we still cover all diffs
            if len(S) > 2:
                for i in range(len(S)):
                    if S[i] == 0:
                        continue
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
            for i in range(1, len(S)):
                # Only try values below the current element to reduce max
                for new_val in range(1, S[i]):
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
    
    # Strategy 1: CSP-like backtracking with forward checking
    def csp_backtrack():
        """Backtracking search with constraint propagation."""
        S = [0]
        covered = set()
        used_diffs = set()
        
        def backtrack(depth=0):
            nonlocal S, covered, used_diffs, best_max, best_set
            
            if time.time() - start_time > deadline - 5:
                return False
            
            if covered.issuperset(set(range(1, 31))):
                if max(S) < best_max:
                    best_max = max(S)
                    best_set = S[:]
                return True
            
            if len(S) > 32:
                return False
            
            missing = set(range(1, 31)) - covered
            
            # Generate candidates: for each missing diff, try to add an element
            candidates = []
            for d in missing:
                for x in S:
                    c1 = x + d
                    c2 = x - d
                    for c in [c1, c2]:
                        if c >= 0 and c not in S:
                            # Check if adding c would violate Sidon condition
                            valid = True
                            for y in S:
                                diff = abs(c - y)
                                if diff in used_diffs and diff != 0:
                                    valid = False
                                    break
                            if valid:
                                candidates.append((c, d))
            
            # Sort by resulting max to prefer smaller solutions
            candidates.sort(key=lambda x: max(x[0], max(S)))
            
            # Deduplicate candidates
            seen = set()
            unique_candidates = []
            for c, d in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append((c, d))
            
            for c, d in unique_candidates:
                # Forward check: see if this addition is promising
                new_S = S + [c]
                new_S.sort()
                
                if not is_valid(new_S):
                    continue
                
                new_diffs = get_diffs(new_S)
                new_covered = new_diffs & set(range(1, 31))
                
                # Check if we're making progress
                if len(new_covered) <= len(covered):
                    continue
                
                # Recurse
                old_S = S[:]
                old_covered = covered.copy()
                old_used = used_diffs.copy()
                
                S = new_S
                covered = new_covered
                used_diffs = new_diffs
                
                if backtrack(depth + 1):
                    return True
                
                # Backtrack
                S = old_S
                covered = old_covered
                used_diffs = old_used
            
            return False
        
        backtrack()
        return best_set
    
    # Strategy 2: Greedy with multiple restarts
    def greedy_build():
        S = [0]
        covered = set()
        
        for _ in range(40):
            if covered.issuperset(set(range(1, 31))):
                break
            
            missing = set(range(1, 31)) - covered
            
            # Generate candidates
            candidates = []
            for d in missing:
                for x in S:
                    c1 = x + d
                    c2 = x - d
                    for c in [c1, c2]:
                        if c >= 0 and c not in S:
                            candidates.append(c)
            
            if not candidates:
                # Add a larger element
                base = max(S) + 1
                for offset in range(1, 100):
                    c = base + offset
                    new_S = S + [c]
                    new_S.sort()
                    if is_valid(new_S):
                        candidates.append(c)
                        break
            
            # Try candidates sorted by resulting max
            random.shuffle(candidates)
            candidates.sort(key=lambda c: max(c, max(S)))
            
            for c in candidates:
                new_S = S + [c]
                new_S.sort()
                if is_valid(new_S):
                    S = new_S
                    covered = get_diffs(S)
                    break
        
        if covered.issuperset(set(range(1, 31))):
            return S
        return None
    
    # Strategy 3: Simulated annealing-like local search
    def simulated_annealing(initial_set):
        S = sorted(initial_set)
        current_max = max(S)
        best = S[:]
        best_max = current_max
        temp = 100.0
        cooling = 0.95
        
        for _ in range(20000):
            if time.time() - start_time > deadline - 5:
                break
            
            if temp < 0.1:
                break
            
            # Random perturbation: swap or replace an element
            i = random.randint(1, len(S) - 1)
            new_val = random.randint(1, current_max + 20)
            
            if new_val in S and new_val != S[i]:
                continue
            
            trial = S[:i] + [new_val] + S[i+1:]
            trial.sort()
            
            if not is_valid(trial):
                temp *= cooling
                continue
            
            trial_max = max(trial)
            delta = trial_max - current_max
            
            # Accept if better or with probability exp(-delta/temp)
            if delta < 0 or random.random() < pow(2.71828, -delta / max(temp, 0.1)):
                S = trial
                current_max = trial_max
                
                if covers_all(S) and current_max < best_max:
                    best = S[:]
                    best_max = current_max
            
            temp *= cooling
        
        return best if covers_all(best) else None
    
    # Run CSP backtracking first
    result = csp_backtrack()
    if result:
        best_set = result
        best_max = max(result)
    
    # If CSP didn't find anything, try greedy
    if best_set is None:
        for restart in range(200):
            if time.time() - start_time > deadline:
                break
            
            random.seed(restart * 104729)
            result = greedy_build()
            if result:
                max_elem = max(result)
                if max_elem < best_max:
                    best_max = max_elem
                    best_set = result[:]
        
        if best_set is None:
            # Fallback: deterministic greedy
            a = [0]
            def diffs(s):
                return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
            for d in range(1, 31):
                if d in diffs(sorted(a)):
                    continue
                found = False
                for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                    trial = sorted(a + [x])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        found = True
                        break
                if not found:
                    y = 2 * max(a) + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if len(set(diffs(trial))) == len(diffs(trial)):
                            a = trial
                            break
                        y += 1
            return a
    
    # Local search to improve
    improved = local_search(best_set[:])
    if improved and max(improved) < best_max:
        best_set = improved
        best_max = max(improved)
    
    # Simulated annealing for further improvement
    if time.time() - start_time < deadline - 10:
        sa_result = simulated_annealing(best_set[:])
        if sa_result and max(sa_result) < best_max:
            best_set = sa_result
            best_max = max(sa_result)
    
    # Final local search
    if time.time() - start_time < deadline - 5:
        improved = local_search(best_set[:])
        if improved and max(improved) < best_max:
            best_set = improved
            best_max = max(improved)
    
    return best_set

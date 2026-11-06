import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    
    def get_diffs(s):
        """Return set of positive differences."""
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if set has all distinct positive differences."""
        diffs = list()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        return len(set(diffs)) == len(diffs)
    
    def covers_1_to_30(s):
        """Check if all differences 1..30 are present."""
        diffs = get_diffs(s)
        return all(d in diffs for d in range(1, 31))
    
    def diameter(s):
        return max(s) - min(s)
    
    # Strategy: Greedy construction followed by local search to tighten the set.
    
    # Phase 1: Greedy construction
    S = [0]
    needed = set(range(1, 31))
    used_diffs = set()
    
    while needed and len(S) < 32:
        best_candidate = None
        best_score = -1
        
        # Generate candidates
        candidates = set()
        for x in S:
            for d in needed:
                c = x + d
                if 0 <= c <= 1000000 and c not in S:
                    candidates.add(c)
        
        # Score each candidate
        for c in candidates:
            # Check Sidon property
            new_diffs = set()
            valid = True
            for x in S:
                d = abs(c - x)
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            
            covered = len(new_diffs & needed)
            # Score: prefer covering more needed diffs, then smaller values
            score = covered * 10000 - c
            if score > best_score:
                best_score = score
                best_candidate = c
        
        if best_candidate is None:
            break
        
        S.append(best_candidate)
        S.sort()
        for x in S[:-1]:
            d = abs(best_candidate - x)
            used_diffs.add(d)
        needed -= used_diffs
    
    # If greedy didn't cover all, try adding more
    while needed and len(S) < 32:
        found = False
        for c in range(1, 10000):
            if c in S:
                continue
            new_diffs = set()
            valid = True
            for x in S:
                d = abs(c - x)
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            if new_diffs & needed:
                S.append(c)
                S.sort()
                for x in S[:-1]:
                    used_diffs.add(abs(c - x))
                needed -= used_diffs
                found = True
                break
        if not found:
            break
    
    if not needed:
        best_set = S[:]
        best_diam = max(S)
    else:
        best_set = None
        best_diam = float('inf')
    
    # Phase 2: Local search to tighten the set
    # Try to replace large elements with smaller ones while preserving properties
    
    def try_local_search(S, max_iterations=5000):
        S = S[:]
        S.sort()
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try to replace each element with a smaller one
            for i in range(len(S) - 1, 0, -1):  # Start from largest
                old_val = S[i]
                if old_val == 0:
                    continue
                
                # Try replacing with smaller values
                for new_val in range(0, old_val):
                    if new_val in S:
                        continue
                    
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    
                    if not is_sidon(trial):
                        continue
                    if not covers_1_to_30(trial):
                        continue
                    if len(trial) > 32:
                        continue
                    
                    # Accept if it reduces diameter or same diameter with smaller max
                    if max(trial) < max(S):
                        S = trial
                        improved = True
                        break
        
        return S
    
    if best_set is not None:
        tightened = try_local_search(best_set)
        if max(tightened) < best_diam:
            best_set = tightened
            best_diam = max(tightened)
    
    # Phase 3: Try multiple random restarts with different greedy orderings
    if time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            S = [0]
            needed = set(range(1, 31))
            used_diffs = set()
            
            while needed and len(S) < 32:
                candidates = set()
                for x in S:
                    for d in needed:
                        c = x + d
                        if 0 <= c <= 1000000 and c not in S:
                            candidates.add(c)
                
                best_candidate = None
                best_score = -1
                
                # Add some randomness
                for c in list(candidates):
                    new_diffs = set()
                    valid = True
                    for x in S:
                        d = abs(c - x)
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    
                    covered = len(new_diffs & needed)
                    score = covered * 10000 - c + random.randint(-500, 500)
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                
                if best_candidate is None:
                    break
                
                S.append(best_candidate)
                S.sort()
                for x in S[:-1]:
                    used_diffs.add(abs(best_candidate - x))
                needed -= used_diffs
            
            if not needed and len(S) <= 32:
                tightened = try_local_search(S, max_iterations=2000)
                if max(tightened) < best_diam:
                    best_set = tightened
                    best_diam = max(tightened)
    
    if best_set is not None:
        return best_set
    
    # Final fallback: construct a valid set
    a = [0]
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in range(1, 10000):
            if x in a:
                continue
            trial = a + [x]
            trial.sort()
            if is_sidon(trial):
                new_diffs = get_diffs(trial)
                if d in new_diffs:
                    a = trial
                    found = True
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

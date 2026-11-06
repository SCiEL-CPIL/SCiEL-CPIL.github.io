import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    target_diffs = set(range(1, 31))
    
    def check_sidon_and_cover(S):
        """Check if S is a valid Sidon set covering all differences 1..30."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False, None
                diffs.add(d)
        if not target_diffs.issubset(diffs):
            return False, None
        return True, max(S)
    
    def greedy_init():
        """Create an initial Sidon set using greedy approach."""
        S = [0]
        covered = set()
        missing = set(target_diffs)
        
        while missing:
            if time.time() > deadline - 5:
                break
            best_y = None
            best_gain = -1
            
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d >= 0:
                        candidates.add(s - d)
            
            for y in sorted(candidates):
                if y in S or y > 500:
                    continue
                new_S = sorted(S + [y])
                new_diffs = set()
                valid = True
                for i in range(len(new_S)):
                    for j in range(i+1, len(new_S)):
                        d = new_S[j] - new_S[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    continue
                new_missing = missing - new_diffs
                gain = len(missing) - len(new_missing)
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
                    if gain == len(missing):
                        break
            
            if best_y is not None:
                S.append(best_y)
                S.sort()
                covered = set()
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        covered.add(S[j] - S[i])
                missing = target_diffs - covered
            else:
                break
        
        return S
    
    def simulated_annealing(initial_S, max_iters=50000):
        """Simulated annealing to minimize max element while maintaining validity."""
        S = list(initial_S)
        n = len(S)
        
        if n < 2:
            return S, None
        
        # Ensure S is sorted and contains 0
        S.sort()
        if 0 not in S:
            S[0] = 0
            S.sort()
        
        current_energy = float('inf')
        best_S = S.copy()
        best_energy = float('inf')
        
        T = 100.0
        cooling_rate = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current energy
            diffs = set()
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    d = S[j] - S[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid:
                uncovered = len(target_diffs - diffs)
                max_val = max(S)
                current_energy = max_val + 100 * uncovered
            else:
                # Count collisions
                diff_count = {}
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        d = S[j] - S[i]
                        diff_count[d] = diff_count.get(d, 0) + 1
                collisions = sum(c - 1 for c in diff_count.values() if c > 1)
                uncovered = len(target_diffs - set(diff_count.keys()))
                current_energy = 1000 + 100 * collisions + 10 * uncovered + max(S)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_S = S.copy()
            
            # Generate neighbor: pick a random element (not 0) and move it
            if n > 1:
                idx = random.randint(1, n-1)
                old_val = S[idx]
                
                # Try to move to a nearby position
                delta = random.randint(-20, 20)
                new_val = old_val + delta
                
                if new_val < 1 or new_val > 1000:
                    continue
                if new_val in S:
                    continue
                
                S[idx] = new_val
                S.sort()
                
                # Compute new energy
                new_diffs = set()
                new_valid = True
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        d = S[j] - S[i]
                        if d in new_diffs:
                            new_valid = False
                            break
                        new_diffs.add(d)
                    if not new_valid:
                        break
                
                if new_valid:
                    new_uncovered = len(target_diffs - new_diffs)
                    new_max = max(S)
                    new_energy = new_max + 100 * new_uncovered
                else:
                    new_diff_count = {}
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            d = S[j] - S[i]
                            new_diff_count[d] = new_diff_count.get(d, 0) + 1
                    new_collisions = sum(c - 1 for c in new_diff_count.values() if c > 1)
                    new_uncovered = len(target_diffs - set(new_diff_count.keys()))
                    new_energy = 1000 + 100 * new_collisions + 10 * new_uncovered + max(S)
                
                # Accept or reject
                if new_energy < current_energy:
                    current_energy = new_energy
                else:
                    delta_e = new_energy - current_energy
                    if T > 0:
                        accept_prob = 1.0 / (1.0 + delta_e / T)
                        if random.random() < accept_prob:
                            current_energy = new_energy
                        else:
                            S[idx] = old_val
                            S.sort()
                    else:
                        S[idx] = old_val
                        S.sort()
            
            T *= cooling_rate
        
        return best_S, best_energy
    
    # Phase 1: Greedy initialization
    initial_S = greedy_init()
    
    # Phase 2: Simulated annealing
    best_S = initial_S
    best_max = float('inf')
    
    # Check if initial is valid
    valid, max_val = check_sidon_and_cover(initial_S)
    if valid:
        best_max = max_val
        best_S = initial_S.copy()
    
    # Run multiple SA restarts
    sa_restarts = 20
    for r in range(sa_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create perturbed initial solution
        if r == 0:
            S_init = initial_S.copy()
        else:
            S_init = list(initial_S)
            # Randomly perturb some elements
            for _ in range(random.randint(1, 3)):
                idx = random.randint(1, len(S_init)-1)
                S_init[idx] += random.randint(-10, 10)
                S_init = sorted(set(S_init))
                if 0 not in S_init:
                    S_init[0] = 0
                    S_init = sorted(S_init)
        
        sa_S, sa_energy = simulated_annealing(S_init, max_iters=10000)
        
        if sa_S:
            valid, max_val = check_sidon_and_cover(sa_S)
            if valid and max_val < best_max:
                best_max = max_val
                best_S = sa_S.copy()
    
    if best_S is not None and best_max < float('inf'):
        return best_S
    
    # Fallback: return a valid set
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

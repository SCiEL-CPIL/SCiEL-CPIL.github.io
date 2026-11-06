"""Rewritten program for Sidon set with differences 1..30."""
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy: Use a known perfect difference set structure.
    For n=32, a perfect difference set has 32 elements and realizes 31*32/2 = 496 differences.
    We need differences 1..30. The minimal diameter for such a set is known.
    
    We'll use a randomized local search / simulated annealing approach starting from
    a greedy construction, then try to minimize the max element.
    """
    def is_sidon(arr):
        """Check if arr is a Sidon set (all positive differences distinct)."""
        diffs = []
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True
    
    def has_all_diffs(arr, target_diffs):
        """Check if all target differences are present in the set."""
        diff_set = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diff_set.add(arr[j] - arr[i])
        return all(d in diff_set for d in target_diffs)
    
    def get_diffs(arr):
        """Get all positive differences."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs
    
    def objective(arr):
        """
        Objective: minimize max element, but penalize missing differences and non-Sidon.
        Returns a score where lower is better.
        """
        if not is_sidon(arr):
            # Count duplicate differences
            diff_list = []
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    diff_list.append(arr[j] - arr[i])
            dup_count = len(diff_list) - len(set(diff_list))
            return 10**9 + dup_count * 1000 + max(arr)
        
        missing = 0
        diff_set = get_diffs(arr)
        for d in range(1, 31):
            if d not in diff_set:
                missing += 1
        
        if missing > 0:
            return 10**6 + missing * 1000 + max(arr)
        
        return max(arr)
    
    def greedy_construct():
        """Greedy construction of a Sidon set with differences 1..30."""
        a = [0]
        target = list(range(1, 31))
        
        # We need to add elements to cover all differences
        # Start with a known good structure: use quadratic residues or similar
        # Let's try a systematic approach
        
        # First, try to build a set that's close to optimal
        # Known: for n elements, perfect difference set has diameter ~ n^2
        # For n=32, diameter ~ 1024 or so
        
        # Use a greedy approach: add elements to cover missing differences
        # while maintaining Sidon property
        
        for _ in range(100):  # Try multiple random restarts
            a = [0]
            # Add elements in a structured way
            # Use the fact that {0, 1, 4, 9, 16, ...} (squares) has some Sidon properties
            # But we need a more systematic approach
            
            # Try adding elements greedily
            missing_diffs = set(range(1, 31))
            
            for step in range(32):
                if not missing_diffs:
                    break
                
                # Find candidate positions
                best_x = None
                best_score = float('inf')
                
                # Try positions near existing elements
                candidates = set()
                for v in a:
                    for d in list(missing_diffs):
                        candidates.add(v + d)
                        candidates.add(v - d)
                
                # Also try some random positions
                max_val = max(a) if a else 0
                for _ in range(50):
                    candidates.add(random.randint(0, max_val + 50))
                
                candidates = [x for x in candidates if x >= 0 and x not in a]
                
                for x in candidates:
                    trial = sorted(a + [x])
                    if not is_sidon(trial):
                        continue
                    
                    trial_diffs = get_diffs(trial)
                    new_missing = missing_diffs - trial_diffs
                    new_covered = len(missing_diffs) - len(new_missing)
                    
                    # Prefer candidates that cover more missing diffs
                    score = -new_covered * 1000 + x
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                
                if best_x is None:
                    # No valid candidate, try larger range
                    for x in range(max_val + 1, max_val + 200):
                        if x in a:
                            continue
                        trial = sorted(a + [x])
                        if not is_sidon(trial):
                            continue
                        trial_diffs = get_diffs(trial)
                        new_missing = missing_diffs - trial_diffs
                        new_covered = len(missing_diffs) - len(new_missing)
                        score = -new_covered * 1000 + x
                        if score < best_score:
                            best_score = score
                            best_x = x
                
                if best_x is None:
                    break
                
                a = sorted(a + [best_x])
                a_diffs = get_diffs(a)
                missing_diffs = missing_diffs - a_diffs
            
            if not missing_diffs and is_sidon(a):
                return a
        
        # Fallback: simple greedy
        a = [0]
        for d in range(1, 31):
            diff_set = get_diffs(a)
            if d in diff_set:
                continue
            found = False
            for x in range(0, max(a) + d + 10):
                if x in a:
                    continue
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                # Add at far end
                x = 2 * max(a) + d + 1
                while not is_sidon(sorted(a + [x])):
                    x += 1
                a = sorted(a + [x])
        
        return a
    
    def local_search(arr, max_iter=50000):
        """Local search to minimize max element while maintaining constraints."""
        best = sorted(arr)
        best_obj = objective(best)
        current = list(best)
        current_obj = best_obj
        
        for it in range(max_iter):
            # Pick a random element to move (not 0)
            idx = random.randint(1, len(current) - 1)
            old_val = current[idx]
            
            # Try moving to a new position
            # Bias towards smaller values
            if old_val > 0:
                new_val = max(0, old_val + random.randint(-50, 50))
            else:
                new_val = random.randint(0, 50)
            
            if new_val == old_val or new_val in current:
                continue
            
            trial = current[:]
            trial[idx] = new_val
            trial = sorted(trial)
            
            trial_obj = objective(trial)
            
            # Accept if better, or with probability for worse (simulated annealing)
            if trial_obj < current_obj:
                current = trial
                current_obj = trial_obj
                if current_obj < best_obj:
                    best = list(current)
                    best_obj = current_obj
            else:
                # Simulated annealing: accept worse with decreasing probability
                T = max(1, 100 * (1 - it / max_iter))
                if random.random() < (1 / (1 + (trial_obj - current_obj) / T)):
                    current = trial
                    current_obj = trial_obj
        
        return best
    
    # Start with greedy construction
    initial = greedy_construct()
    
    # Verify initial is valid
    if not is_sidon(initial) or not has_all_diffs(initial, range(1, 31)):
        # Try again with different random seed
        for _ in range(10):
            initial = greedy_construct()
            if is_sidon(initial) and has_all_diffs(initial, range(1, 31)):
                break
    
    # Run local search to improve
    result = local_search(initial, max_iter=30000)
    
    # Final verification
    if not is_sidon(result) or not has_all_diffs(result, range(1, 31)):
        # Fallback to initial if local search broke it
        if is_sidon(initial) and has_all_diffs(initial, range(1, 31)):
            result = initial
    
    return sorted(result)
# EVOLVE-BLOCK-END

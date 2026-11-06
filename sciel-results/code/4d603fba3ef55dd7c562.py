"""Construct a Sidon set containing 0 with all differences 1..30, minimizing the largest element."""
import random
import math

def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy: Use quadratic residues modulo p (Paley difference sets) as a starting point,
    then use local search to minimize the diameter.
    """
    
    def is_sidon(arr):
        """Check if arr is a Sidon set (all positive differences distinct)."""
        n = len(arr)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def get_diffs(arr):
        """Get all positive differences."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs
    
    def has_all_diffs(arr, target_diffs):
        """Check if all target differences are present in the set."""
        diff_set = get_diffs(arr)
        return all(d in diff_set for d in target_diffs)
    
    def objective(arr):
        """
        Objective: minimize max element, but penalize missing differences and non-Sidon.
        Returns a score where lower is better.
        """
        if not is_sidon(arr):
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
    
    def quadratic_residues_mod_p(p):
        """Generate quadratic residues modulo p."""
        if p % 4 != 3:
            return None
        q = (p - 1) // 2
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        # Remove 0 if present
        residues.discard(0)
        return sorted(residues)
    
    def try_paley(p):
        """Try to use quadratic residues mod p as a Sidon set."""
        residues = quadratic_residues_mod_p(p)
        if residues is None:
            return None
        
        # Check size
        if len(residues) > 32:
            # Try to remove elements to get size <= 32
            # We need to keep 0 in the set, so we need to shift
            # Actually, let's just work with the residues as-is and see if they work
            pass
        
        # Check if 0 is in the set (it should be since 0^2 = 0)
        # But we excluded 0, so we need to add it back or shift
        # Let's try with 0 included
        full_residues = sorted(set(residues) | {0})
        
        # Check if this is a Sidon set
        if not is_sidon(full_residues):
            return None
        
        # Check if all differences 1..30 are covered
        if not has_all_diffs(full_residues, range(1, 31)):
            return None
        
        # Check size constraint
        if len(full_residues) > 32:
            # Try to remove elements while maintaining constraints
            # This is complex, so let's just return None and try another p
            return None
        
        return full_residues
    
    def local_search(arr, max_iter=50000):
        """Local search to minimize max element while maintaining constraints."""
        best = sorted(arr)
        best_obj = objective(best)
        current = list(best)
        current_obj = best_obj
        
        for it in range(max_iter):
            # Pick a random element to move (not 0)
            non_zero_indices = [i for i, v in enumerate(current) if v > 0]
            if not non_zero_indices:
                break
            idx = random.choice(non_zero_indices)
            old_val = current[idx]
            
            # Try moving to a new position
            # Bias towards smaller values
            new_val = max(0, old_val + random.randint(-30, 30))
            
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
    
    # Try Paley difference sets for various primes
    best_result = None
    best_max = float('inf')
    
    # Primes p ≡ 3 (mod 4) where (p-1)/2 ≤ 32, so p ≤ 65
    # p = 3, 7, 11, 19, 23, 31, 43, 47, 59, 67 (but 67 > 65)
    candidate_primes = [7, 11, 19, 23, 31, 43, 47, 59]
    
    for p in candidate_primes:
        result = try_paley(p)
        if result is not None:
            max_val = max(result)
            if max_val < best_max:
                best_max = max_val
                best_result = result
    
    # If no Paley set works, use greedy construction
    if best_result is None:
        def greedy_construct():
            """Greedy construction of a Sidon set with differences 1..30."""
            a = [0]
            missing_diffs = set(range(1, 31))
            
            for step in range(32):
                if not missing_diffs:
                    break
                
                best_x = None
                best_score = float('inf')
                
                # Try positions near existing elements
                candidates = set()
                for v in a:
                    for d in list(missing_diffs):
                        candidates.add(v + d)
                
                max_val = max(a) if a else 0
                for x in range(max_val + 1, max_val + 100):
                    candidates.add(x)
                
                candidates = [x for x in candidates if x >= 0 and x not in a]
                
                for x in candidates:
                    trial = sorted(a + [x])
                    if not is_sidon(trial):
                        continue
                    
                    trial_diffs = get_diffs(trial)
                    new_covered = len(missing_diffs & trial_diffs)
                    
                    score = -new_covered * 1000 + x
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                
                if best_x is None:
                    break
                
                a = sorted(a + [best_x])
                a_diffs = get_diffs(a)
                missing_diffs = missing_diffs - a_diffs
            
            return a
        
        best_result = greedy_construct()
    
    # Run local search to improve
    if best_result is not None:
        improved = local_search(best_result, max_iter=30000)
        
        # Final verification
        if is_sidon(improved) and has_all_diffs(improved, range(1, 31)):
            best_result = improved
    
    # Final check
    if best_result is None or not is_sidon(best_result) or not has_all_diffs(best_result, range(1, 31)):
        # Fallback: simple construction
        a = [0]
        for d in range(1, 31):
            diff_set = get_diffs(a)
            if d in diff_set:
                continue
            found = False
            for x in range(0, max(a) + d + 20):
                if x in a:
                    continue
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                x = 2 * max(a) + d + 1
                while not is_sidon(sorted(a + [x])):
                    x += 1
                a = sorted(a + [x])
        best_result = a
    
    return sorted(best_result)

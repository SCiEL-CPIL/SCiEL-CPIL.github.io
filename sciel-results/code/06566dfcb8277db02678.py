"""
Deterministic CSP Solver for Max Pairwise-Disjoint Residue Classes.

Strategy:
1. Backtracking search with Forward Checking.
2. MRV (Minimum Remaining Values) heuristic to select the next modulus to assign.
3. Time-limited search to maximize the number of classes (clique size).
4. Fallback to greedy construction if time runs out.
"""
import time
import random
from math import gcd

MAX_MOD = 128
DEADLINE = 85.0  # Safety margin

# Precompute GCDs for all pairs of moduli 2..128
GCD_CACHE = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
for m1 in range(2, MAX_MOD + 1):
    for m2 in range(m1, MAX_MOD + 1):
        g = gcd(m1, m2)
        GCD_CACHE[m1][m2] = g
        GCD_CACHE[m2][m1] = g

def _is_pairwise_disjoint(classes):
    """Exact gcd criterion: classes (r,m),(r',m') disjoint iff gcd(m,m') ∤ (r-r')."""
    n = len(classes)
    for i in range(n):
        r, m = classes[i]
        for j in range(i + 1, n):
            r2, m2 = classes[j]
            g = GCD_CACHE[m][m2]
            if (r - r2) % g == 0:
                return False
    return True

def _is_valid(classes):
    if not classes:
        return False
    seen = set()
    for r, m in classes:
        if not (2 <= m <= MAX_MOD):
            return False
        if not (0 <= r < m):
            return False
        if m in seen:
            return False
        seen.add(m)
    return _is_pairwise_disjoint(classes)

def _greedy_construct():
    """
    Fast greedy construction to get a baseline solution.
    Orders moduli by 'difficulty' (number of prime factors or just random) to avoid conflicts early.
    """
    # Sort moduli by number of divisors? Or just random shuffle for variety in restarts?
    # Let's use a deterministic order first: sort by modulus.
    # Actually, moduli with more constraints (higher gcd with others) are harder.
    # Let's just do a simple greedy: iterate moduli 2..128, pick smallest valid residue.
    classes = []
    for m in range(2, MAX_MOD + 1):
        found = False
        for r in range(m):
            ok = True
            for r2, m2 in classes:
                g = GCD_CACHE[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                classes.append([r, m])
                found = True
                break
    return classes

def _backtrack_search(target_size, deadline, initial_classes=None):
    """
    Attempts to find a set of 'target_size' classes.
    Returns a list of classes if found, else None.
    Uses MRV heuristic.
    """
    if initial_classes is None:
        initial_classes = []
    
    # We want to add (target_size - len(initial_classes)) more classes.
    # Available moduli: those not in initial_classes.
    used_mods = set(m for _, m in initial_classes)
    available_mods = [m for m in range(2, MAX_MOD + 1) if m not in used_mods]
    
    # We need to select 'k' moduli from available_mods and assign residues.
    k_needed = target_size - len(initial_classes)
    if k_needed <= 0:
        return initial_classes
    
    if k_needed > len(available_mods):
        return None

    # State:
    # current_classes: list of [r, m]
    # We will use a recursive backtracking function.
    # To speed up, we maintain the list of currently assigned classes.
    
    # MRV Heuristic:
    # For each available modulus m, calculate the number of valid residues r given current_classes.
    # Pick the m with the fewest valid residues (but > 0).
    
    # Optimization: Precompute valid residues for each modulus against the current set?
    # Since the set changes, we recompute. But the set is small (<= 126).
    
    def get_valid_residues(m, current_classes):
        valid = []
        for r in range(m):
            ok = True
            for r2, m2 in current_classes:
                g = GCD_CACHE[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    # We need to pick a subset of size k_needed from available_mods.
    # This is a combinatorial explosion if we just pick moduli.
    # Better: Pick moduli one by one, using MRV on the *residue* availability.
    
    # However, the choice of *which* moduli to include is also part of the search.
    # We can treat this as:
    # 1. Select a modulus m from available_mods.
    # 2. Assign a residue r.
    # 3. Recurse.
    
    # To prune: If the number of available moduli remaining < k_needed_remaining, fail.
    
    # Let's use a global timeout check.
    start_time = time.time()
    
    # We'll implement a helper that returns the best solution found so far (partial)
    # to allow returning a valid (but smaller) set if target_size is too high.
    best_partial = initial_classes[:]
    
    def backtrack(current_classes, available_mods_list, k_needed):
        nonlocal best_partial
        if time.time() > deadline:
            return None
        
        if k_needed == 0:
            return current_classes
        
        if not available_mods_list:
            return None
            
        if len(available_mods_list) < k_needed:
            return None
            
        # MRV: Find modulus with fewest valid residues
        # Note: We should also consider which moduli are "harder" to place.
        # For each m in available_mods_list, count valid residues.
        
        # Optimization: If we have many moduli, sorting them every step is O(N log N * Cost).
        # N is at most 126. Cost is O(N * M) where M is max modulus.
        # 126 * 126 * 128 ~ 2M ops per level. Depth is ~100. 200M ops. Might be slow in Python.
        # We can approximate or just pick the first one with fewest options.
        
        min_count = MAX_MOD + 1
        best_m = -1
        best_residues = []
        
        # To avoid O(N^2) in the inner loop of MRV selection, we can just iterate.
        # Since we are pruning heavily, the branching factor should be small.
        
        for m in available_mods_list:
            residues = get_valid_residues(m, current_classes)
            count = len(residues)
            if count == 0:
                # This modulus cannot be used with current classes. 
                # We can skip it (remove from available) for this branch.
                # But wait, if we skip it, we reduce available_mods_list.
                # We should filter out unusable moduli first.
                pass
            elif count < min_count:
                min_count = count
                best_m = m
                best_residues = residues
        
        if best_m == -1:
            # No modulus has valid residues
            return None
            
        # Try assigning residues for best_m
        # Shuffle residues to break symmetry? Or try in order?
        # Order doesn't matter much for existence, but might for speed.
        # Let's try all.
        
        new_available = [m for m in available_mods_list if m != best_m]
        
        # Randomize residue order to explore different branches?
        # For deterministic CSP, order is fine.
        for r in best_residues:
            new_classes = current_classes + [[r, best_m]]
            # Update best_partial if this is larger
            if len(new_classes) > len(best_partial):
                best_partial = new_classes[:]
                
            result = backtrack(new_classes, new_available, k_needed - 1)
            if result is not None:
                return result
            if time.time() > deadline:
                return None

    # Initial call
    # To improve performance, we can sort available_mods initially by some heuristic (e.g., primality)
    # But MRV handles the ordering dynamically.
    
    # One optimization: If k_needed is large, the search space is huge.
    # We should start with a smaller k_needed and increase?
    # The caller will handle the binary search or linear increase.
    
    result = backtrack(initial_classes, available_mods, k_needed)
    
    if result is not None:
        return result
    else:
        # Return the best partial solution found
        # Only return if it's valid and non-empty?
        # The caller should check validity.
        if len(best_partial) > len(initial_classes):
            return best_partial
        return None

def construct():
    t0 = time.time()
    
    # 1. Get a baseline greedy solution
    baseline = _greedy_construct()
    best_solution = baseline
    best_size = len(baseline)
    
    # 2. Try to improve using Backtracking
    # We know the maximum possible is 127 (moduli 2..128).
    # Let's try to find a solution of size K.
    # Start from best_size + 1 and go up? Or start high and go down?
    # Starting high is risky if it's infeasible.
    # Let's try to find a solution of size 'best_size + 1'.
    # If found, increase target.
    
    # To be safe and fast, we'll use a time-sliced approach.
    # Allocate time for each attempt.
    
    # First, let's see if we can just validate the baseline.
    if not _is_valid(baseline):
        # Fallback to a known valid small set
        return [[0, 2]]
        
    # Try to expand
    current_size = best_size
    # Max possible is 127
    while current_size < 127:
        # Check time
        if time.time() > DEADLINE - 5.0:
            break
            
        target_size = current_size + 1
        # Allocate a slice of time, e.g., 5 seconds
        slice_deadline = time.time() + 5.0
        
        # Use the current best solution as the initial state for backtracking
        # This forces the new solution to contain the old one? 
        # No, _backtrack_search(initial_classes) adds to it.
        # This is a "completion" search.
        # It might be better to restart from scratch with a target size?
        # But completion is faster if the initial state is good.
        
        # However, if the initial state is "bad" (locks out many moduli), completion fails.
        # We can try completion first. If it fails, we can try a fresh search for that size.
        
        result = _backtrack_search(target_size, slice_deadline, initial_classes=best_solution)
        
        if result is not None and _is_valid(result) and len(result) >= target_size:
            best_solution = result
            best_size = len(result)
            current_size = best_size
        else:
            # Completion failed.
            # Try a fresh search for this target size?
            # This is expensive. Let's just stop if completion fails, 
            # or try one fresh search with a very short time limit.
            if time.time() < slice_deadline - 1.0:
                # Try fresh search with remaining time
                fresh_deadline = time.time() + 2.0
                fresh_result = _backtrack_search(target_size, fresh_deadline, initial_classes=[])
                if fresh_result is not None and _is_valid(fresh_result) and len(fresh_result) >= target_size:
                    best_solution = fresh_result
                    best_size = len(result)
                    current_size = best_size
                else:
                    # If fresh also fails, assume this size is hard/infeasible with current strategy
                    # Break to avoid wasting time? Or continue to next size?
                    # Usually, if you can't find K+1 from a good K, you might not find K+2.
                    # But maybe a different K is possible?
                    # Let's just break to be safe and return best found.
                    break
            else:
                break

    # Final validation
    if _is_valid(best_solution):
        return best_solution
    else:
        # Fallback
        return [[0, 2]]

# Ensure the function is exposed
if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found {len(res)} classes")
    print(res[:5])

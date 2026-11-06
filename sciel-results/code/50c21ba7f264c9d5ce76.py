import time
import random
import sys
import heapq

# EVOLVE-BLOCK-START

def _valid_set(a):
    """Check Sidon property and coverage 1..30. Return (ok, max_diff_set, diam)."""
    a = sorted(a)
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False, diffs, a[-1]
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False, diffs, a[-1]
    return True, diffs, a[-1]

def _cost(a):
    """Cost function: penalize missing differences and diameter. Lower is better."""
    a = sorted(a)
    n = len(a)
    diffs = set()
    conflict = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                conflict += 1
            diffs.add(d)
    missing = sum(1 for d in range(1, 31) if d not in diffs)
    diam = a[-1]
    return 1000 * conflict + 100 * missing + diam

def _is_valid(a):
    ok, _, _ = _valid_set(a)
    return ok

def _greedy_build():
    """Build a valid Sidon set greedily, covering 1..30."""
    a = [0]
    used = set()
    for _ in range(50):
        covered = set()
        for x in a:
            for y in a:
                if x != y:
                    covered.add(abs(x - y))
        missing = set(range(1, 31)) - covered
        if not missing:
            break
        best_x = None
        best_score = -1
        for x in range(1, 200):
            if x in a:
                continue
            new_diffs = set()
            ok = True
            for v in a:
                d = abs(x - v)
                if d in used or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            covers = len(new_diffs & missing)
            if covers > best_score:
                best_score = covers
                best_x = x
        if best_x is None:
            break
        a.append(best_x)
        a.sort()
        for v in a[:-1]:
            used.add(abs(best_x - v))
    for d in range(1, 31):
        if d not in used:
            found = False
            for x in range(1, 300):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok and d in new_diffs:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                x = max(a) + d + 1
                while True:
                    new_diffs = set()
                    ok = True
                    for v in a:
                        diff = x - v
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        break
                    x += 1
    return a

def _branch_and_bound(deadline):
    """
    Branch and Bound search for minimal diameter Sidon set covering 1..30.
    """
    best_diam = float('inf')
    best_set = None
    start_time = time.time()
    
    # Initial greedy solution to set an upper bound
    init_set = _greedy_build()
    if _is_valid(init_set):
        best_diam = max(init_set)
        best_set = list(init_set)
    
    # We will search for solutions with diameter < best_diam
    # We can iterate deepening or just use the bound.
    # Since best_diam is likely small (e.g., ~30-40), we can search directly.
    
    # State: (current_set, used_diffs, current_diam)
    # We use a stack for DFS to avoid recursion depth issues, though recursion is fine for small n.
    # Let's use recursion with a deadline check.
    
    def dfs(a, used_diffs, depth):
        nonlocal best_diam, best_set, start_time
        
        if time.time() - start_time > deadline:
            return False # Timeout
        
        # Pruning 1: Diameter bound
        if a[-1] >= best_diam:
            return False
            
        # Pruning 2: Coverage check
        missing = [d for d in range(1, 31) if d not in used_diffs]
        if not missing:
            # Found a valid solution
            if a[-1] < best_diam:
                best_diam = a[-1]
                best_set = list(a)
            return True # Found a solution, we can stop this branch (or continue to find smaller? No, DFS finds first, but we want minimal. Since we sort candidates, the first found is likely good, but not guaranteed minimal if we don't check all. However, B&B with diameter bound ensures we find the minimal if we search in increasing diameter order or use the bound effectively).
            # Actually, since we return True, we stop exploring this branch. 
            # To ensure minimal, we should continue searching if we haven't exhausted all possibilities for smaller diameters.
            # But since we update best_diam, subsequent branches will be pruned if they can't beat it.
            
        # Pruning 3: Max possible new diffs
        # Each new element x can add at most len(a) new diffs.
        # We have limited "slots" left? No, we can add up to 32 elements total.
        # But we can estimate the max coverage.
        # A simpler bound: If the number of missing diffs is large, and we are close to the diameter limit, prune.
        
        # Candidate Generation
        # We want to add x > a[-1] such that x - y is not in used_diffs for any y in a.
        # And x should be < best_diam.
        
        last = a[-1]
        # Candidates are generated by missing diffs:
        # If d is missing, and y is in a, then x = y + d is a candidate.
        # We collect all such candidates.
        
        candidates = set()
        for d in missing:
            for y in a:
                x = y + d
                if x > last:
                    candidates.add(x)
        
        # Sort candidates to try smaller ones first (to find small diameter solutions faster)
        candidates = sorted(candidates)
        
        # Filter candidates:
        # 1. x < best_diam
        # 2. x doesn't create duplicate diffs
        
        for x in candidates:
            if x >= best_diam:
                continue
            
            # Check if x is valid
            new_diffs = set()
            valid = True
            for y in a:
                d = x - y
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Additional pruning: Does this x actually help?
            # It must cover at least one missing diff. (It does by construction, but let's verify)
            # covers = new_diffs & set(missing)
            # if not covers: continue # This might be too aggressive? If it doesn't cover missing, it might still be useful later?
            # No, if it doesn't cover any *currently* missing diff, it's not helpful for the immediate goal.
            # But maybe it allows a smaller diameter later? Unlikely.
            
            a.append(x)
            dfs(a, used_diffs | new_diffs, depth + 1)
            a.pop()
            
            if time.time() - start_time > deadline:
                return False
            
            # If we found a solution with diameter < x, we can break?
            # No, we want the global minimum.
            
    dfs([0], set(), 0)
    
    return best_set, best_diam

def _local_optimize(a, deadline):
    """Local optimization: try reducing each element."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    best_valid = _is_valid(a)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            for new_val in range(1, old):
                if new_val in a:
                    continue
                trial = list(a)
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        best_valid = True
                        improved = True
                        break
    
    return best, best_diam, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Branch and Bound (Primary Strategy)
    if time.time() < deadline - 10:
        bb_set, bb_diam = _branch_and_bound(deadline - 10)
        if bb_set is not None:
            best_set = bb_set
            best_diam = bb_diam
    
    # Phase 2: Fallback Greedy if B&B fails or times out without finding
    if best_set is None or not _is_valid(best_set):
        best_set = _greedy_build()
        best_diam = max(best_set)
        if not _is_valid(best_set):
            # Last resort: hardcode a known solution or expand greedy
            # Greedy should always work for 1..30
            pass

    # Phase 3: Local Optimization
    if best_set is not None and time.time() < deadline - 2:
        opt, opt_diam, opt_valid = _local_optimize(best_set, deadline - 2)
        if opt_valid and opt_diam < best_diam:
            best_set = opt
            best_diam = opt_diam
            
    return best_set

# EVOLVE-BLOCK-END

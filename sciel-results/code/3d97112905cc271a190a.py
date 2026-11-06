import time
import numpy as np
from scipy.optimize import milp, LinearConstraint
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate triples in [51, 100] summing to 200
    values_large = list(range(51, 101))
    n_vars = len(values_large)
    val_to_idx = {v: i for i, v in enumerate(values_large)}
    
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        S_large = set(values_large)
    else:
        # Solve Minimum Hitting Set via ILP
        A = np.zeros((len(triples), n_vars), dtype=int)
        for i, t in enumerate(triples):
            for v in t:
                A[i, val_to_idx[v]] = 1
                
        c = np.ones(n_vars, dtype=int)
        bounds = [(0, 1)] * n_vars
        integrality = np.ones(n_vars, dtype=int)
        
        H = None
        try:
            constraints = LinearConstraint(A, 1, None)
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
            if res.status == 0:
                H_candidate = set()
                for i in range(n_vars):
                    if res.x[i] > 0.5:
                        H_candidate.add(values_large[i])
                # Verify
                valid = True
                for t in triples:
                    if not any(v in H_candidate for v in t):
                        valid = False
                        break
                if valid:
                    H = H_candidate
        except Exception:
            H = None

        if H is None:
            # Fallback: Greedy Hitting Set
            active_triples = set(triples)
            H = set()
            while active_triples and time.time() < deadline - 10:
                counts = {}
                for t in active_triples:
                    for v in t:
                        counts[v] = counts.get(v, 0) + 1
                if not counts:
                    break
                best_elem = max(counts, key=counts.get)
                H.add(best_elem)
                active_triples = {t for t in active_triples if best_elem not in t}
        
        S_large = set(v for v in values_large if v not in H)

    # Helper to compute reachable sums
    def get_reachable(S_set):
        reach = [False] * 201
        reach[0] = True
        for v in S_set:
            for s in range(200, v - 1, -1):
                if reach[s - v]:
                    reach[s] = True
        return reach

    # Helper to greedily construct smalls given forbidden targets
    def greedy_smalls(forbidden_targets):
        # forbidden_targets is a set of sums that cannot be formed by any subset of the result
        # We want max subset of [1, 50] such that no subset sums to any t in forbidden_targets
        # Note: 0 is always reachable (empty set), but we care about non-empty subsets summing to t.
        # If t=0, we must not have any subset summing to 0 (only empty set does, so OK unless we allow 0 in S, which we don't).
        # The condition "no subset summing to 200" means if 200 is in forbidden_targets, we just can't form 200.
        
        S_small = set()
        # We maintain reachable sums of S_small
        reach_small = [False] * 201
        reach_small[0] = True
        
        # Try adding elements. Order matters.
        # We try to add elements that are "least likely" to create conflicts?
        # Or just try all orders?
        # For speed, we use a fixed order but we can shuffle.
        # Let's try descending order first (large smalls first)
        candidates = list(range(50, 0, -1))
        
        for x in candidates:
            # Check if adding x creates a forbidden sum
            # New sums will be s + x where s is in reach_small
            # We need to ensure no new sum is in forbidden_targets
            # AND that x itself doesn't form a forbidden sum with previous elements
            # Actually, if we add x, the new reachable sums are old_reach + old_reach shifted by x.
            # We need to check if any of these new sums are in forbidden_targets.
            # Also, if x itself is in forbidden_targets (sum of {x}), it's invalid.
            
            is_safe = True
            # Check if x itself is forbidden
            if x in forbidden_targets:
                is_safe = False
            else:
                # Check combinations
                for s in range(200 - x + 1, -1, -1):
                    if reach_small[s]:
                        new_sum = s + x
                        if new_sum > 200:
                            continue
                        if new_sum in forbidden_targets:
                            is_safe = False
                            break
            if is_safe:
                S_small.add(x)
                # Update reach_small
                for s in range(200, x - 1, -1):
                    if reach_small[s - x]:
                        reach_small[s] = True
        return S_small

    # Initial state
    R_large = get_reachable(S_large)
    # Forbidden targets for smalls:
    # If a subset of S_small sums to t, and a subset of S_large sums to 200-t, total is 200.
    # So t is forbidden if 200-t is in R_large.
    forbidden_targets = set()
    for r in range(201):
        if R_large[r]:
            forbidden_targets.add(200 - r)
    
    S_small = greedy_smalls(forbidden_targets)
    S = S_large.union(S_small)
    best_size = len(S)
    best_S = S

    # Local Search: Try removing one large element to allow more smalls
    # We iterate a few times
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each element from S_large
        # To speed up, we can just try a few random ones or all of them
        # Since |S_large| <= 50, trying all is fine if greedy_smalls is fast.
        for l in list(S_large):
            if time.time() > deadline - 5:
                break
            
            S_large_trial = S_large - {l}
            R_large_trial = get_reachable(S_large_trial)
            forbidden_targets_trial = set()
            for r in range(201):
                if R_large_trial[r]:
                    forbidden_targets_trial.add(200 - r)
            
            S_small_trial = greedy_smalls(forbidden_targets_trial)
            size_trial = len(S_large_trial) + len(S_small_trial)
            
            if size_trial > best_size:
                S_large = S_large_trial
                S_small = S_small_trial
                S = S_large.union(S_small)
                best_size = size_trial
                improved = True
                # If we improved, we should probably restart the loop or continue
                # Restarting might be better to find new removals
                break 

    # Final verification
    reach_final = get_reachable(S)
    if reach_final[200]:
        # Fallback: return just S_large (which is valid)
        return sorted(S_large)
        
    return sorted(S)

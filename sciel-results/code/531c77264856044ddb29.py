"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _is_valid_addition(S, x):
    """
    Check if adding x to valid set S keeps it valid.
    S is a list of distinct integers. x is not in S.
    Conditions:
    1. x does not divide sum of any 3 distinct elements in S.
    2. For each a in S, a does not divide (x + u + v) for any distinct u, v in S \ {a}.
    """
    n = len(S)
    if n < 3:
        # If S has < 3 elements, condition 1 is vacuously true (no triples).
        # Condition 2: for each a in S, check if a divides x + u + v where u,v are others.
        # If n < 2, no pairs u,v exist.
        for a in S:
            others = [v for v in S if v != a]
            for u, v in combinations(others, 2):
                if (x + u + v) % a == 0:
                    return False
        return True

    # Check 1: x does not divide sum of any 3 in S
    # Optimization: Precompute sums? No, just iterate.
    # S is small (~48). 48C3 ~ 17k.
    for t in combinations(S, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: For each a in S, a does not divide x + u + v for u, v in S \ {a}
    # We can iterate a, then u, v.
    # To speed up, we can precompute for each a, the set of sums of pairs from S\{a} mod a?
    # No, just check.
    for i, a in enumerate(S):
        # We need u, v from S excluding a.
        # If n is small, just nested loops.
        # x + u + v == k*a
        # u + v == k*a - x
        # We can check if there exist u, v in S\{a} such that (u+v) % a == (-x) % a.
        target_res = (-x) % a
        others = [S[j] for j in range(n) if j != i]
        # Check if any pair in others sums to target_res mod a
        # 48C2 ~ 1k. 48 * 1k = 48k ops.
        for u_idx in range(len(others)):
            u = others[u_idx]
            # v must be in others, v != u
            # (u + v) % a == target_res
            # v % a == (target_res - u) % a
            needed_res = (target_res - u) % a
            # Check if any v in others (v != u) has v % a == needed_res
            # We can just iterate
            for v_idx in range(u_idx + 1, len(others)):
                v = others[v_idx]
                if v % a == needed_res:
                    return False
    return True

def _valid_set(S):
    """Full validation, used for final check or initialization."""
    S = list(S)
    n = len(S)
    for i, a in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_large_first():
    """
    Greedy construction starting from largest numbers.
    Large numbers are less likely to divide sums of others because the sums are bounded.
    """
    selected = []
    # Try adding from 192 down to 1
    for x in range(_N, 0, -1):
        if _is_valid_addition(selected, x):
            selected.append(x)
        if time.time() > _DEADLINE - 5:
            break
    return selected

def _local_search(S, time_budget=30.0):
    """
    Local search with 1-swap, 2-swap, and insertion.
    """
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = set(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE - 5:
        improved = False
        
        # 1. Try Insertion
        if len(S) < _MAX:
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            # Try a few random candidates first, then all if fast
            for x in candidates[:20]:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # 2. Try 1-Swap
        if not improved and len(S) > 0:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for a in to_remove[:10]:
                for x in candidates[:10]:
                    # Check if swapping a with x is valid
                    S_new = [v for v in S if v != a] + [x]
                    # We can optimize this check, but _valid_set is safe.
                    # Since S was valid, we only need to check:
                    # 1. x doesn't divide sum of 3 in S\{a}
                    # 2. a is gone, so no check for a.
                    # 3. For each b in S\{a}, b doesn't divide x + u + v where u,v in S\{a,b}
                    # This is same as _is_valid_addition(S\{a}, x)
                    S_minus_a = [v for v in S if v != a]
                    if _is_valid_addition(S_minus_a, x):
                        S = S_new
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # 3. Try 2-Swap
        if not improved and len(S) >= 2:
            # Pick 2 to remove, 2 to add
            to_remove = random.sample(S, 2)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for x1 in candidates[:5]:
                for x2 in candidates[:5]:
                    if x1 == x2: continue
                    S_minus_2 = [v for v in S if v not in to_remove]
                    # Check if x1 can be added to S_minus_2
                    if _is_valid_addition(S_minus_2, x1):
                        S_temp = S_minus_2 + [x1]
                        # Check if x2 can be added to S_temp
                        if _is_valid_addition(S_temp, x2):
                            S = S_temp + [x2]
                            improved = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = list(S)
                            break
                if improved:
                    break
        
        # 4. Perturbation if stuck
        if not improved:
            if len(S) > 5:
                # Remove 3 random, add 3 random large
                to_remove = random.sample(S, 3)
                S = [v for v in S if v not in to_remove]
                candidates = [v for v in range(_N, 1, -1) if v not in S]
                random.shuffle(candidates)
                for x in candidates[:3]:
                    if _is_valid_addition(S, x):
                        S.append(x)
            else:
                # Reset to greedy if too small
                S = _greedy_large_first()
                if len(S) > best_len:
                    best_len = len(S)
                    best = list(S)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Greedy from large to small
    best = _greedy_large_first()
    
    # Phase 2: Local search
    if time.time() < _DEADLINE - 10:
        S_ls = _local_search(best, time_budget=40.0)
        if len(S_ls) > len(best):
            best = S_ls
            
    # Phase 3: Another local search pass with different random seed
    if time.time() < _DEADLINE - 5:
        random.seed(int(time.time() * 1000) % 10000)
        S_ls2 = _local_search(best, time_budget=10.0)
        if len(S_ls2) > len(best):
            best = S_ls2

    # Final sanity check
    if not _valid_set(best):
        # Fallback to a known small valid set if something went wrong
        return list(range(1, 10)) # Just a placeholder, shouldn't happen
        
    return best
# EVOLVE-BLOCK-END

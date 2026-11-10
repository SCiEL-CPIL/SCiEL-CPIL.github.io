"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations
from collections import Counter

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _valid_set(S):
    """Full validation of a set."""
    S = list(S)
    n = len(S)
    if n < 4:
        return True
    for i, a in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _is_valid_addition(S, x):
    """
    Check if adding x to valid set S keeps it valid.
    S is a list of distinct integers. x is not in S.
    """
    n = len(S)
    
    # Check 1: x does not divide sum of any 3 in S
    if n >= 3:
        for t in combinations(S, 3):
            if sum(t) % x == 0:
                return False
    
    # Check 2: For each a in S, a does not divide x + u + v for u, v in S \ {a}
    if n >= 3:
        for a in S:
            others = [v for v in S if v != a]
            if len(others) < 2:
                continue
            
            # Count residues of others modulo a
            res_counts = Counter(v % a for v in others)
            target_res = (-x) % a
            found = False
            for u in others:
                r_u = u % a
                r_needed = (target_res - r_u) % a
                if r_u == r_needed:
                    if res_counts[r_u] >= 2:
                        found = True
                        break
                else:
                    if res_counts.get(r_needed, 0) >= 1:
                        found = True
                        break
            if found:
                return False
                
    return True

def _greedy_large_first():
    """
    Greedy construction starting from largest numbers.
    """
    selected = []
    for x in range(_N, 0, -1):
        if _is_valid_addition(selected, x):
            selected.append(x)
        if time.time() > _DEADLINE - 5:
            break
    return selected

def _local_search(S, time_budget=30.0):
    """
    Local search with Insertion and Swap.
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
            for x in candidates[:50]: # Check more candidates
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # 2. Try 1-Swap (Remove one, add one)
        if not improved and len(S) > 0 and len(S) < _MAX:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for a in to_remove[:10]:
                for x in candidates[:10]:
                    S_minus_a = [v for v in S if v != a]
                    if _is_valid_addition(S_minus_a, x):
                        S = S_minus_a + [x]
                        improved = True
                        break
                if improved:
                    break
        
        # 3. Perturbation if stuck
        if not improved:
            if len(S) > 10:
                to_remove = random.sample(S, 3)
                S = [v for v in S if v not in to_remove]
            else:
                S = _greedy_large_first()
                if len(S) > best_len:
                    best_len = len(S)
                    best = list(S)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Greedy from large to small
    best = _greedy_large_first()
    
    # Phase 2: Local Search
    if time.time() < _DEADLINE - 10:
        S_ls = _local_search(best, time_budget=40.0)
        if len(S_ls) > len(best):
            best = S_ls
    
    # Phase 3: Another local search pass if time remains
    if time.time() < _DEADLINE - 3:
        S_ls2 = _local_search(best, time_budget=5.0)
        if len(S_ls2) > len(best):
            best = S_ls2

    return best
# EVOLVE-BLOCK-END

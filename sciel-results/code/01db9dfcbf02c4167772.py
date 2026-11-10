"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

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
    # If n < 3, this check passes trivially
    if n >= 3:
        # Optimization: If x is large, sum of 3 max is 3*192=576.
        # If x > 576, it can't divide anything (but x <= 192).
        # We just check combinations.
        for t in combinations(S, 3):
            if sum(t) % x == 0:
                return False
    
    # Check 2: For each a in S, a does not divide x + u + v for u, v in S \ {a}
    # This requires u, v to be distinct and not a.
    # If n < 2, we can't form u, v distinct from a (since we need 2 others).
    # Wait, if n=2, S={a, b}. Others for a is {b}. Can't pick 2 distinct from {b}.
    # So Check 2 only applies if n >= 3?
    # "sum of any k=3 DISTINCT OTHER selected integers".
    # The element being tested is 'a'. The summands are 3 distinct OTHERS.
    # So we need 3 distinct elements from S \ {a}.
    # This means n-1 >= 3 => n >= 4.
    
    if n >= 4:
        for a in S:
            # We need to check if there exist u, v, w in S \ {a} distinct such that (x is not involved in this sum? No.)
            # Wait, the condition is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
            # When we add x, we must check:
            # 1. Does x divide sum of 3 others in S? (Checked above)
            # 2. For each a in S, does a divide sum of 3 others in S U {x}?
            #    The 3 others can be from S\{a} or include x.
            #    Case 2a: All 3 others from S\{a}. This was already valid in S. So no need to check.
            #    Case 2b: 2 others from S\{a}, 1 is x. Sum = x + u + v.
            #    Case 2c: 1 other from S\{a}, 2 are x? No, x is distinct, only one x.
            #    Case 2d: 0 others from S\{a}, 3 are x? No.
            
            # So we only need to check if a divides (x + u + v) for distinct u, v in S \ {a}.
            
            others = [v for v in S if v != a]
            # We need to check if there exist distinct u, v in others such that (x + u + v) % a == 0
            # This is equivalent to (u + v) % a == (-x) % a
            
            target_res = (-x) % a
            found = False
            # To speed this up, we can use a set of residues.
            # But n is small (~48). O(n^2) is fine.
            for i in range(len(others)):
                u = others[i]
                # We need v such that (u + v) % a == target_res
                # v % a == (target_res - u) % a
                needed_res = (target_res - u) % a
                for j in range(i + 1, len(others)):
                    v = others[j]
                    if v % a == needed_res:
                        found = True
                        break
                if found:
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
        # This helps if we are stuck at a local maximum where no single insertion works,
        # but swapping might free up space for a better configuration.
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
                        # Size didn't change, but structure might be better.
                        # We don't update best_len unless size increases, but we keep S for further exploration.
                        break
                if improved:
                    break
        
        # 3. Perturbation if stuck
        if not improved:
            if len(S) > 10:
                # Remove 3 random elements to break local optima
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

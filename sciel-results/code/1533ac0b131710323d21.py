"""Optimized solver for no-dividing-3-sum task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill


def _is_valid_addition(S, x):
    """Check if adding x to valid set S keeps it valid.
    Only need to check:
    1. x does not divide sum of any 3 in S
    2. No a in S divides sum of x + 2 others from S
    """
    S_list = list(S)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    for t in combinations(S_list, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for i, a in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        for t in combinations(others, 2):
            if (x + t[0] + t[1]) % a == 0:
                return False
    
    return True


def _is_valid_swap(S, remove_val, add_val):
    """Check if swapping remove_val with add_val in valid set S keeps it valid."""
    if add_val in S:
        return False
    if remove_val not in S:
        return False
    
    S_without = [v for v in S if v != remove_val]
    
    # Check: add_val does not divide sum of any 3 in S_without
    for t in combinations(S_without, 3):
        if sum(t) % add_val == 0:
            return False
    
    # Check: for each a in S_without, a does not divide sum of add_val + 2 others
    n = len(S_without)
    for i, a in enumerate(S_without):
        others = [S_without[j] for j in range(n) if j != i]
        for t in combinations(others, 2):
            if (add_val + t[0] + t[1]) % a == 0:
                return False
    
    return True


def _greedy(descending=True):
    """Greedy construction."""
    selected = []
    if descending:
        order = range(_N, 0, -1)
    else:
        order = range(1, _N + 1)
    for x in order:
        if time.time() > _DEADLINE:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected


def _local_search(S, time_budget=30.0):
    """Local search with swaps and insertions to improve set size."""
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:80]:
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps: remove one from S, add one not in S
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:40]:
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:40]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S, a, x):
                        S = [v for v in S if v != a] + [x]
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # If no improvement, perturb
        if not improved:
            if len(S) > 3:
                to_remove = random.sample(S, min(3, len(S) - 2))
                S = [v for v in S if v not in to_remove]
    
    return best


def _multi_start_greedy(num_starts=30, time_budget=20.0):
    """Try multiple greedy orders with randomization."""
    best = []
    start_time = time.time()
    
    for _ in range(num_starts):
        if time.time() - start_time > time_budget or time.time() > _DEADLINE:
            break
        
        # Randomized greedy: shuffle order with bias toward larger
        order = list(range(1, _N + 1))
        weights = [random.random() * _N + v for v in order]
        order.sort(key=lambda v: -weights[v - 1])
        
        selected = []
        for x in order:
            if time.time() > _DEADLINE:
                break
            if _is_valid_addition(selected, x):
                selected.append(x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best


def _hill_climbing(S, time_budget=20.0):
    """Hill climbing: try to increase size by removing minimal elements."""
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try all possible single additions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:50]:
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:50]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S, a, x):
                        S = [v for v in S if v != a] + [x]
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        if not improved:
            # Perturb: remove 1-2 random elements
            if len(S) > 2:
                k = random.randint(1, 2)
                to_remove = random.sample(S, k)
                S = [v for v in S if v not in to_remove]
    
    return best


def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Basic greedy
    S1 = _greedy(descending=True)
    S2 = _greedy(descending=False)
    best = max([S1, S2], key=len)
    
    # Phase 2: Multi-start greedy
    if time.time() < _DEADLINE - 25:
        S3 = _multi_start_greedy(num_starts=40, time_budget=15.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search from best
    if time.time() < _DEADLINE - 15:
        S4 = _local_search(best, time_budget=12.0)
        if len(S4) > len(best):
            best = S4
    
    # Phase 4: Hill climbing
    if time.time() < _DEADLINE - 8:
        S5 = _hill_climbing(best, time_budget=6.0)
        if len(S5) > len(best):
            best = S5
    
    # Phase 5: Another local search round if time permits
    if time.time() < _DEADLINE - 3:
        S6 = _local_search(best, time_budget=3.0)
        if len(S6) > len(best):
            best = S6
    
    return best
# EVOLVE-BLOCK-END

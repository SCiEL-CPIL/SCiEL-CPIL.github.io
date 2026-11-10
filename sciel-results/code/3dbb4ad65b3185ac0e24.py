"""Optimized solver for no-dividing-3-sum task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 82.0  # Safety margin before 90s kill

# Precomputation for fast validity checks using bitsets
# For each x, we precompute for each residue r, the set of values v where v % x == r
# We use Python ints as bitsets for fast intersection checks

# bitsets[x][r] = integer bitset where bit v is set if v % x == r
_bitsets = [None] * (_N + 1)

def _precompute():
    global _bitsets
    for x in range(1, _N + 1):
        _bitsets[x] = [0] * x
        for v in range(1, _N + 1):
            r = v % x
            _bitsets[x][r] |= (1 << v)

# Run precomputation once at start
_precompute()

def _make_bitset(S_set):
    """Convert a set to a bitset integer."""
    bs = 0
    for v in S_set:
        bs |= (1 << v)
    return bs

def _count_bits(n):
    """Count set bits in integer n."""
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


def _is_valid_addition(S_set, S_bitset, x):
    """Check if adding x to valid set S_set keeps it valid."""
    S_list = list(S_set)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            r = (-(b + c)) % x
            candidates = _bitsets[x][r] & S_bitset
            candidates &= ~(1 << b)
            candidates &= ~(1 << c)
            if candidates:
                return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for a in S_set:
        for b in S_set:
            if b == a:
                continue
            r_c = (-x - b) % a
            candidates_c = _bitsets[a][r_c] & S_bitset
            candidates_c &= ~(1 << a)
            candidates_c &= ~(1 << b)
            if candidates_c:
                return False
                
    return True


def _is_valid_swap(S_set, S_bitset, remove_val, add_val):
    """Check if swapping remove_val with add_val in valid set S_set keeps it valid."""
    if add_val in S_set:
        return False
    if remove_val not in S_set:
        return False
    
    S_without = S_set - {remove_val}
    S_without_bitset = S_bitset & ~(1 << remove_val)
    
    # Check: add_val does not divide sum of any 3 in S_without
    S_list = list(S_without)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            r = (-(b + c)) % add_val
            candidates = _bitsets[add_val][r] & S_without_bitset
            candidates &= ~(1 << b)
            candidates &= ~(1 << c)
            if candidates:
                return False
    
    # Check: for each a in S_without, a does not divide sum of add_val + 2 others
    for a in S_without:
        r = (-add_val) % a
        for b in S_without:
            if b == a:
                continue
            r_c = (-add_val - b) % a
            candidates_c = _bitsets[a][r_c] & S_without_bitset
            candidates_c &= ~(1 << a)
            candidates_c &= ~(1 << b)
            if candidates_c:
                return False
                
    return True


def _greedy(descending=True):
    """Greedy construction."""
    selected = []
    S_set = set()
    S_bitset = 0
    if descending:
        order = range(_N, 0, -1)
    else:
        order = range(1, _N + 1)
    for x in order:
        if time.time() > _DEADLINE:
            break
        if _is_valid_addition(S_set, S_bitset, x):
            selected.append(x)
            S_set.add(x)
            S_bitset |= (1 << x)
    return selected


def _local_search(S, time_budget=30.0):
    """Local search with swaps and insertions to improve set size."""
    S = list(S)
    S_set = set(S)
    S_bitset = _make_bitset(S_set)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            for x in candidates[:100]:
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S_set, S_bitset, x):
                    S.append(x)
                    S_set.add(x)
                    S_bitset |= (1 << x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps: remove one from S, add one not in S
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            
            for a in to_remove[:60]:
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:60]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S_set, S_bitset, a, x):
                        # Perform swap
                        S.remove(a)
                        S.append(x)
                        S_set.remove(a)
                        S_set.add(x)
                        S_bitset = S_bitset & ~(1 << a)
                        S_bitset |= (1 << x)
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
                k = random.randint(1, 3)
                to_remove = random.sample(S, k)
                for v in to_remove:
                    S.remove(v)
                    S_set.remove(v)
                    S_bitset &= ~(1 << v)
    
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
        S_set = set()
        S_bitset = 0
        for x in order:
            if time.time() > _DEADLINE:
                break
            if _is_valid_addition(S_set, S_bitset, x):
                selected.append(x)
                S_set.add(x)
                S_bitset |= (1 << x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best


def _final_sweep(S):
    """Try to add any remaining element in sorted order."""
    S_set = set(S)
    S_bitset = _make_bitset(S_set)
    S_list = list(S)
    for x in range(1, _N + 1):
        if time.time() > _DEADLINE:
            break
        if x not in S_set and len(S_list) < _MAX:
            if _is_valid_addition(S_set, S_bitset, x):
                S_list.append(x)
                S_set.add(x)
                S_bitset |= (1 << x)
    return S_list


def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Basic greedy
    S1 = _greedy(descending=True)
    S2 = _greedy(descending=False)
    best = max([S1, S2], key=len)
    
    # Phase 2: Multi-start greedy
    if time.time() < _DEADLINE - 30:
        S3 = _multi_start_greedy(num_starts=50, time_budget=18.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search from best
    if time.time() < _DEADLINE - 15:
        S4 = _local_search(best, time_budget=12.0)
        if len(S4) > len(best):
            best = S4
    
    # Phase 4: Final sweep - try all additions systematically
    if time.time() < _DEADLINE - 5:
        best = _final_sweep(best)
    
    # Phase 5: Another local search round if time permits
    if time.time() < _DEADLINE - 3:
        S5 = _local_search(best, time_budget=3.0)
        if len(S5) > len(best):
            best = S5
    
    return best
# EVOLVE-BLOCK-END

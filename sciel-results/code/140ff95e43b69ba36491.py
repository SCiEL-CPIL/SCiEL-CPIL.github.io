"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 192
MAX_SIZE = 48

# Precomputation for fast constraint checking
# For each x in 1..N, precompute which triples of (i,j,k) with i<j<k, i,j,k != x
# have (i+j+k) % x == 0. But that's too much memory.
# Instead, we'll precompute pair sums modulo x for fast checking.

def _precompute_pair_sums():
    """
    For each x in 1..N, precompute a set of (b+c) % x for all pairs (b,c) with b<c, b!=x, c!=x.
    Actually, we need to know WHICH pairs, not just the sums, because we need to check
    if a specific pair (b,c) from our current set causes a conflict.
    
    Better approach: Precompute for each x, a dictionary mapping sum_mod -> list of pairs.
    But even better: just do the check on the fly with early termination.
    
    For the incremental check when adding x to set S:
    1. Check if x divides sum of any 3 elements in S: O(|S|^3)
    2. For each a in S, check if a divides (x + b + c) for any b,c in S\{a}: O(|S|^3)
    
    To speed this up, we can precompute for each a, and for each possible value of (x % a),
    the set of pairs (b,c) such that (b+c) % a == (a - x%a) % a.
    
    Let's precompute: for each a in 1..N, and for each remainder r in 0..a-1,
    the list of pairs (b,c) with b<c, b!=a, c!=a, such that (b+c) % a == r.
    """
    pair_sums_mod = {}
    for a in range(1, N + 1):
        pair_sums_mod[a] = {}
        for r in range(a):
            pair_sums_mod[a][r] = []
        
        # For each pair (b,c) with b<c, b!=a, c!=a
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(b + 1, N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                pair_sums_mod[a][r].append((b, c))
    
    return pair_sums_mod

def _precompute_triples_for_x():
    """
    For each x in 1..N, precompute the list of triples (i,j,k) with i<j<k, i!=x, j!=x, k!=x
    such that (i+j+k) % x == 0.
    """
    triples = {}
    for x in range(1, N + 1):
        triples[x] = []
        for i in range(1, N + 1):
            if i == x:
                continue
            for j in range(i + 1, N + 1):
                if j == x:
                    continue
                for k in range(j + 1, N + 1):
                    if k == x:
                        continue
                    if (i + j + k) % x == 0:
                        triples[x].append((i, j, k))
    return triples

# Precompute once at module load time
_PAIR_SUMS_MOD = _precompute_pair_sums()
_TRIPLES_FOR_X = _precompute_triples_for_x()

def _can_add_fast(x, current_list):
    """
    Fast check if x can be added to current_list (which is valid).
    Uses precomputed data for speed.
    """
    n = len(current_list)
    current_set = set(current_list)
    
    # Check 1: Does x divide sum of any 3 elements in current_list?
    if n >= 3:
        # Use precomputed triples for x
        # But we need to check if any triple from _TRIPLES_FOR_X[x] is a subset of current_list
        # This is still O(|triples|) which could be large for small x.
        # Alternative: just iterate over combinations of current_list
        for i in range(n - 2):
            a = current_list[i]
            for j in range(i + 1, n - 1):
                b = current_list[j]
                for k in range(j + 1, n):
                    c = current_list[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # Check 2: For each a in current_list, does a divide (x + b + c) for any b,c in current_list\{a}?
    if n >= 2:
        for idx, a in enumerate(current_list):
            others = [current_list[j] for j in range(n) if j != idx]
            m = len(others)
            if m >= 2:
                # We need to check if there exist b,c in others such that (x+b+c) % a == 0
                # i.e., (b+c) % a == (a - x%a) % a
                target_r = (a - x % a) % a
                # Check all pairs in others
                for p in range(m - 1):
                    b = others[p]
                    for q in range(p + 1, m):
                        c = others[q]
                        if (b + c) % a == target_r:
                            return False
    
    return True

def _greedy_construct(order, deadline):
    """Greedy construction with fast checking."""
    selected = []
    selected_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _can_add_fast(x, selected):
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_improve(selected, deadline):
    """Improve a valid set by adding elements or doing swaps."""
    best = list(selected)
    best_set = set(best)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            if _can_add_fast(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one element, then greedily add as many as possible
        if len(best) > 0:
            for rem_idx in range(len(best)):
                if time.time() > deadline:
                    break
                rem_elem = best[rem_idx]
                temp = [e for i, e in enumerate(best) if i != rem_idx]
                temp_set = set(temp)
                
                # Greedily add elements to temp
                for x in range(N, 0, -1):
                    if time.time() > deadline:
                        break
                    if x in temp_set:
                        continue
                    if _can_add_fast(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                
                if len(temp) > len(best):
                    best = temp
                    best_set = temp_set
                    improved = True
                    break
    
    return best

def _walksat_improve(selected, deadline):
    """
    WalkSAT-style local search to improve the set.
    We want to maximize the size of the valid set.
    """
    best = list(selected)
    best_size = len(best)
    best_set = set(best)
    
    # Try random restarts with local search
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from a perturbed version of best
        current = list(best)
        current_set = set(current)
        
        # Randomly remove a few elements
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            current_set.discard(removed)
        
        # Greedily add elements
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if _can_add_fast(x, current):
                current.append(x)
                current_set.add(x)
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
            best_set = current_set
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Phase 1: Multiple greedy starts with different orderings
    best = []
    
    # Start 1: Descending order
    order1 = list(range(N, 0, -1))
    cand1 = _greedy_construct(order1, deadline)
    if len(cand1) > len(best):
        best = cand1
    
    # Start 2: Primes first
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(N, 0, -1) if is_prime(x)]
    non_primes = [x for x in range(N, 0, -1) if not is_prime(x)]
    order2 = primes + non_primes
    cand2 = _greedy_construct(order2, deadline)
    if len(cand2) > len(best):
        best = cand2
    
    # Start 3: Randomized restarts
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 10.0:
            break
        order = list(range(N, 0, -1))
        random.shuffle(order)
        cand = _greedy_construct(order, deadline)
        if len(cand) > len(best):
            best = cand
    
    # Phase 2: Local improvement
    if time.time() < deadline - 5.0:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: WalkSAT-style improvement
    if time.time() < deadline - 2.0:
        improved = _walksat_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if len(best) == 0:
        best = [N]
    
    # Ensure distinct positive integers <= N
    best = list(set(best))
    best = [x for x in best if 1 <= x <= N]
    best.sort(reverse=True)
    
    # Enforce max size
    if len(best) > MAX_SIZE:
        best = best[:MAX_SIZE]
    
    return best
# EVOLVE-BLOCK-END

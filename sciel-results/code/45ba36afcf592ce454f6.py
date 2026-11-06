"""Combined: precomputed gcd matrix + multi-greedy + potential-guided local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_TIME_LIMIT = 88.0
_DEADLINE = None

# Precompute gcd matrix for all pairs in [1, 128]
_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g

_UNIVERSE = list(range(1, _N + 1))

def _can_add(S, x):
    """Check if adding x to S keeps the set triple-free using precomputed gcd matrix."""
    # S is a list
    n = len(S)
    if n < 2:
        return True
    
    # Precompute gcd of x with all elements in S
    # To avoid repeated lookups and allow fast comparison
    gcds = [_gcd_matrix[s][x] for s in S]
    
    for i in range(n):
        gi = gcds[i]
        if gi == 1: 
            # If gcd is 1, it can only match if others are 1 and gcd(S[i], S[j]) is 1.
            # But if gcd is 1, it's less likely to cause conflicts than higher gcds?
            # Actually, we must check all.
            pass
        for j in range(i + 1, n):
            gj = gcds[j]
            if gi == gj:
                if _gcd_matrix[S[i]][S[j]] == gi:
                    return False
    return True

def _potential(S):
    """Count how many elements in the universe can be added to S."""
    Sset = set(S)
    count = 0
    for x in _UNIVERSE:
        if x in Sset:
            continue
        if _can_add(S, x):
            count += 1
    return count

def _greedy(order, deadline):
    """Greedy: iterate order, add if valid."""
    S = []
    Sset = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x):
            S.append(x)
            Sset.add(x)
    return S

def _add_all_possible(S, Sset, deadline):
    """Add all elements that can be added, in universe order."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in _UNIVERSE:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
                break
    return S, Sset

def _local_search(S, deadline, rng):
    """
    Local search with 1-swap, 2-swap, and potential guidance.
    Objective: Maximize size. Tie-breaker: Maximize potential.
    """
    S = list(S)
    Sset = set(S)
    current_size = len(S)
    current_pot = _potential(S)
    
    # Phase 1: Add all possible
    S, Sset = _add_all_possible(S, Sset, deadline - 10)
    current_size = len(S)
    current_pot = _potential(S)
    
    d = deadline - 5
    
    # Phase 2: Local Search Loop
    for _iter in range(100):
        if time.time() > d:
            break
        
        progress = False
        
        # Try 1-Swap: Remove 1, Add 1+
        for idx in range(len(S)):
            if time.time() > d:
                break
            removed = S[idx]
            S2 = S[:idx] + S[idx + 1:]
            S2set = set(S2)
            
            best_swap = None
            best_size = len(S2)
            best_pot = -1
            
            # Try adding any x
            for x in _UNIVERSE:
                if x in S2set:
                    continue
                if _can_add(S2, x):
                    S3 = S2 + [x]
                    S3set = set(S3)
                    # Try to add more greedily to see full gain
                    added = []
                    for y in _UNIVERSE:
                        if y in S3set:
                            continue
                        if _can_add(S3, y):
                            S3.append(y)
                            S3set.add(y)
                            added.append(y)
                            # Stop early if we already have a good gain
                            if len(added) >= 3: 
                                break
                    
                    new_size = len(S3)
                    if new_size > best_size:
                        best_size = new_size
                        best_swap = (x, added)
                        best_pot = _potential(S3)
                        if new_size > current_size + 2: # Significant gain
                            break
                    elif new_size == best_size:
                        # Tie-break by potential
                        pot = _potential(S3)
                        if pot > best_pot:
                            best_pot = pot
                            best_swap = (x, added)
            
            if best_swap:
                x, added = best_swap
                S = S2 + [x]
                Sset = set(S)
                for y in added:
                    S.append(y)
                    Sset.add(y)
                current_size = len(S)
                current_pot = _potential(S)
                progress = True
                break
        
        if progress:
            continue
            
        # If 1-swap failed, try 2-Swap: Remove 2, Add 2+
        found_2swap = False
        n = len(S)
        for i in range(n):
            if time.time() > d:
                break
            for j in range(i + 1, n):
                if time.time() > d:
                    break
                S2 = S[:i] + S[i+1:j] + S[j+1:]
                S2set = set(S2)
                
                best_swap = None
                best_size = len(S2)
                best_pot = -1
                
                for x in _UNIVERSE:
                    if x in S2set:
                        continue
                    if _can_add(S2, x):
                        S3 = S2 + [x]
                        S3set = set(S3)
                        added = []
                        for y in _UNIVERSE:
                            if y in S3set:
                                continue
                            if _can_add(S3, y):
                                S3.append(y)
                                S3set.add(y)
                                added.append(y)
                                if len(added) >= 2:
                                    break
                        
                        new_size = len(S3)
                        if new_size > best_size:
                            best_size = new_size
                            best_swap = (x, added)
                            best_pot = _potential(S3)
                            if new_size > current_size + 2:
                                break
                        elif new_size == best_size:
                            pot = _potential(S3)
                            if pot > best_pot:
                                best_pot = pot
                                best_swap = (x, added)
                
                if best_swap:
                    x, added = best_swap
                    S = S2 + [x]
                    Sset = set(S)
                    for y in added:
                        S.append(y)
                        Sset.add(y)
                    current_size = len(S)
                    current_pot = _potential(S)
                    found_2swap = True
                    break
            if found_2swap:
                break
        
        if found_2swap:
            continue
            
        # If no progress, try a "kick": randomly remove 2 elements and restart local search
        if current_size > 5:
            # Remove 2 random elements
            remove_idx = rng.sample(range(len(S)), 2)
            remove_idx.sort(reverse=True)
            for idx in remove_idx:
                S.pop(idx)
            Sset = set(S)
            current_size = len(S)
            # Re-run add_all_possible to fill back up
            S, Sset = _add_all_possible(S, Sset, d)
            current_size = len(S)
            current_pot = _potential(S)
            
    # Final polish
    S, Sset = _add_all_possible(S, Sset, deadline)
    
    return S

def _multi_greedy_local(deadline):
    rng = random.Random(42)
    best = []
    best_size = 0
    
    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0] + [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 3 == 0] + [x for x in range(1, 129) if x % 3 != 0],
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_count(), x)),
        # Prime-heavy first
        sorted(range(1, 129), key=lambda x: -len([p for p in range(2, x+1) if x % p == 0])),
    ]
    # Add random orderings
    for _ in range(10):
        o = _UNIVERSE[:]
        rng.shuffle(o)
        orders.append(o)

    candidates = []
    for order in orders:
        if time.time() > deadline - 20:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > best_size:
            best_size = len(cand)
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    
    # Run local search on top 5 candidates
    for cand in candidates[:5]:
        if time.time() > deadline - 10:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > best_size:
            best_size = len(improved)
            best = improved
            
    # If time remains, try to improve the best solution further with random kicks
    if time.time() < deadline - 5 and best:
        # Try a few random perturbations of the best solution
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            perturbed = best[:]
            # Remove 2-3 random elements
            k = rng.randint(2, 3)
            if len(perturbed) > k:
                remove_idx = rng.sample(range(len(perturbed)), k)
                remove_idx.sort(reverse=True)
                for idx in remove_idx:
                    perturbed.pop(idx)
            
            improved = _local_search(perturbed, deadline, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best = improved

    return best

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy
        result = []
        for x in range(1, 129):
            ok = True
            for i in range(len(result)):
                a = result[i]
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    if ga == _gcd_matrix[b][x] == _gcd_matrix[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END

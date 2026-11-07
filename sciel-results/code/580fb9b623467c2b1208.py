"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # Evaluator kills at 90s; 8s safety margin

# ---------- Prime Exponent Structure ----------
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PN = len(_PRIMES)
_EXP = {}          # n -> tuple of exponents
_EXP_LIST = {}     # n -> list of exponents

def _build_exp_table():
    for n in range(2, 41):
        m = n
        vec = [0] * _PN
        for i, p in enumerate(_PRIMES):
            while m % p == 0:
                vec[i] += 1
                m //= p
        _EXP[n] = tuple(vec)
        _EXP_LIST[n] = vec

_build_exp_table()

# ---------- Conflict Graph Pre-computation ----------
# Two numbers a and b are in conflict if their exponent vectors are rationally dependent.
# This is a necessary condition for them to cause a collision.

def _are_rationally_dependent(a, b):
    va = _EXP_LIST[a]
    vb = _EXP_LIST[b]
    # Find first non-zero index for each
    ia = -1
    for i in range(_PN):
        if va[i] != 0:
            ia = i
            break
    ib = -1
    for i in range(_PN):
        if vb[i] != 0:
            ib = i
            break
            
    if ia == -1 or ib == -1:
        return False
        
    if ia != ib:
        return False

    # ia == ib
    pivot_a = va[ia]
    pivot_b = vb[ib]
    
    for i in range(_PN):
        if va[i] * pivot_b != vb[i] * pivot_a:
            return False
    return True

# Precompute conflict pairs
_CONFLICT = {n: set() for n in range(2, 41)}
for a in range(2, 41):
    for b in range(a + 1, 41):
        if _are_rationally_dependent(a, b):
            _CONFLICT[a].add(b)
            _CONFLICT[b].add(a)

# ---------- Core Validity Machinery ----------

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    # Optimization: if x is in prods, return False (covered by q=x, q//x=1)
    # Standard check
    for q in prods:
        if q % x == 0:
            if (q // x) in prods:
                return False
    return True

def _try_add(selected, prods, x):
    if not _can_add(prods, x):
        return False, selected, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, selected + [x], new_prods

# ---------- Construction ----------

def _construct_initial_set():
    """Pure-powers heuristic (known valid 16-set)."""
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]

def _greedy_construct(order):
    selected = []
    prods = {1}
    for x in order:
        if len(selected) >= 18:
            break
        # Fast check: is x in conflict with any selected?
        conflict = False
        for s in selected:
            if s in _CONFLICT[x]:
                conflict = True
                break
        if conflict:
            continue
        if _can_add(prods, x):
            selected.append(x)
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
    return selected, prods

def _construct_random_initial_set(rng):
    selected = []
    prods = {1}
    candidates = list(range(2, 41))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= 18:
            break
        # Fast conflict check
        conflict = False
        for s in selected:
            if s in _CONFLICT[x]:
                conflict = True
                break
        if conflict:
            continue
        if _can_add(prods, x):
            selected.append(x)
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
    return selected, prods

# ---------- Local Search ----------

def _local_search(selected, prods, deadline, max_rounds=20):
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # 1. Try adding any candidate
        # Filter candidates that are in conflict with existing set to save time
        valid_candidates = []
        for x in candidates:
            # Check if x conflicts with any selected
            is_conflict = False
            for s in selected:
                if s in _CONFLICT[x]:
                    is_conflict = True
                    break
            if not is_conflict:
                valid_candidates.append(x)
                
        for x in valid_candidates:
            if time.time() > deadline:
                break
            ok, new_sel, new_prods = _try_add(selected, prods, x)
            if ok:
                selected = new_sel
                prods = new_prods
                improved = True
                used = set(selected)
                candidates = [c for c in range(2, 41) if c not in used]
                if len(selected) >= 18:
                    break

        if not improved:
            # 2. Single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                # Recompute products without removed
                rem_products = {1}
                for x in selected:
                    if x == removed:
                        continue
                    new_rem = set(rem_products)
                    for p2 in rem_products:
                        new_rem.add(p2 * x)
                    rem_products = new_rem
                
                # Try adding each candidate
                for x in range(2, 41):
                    if x in selected:
                        continue
                    if x in _CONFLICT[removed]:
                        continue
                    ok, _, _ = _try_add(selected, rem_products, x)
                    if ok:
                        # Perform the swap
                        new_sel = selected[:i] + selected[i+1:] + [x]
                        new_prods = set(rem_products)
                        for p2 in rem_products:
                            new_prods.add(p2 * x)
                        selected = new_sel
                        prods = new_prods
                        improved = True
                        break
                if improved:
                    break
    
    return selected, prods

def _double_swap(selected, prods, deadline):
    """Try removing two elements and adding two different ones."""
    if len(selected) < 4:
        return selected, prods
    
    n = len(selected)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if time.time() > deadline:
                break
            
        removed1 = selected[i]
        removed2 = selected[j]
        
        # Recompute products without removed1 and removed2
        rem_products = {1}
        for x in selected:
            if x == removed1 or x == removed2:
                continue
            new_rem = set(rem_products)
            for p2 in rem_products:
                new_rem.add(p2 * x)
            rem_products = new_rem
        
        # Try adding pairs of candidates
        candidates = [c for c in range(2, 41) if c not in selected]
        for x in candidates:
            if time.time() > deadline:
                break
            if x in _CONFLICT[removed1] or x in _CONFLICT[removed2]:
                continue
            ok1, _, prods_after_x = _try_add(selected, rem_products, x)
            if not ok1:
                continue
            for y in candidates:
                if y == x:
                    continue
                if y in _CONFLICT[removed1] or y in _CONFLICT[removed2]:
                    continue
                ok2, _, prods_after_y = _try_add(selected, prods_after_x, y)
                if ok2:
                    # Perform the double swap
                    new_sel = [s for s in selected if s != removed1 and s != removed2]
                    new_sel.append(x)
                    new_sel.append(y)
                    return new_sel, prods_after_y
    
    return selected, prods

# ---------- Main Construct ----------

def construct():
    deadline = _DEADLINE
    
    # Start with the known 16-set
    best_set = _construct_initial_set()
    best_prods = _subset_products_set(best_set)
    best_len = len(best_set)
    
    # Verify the initial set is valid
    if len(best_prods) != 2 ** best_len:
        # Fallback to a smaller known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        best_prods = _subset_products_set(best_set)
        best_len = len(best_set)
    
    # Try greedy constructions with different orderings
    orders = [
        sorted(range(2, 41)),
        sorted(range(2, 41), reverse=True),
        sorted(range(2, 41), key=lambda n: (len([1 for e in _EXP_LIST[n] if e]), n)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s, p = _greedy_construct(order)
        if len(s) > best_len:
            best_set = s
            best_prods = p
            best_len = len(s)
    
    # Random restarts with local search
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Create a random initial set
        s, p = _construct_random_initial_set(rng)
        
        # Apply local search
        s, p = _local_search(s, p, deadline, max_rounds=10)
        
        if len(s) > best_len:
            best_set = s
            best_prods = p
            best_len = len(s)
        
        # Try double swap if still not at 18
        if len(s) < 18 and time.time() < deadline - 5:
            s2, p2 = _double_swap(s, p, deadline)
            if len(s2) > best_len:
                best_set = s2
                best_prods = p2
                best_len = len(s2)
    
    # Final verification
    if best_set:
        prods = _subset_products_set(best_set)
        if len(prods) != 2 ** len(best_set):
            # Fallback to a known valid set
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END

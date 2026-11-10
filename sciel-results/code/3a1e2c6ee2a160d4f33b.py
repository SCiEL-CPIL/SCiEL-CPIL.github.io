"""Rewritten solver: exact integer arithmetic, multi-start greedy + local search with deadline guard."""
from math import gcd, lcm
from fractions import Fraction
import random
import time

N = 256
# Precompute reciprocal as exact Fraction and float
RECI = {}
RECI_F = {}
for x in range(2, N + 1):
    RECI[x] = Fraction(1, x)
    RECI_F[x] = 1.0 / x

# Precompute lcm matrix for 2..256 (only what we need)
# We'll use lcm(x, y) > 256 check; precompute a boolean "conflict" matrix
# conflict[x][y] = True if lcm(x, y) <= 256 (i.e., NOT allowed)
# But that's 255x255 ~ 65k entries, fine.
# Actually we only need to check when adding elements; we can compute on the fly
# but precomputing helps speed.

# Let's precompute a list of "incompatible partners" for each x:
# incompat[x] = set of y != x such that lcm(x, y) <= 256
incompat = [set() for _ in range(N + 1)]
for x in range(2, N + 1):
    for y in range(x + 1, N + 1):
        if lcm(x, y) <= 256:
            incompat[x].add(y)
            incompat[y].add(x)


def val_score(S):
    """Exact sum of reciprocals."""
    s = Fraction(0, 1)
    for x in S:
        s += RECI[x]
    return s


def val_score_float(S):
    s = 0.0
    for x in S:
        s += RECI_F[x]
    return s


def can_add(x, S):
    """Check if x can be added to set S (no lcm conflict)."""
    if x in S:
        return False
    for y in S:
        if y in incompat[x]:
            return False
    return True


def greedy_construct(order, time_budget):
    """Greedy add elements in given order if compatible."""
    S = set()
    start = time.time()
    for x in order:
        if time.time() - start > time_budget:
            break
        if can_add(x, S):
            S.add(x)
    return S


def local_search(S, deadline):
    """Local search: try swaps and adds to improve reciprocal sum.
    Uses exact Fractions for objective comparison but float for speed in some checks.
    Returns improved set or same.
    """
    S = set(S)
    S_list = sorted(S)
    current_exact = val_score(S)
    current_float = sum(RECI_F[x] for x in S)
    
    all_candidates = list(range(2, N + 1))
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try adding a single element
        best_add = None
        best_add_gain = Fraction(0, 1)
        for x in all_candidates:
            if x in S:
                continue
            if can_add(x, S):
                gain = RECI[x]
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add = x
        
        if best_add is not None:
            S.add(best_add)
            current_exact += best_add_gain
            current_float += RECI_F[best_add]
            improved = True
            continue
        
        # Try swaps: remove one, add another
        best_swap = None
        best_swap_gain = Fraction(0, 1)
        
        # Sort removals by their reciprocal (remove low-value first for potential gain)
        S_sorted = sorted(S, key=lambda x: RECI_F[x])
        
        for rem in S_sorted:
            if time.time() > deadline:
                return S
            # After removing rem, which elements can we add?
            # We need elements not in S, and compatible with S - {rem}
            S_without_rem = S - {rem}
            # The gain of swapping rem with x is RECI[x] - RECI[rem]
            # We want to maximize this, so pick largest RECI[x] among compatible
            # But x must not be in S_without_rem, and must be compatible with all in S_without_rem
            # Since rem was in S, removing it only relaxes constraints for x that were
            # conflicting with rem.
            # Let's find the best x to add.
            
            # Elements that were previously blocked by rem but could now be added:
            # Any x not in S that is compatible with S_without_rem
            # This is equivalent to: x not in S, and for all y in S_without_rem, y not in incompat[x]
            
            # Optimize: check candidates in order of descending reciprocal
            # But we need to check compatibility with S_without_rem
            # S_without_rem can be large, but let's just iterate candidates
            
            # To speed up, only consider x with RECI[x] > RECI[rem] (positive gain)
            rem_recip = RECI[rem]
            for x in all_candidates:
                if x in S:
                    continue
                if RECI[x] <= rem_recip:
                    continue  # no gain or negative
                # Check compatibility with S_without_rem
                compat = True
                for y in S_without_rem:
                    if y in incompat[x]:
                        compat = False
                        break
                if compat:
                    gain = RECI[x] - rem_recip
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (rem, x)
        
        if best_swap is not None:
            rem, x = best_swap
            S.discard(rem)
            S.add(x)
            current_exact += best_swap_gain
            current_float += RECI_F[x] - RECI_F[rem]
            improved = True
    
    return S


def multi_start_greedy_local(deadline):
    """Multi-start greedy with local search."""
    best_S = set()
    best_score = Fraction(0, 1)
    
    # Several random orders
    random.seed(42)
    
    # First, a natural greedy (descending by value, i.e., ascending by x)
    order_desc = list(range(2, N + 1))  # smallest first = highest reciprocal
    S = greedy_construct(order_desc, 1.0)
    S = local_search(S, deadline)
    score = val_score(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    if time.time() > deadline:
        return best_S
    
    # Try various randomized orders
    num_starts = 0
    while time.time() < deadline - 2.0:  # leave 2s safety margin
        num_starts += 1
        # Random permutation
        order = list(range(2, N + 1))
        random.shuffle(order)
        S = greedy_construct(order, 0.5)
        if not S:
            continue
        S = local_search(S, deadline)
        score = val_score(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
        
        # Try a "value-priority" greedy with random ties
        if num_starts % 3 == 0:
            # Sort by reciprocal (ascending x) but with random perturbation
            order2 = list(range(2, N + 1))
            random.shuffle(order2)
            S2 = greedy_construct(order2, 0.5)
            if S2:
                S2 = local_search(S2, deadline)
                score2 = val_score(S2)
                if score2 > best_score:
                    best_score = score2
                    best_S = set(S2)
    
    return best_S


def construct():
    start_time = time.time()
    # Deadline: 85 seconds (5s safety margin before 90s limit)
    deadline = start_time + 85.0
    
    # Phase 1: Quick greedy + local search
    best_S = multi_start_greedy_local(deadline)
    
    if not best_S:
        # Fallback: should not happen, but return at least one element
        # 2 is always valid as a singleton
        return [2]
    
    return sorted(best_S)

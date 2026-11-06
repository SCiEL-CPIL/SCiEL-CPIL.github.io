import time
import random
from collections import defaultdict

def _check_add(S, x):
    """S is a set of ints. Check if x can be added to S."""
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: for x, no 3 distinct in S sum to 0 mod x
    if n >= 3:
        if x == 1:
            return False
        rem_idx = defaultdict(list)
        for i, v in enumerate(S_list):
            rem_idx[v % x].append(i)
        for i in range(n):
            for j in range(i + 1, n):
                target = (-(S_list[i] + S_list[j])) % x
                if target in rem_idx:
                    for k in rem_idx[target]:
                        if k != i and k != j:
                            return False
    
    # Condition 2: for each a in S, new triples involving x
    for i in range(n):
        a = S_list[i]
        # The set S ∪ {x} \ {a} has n elements
        # We need to check if any 3 distinct elements in this set sum to 0 mod a
        # New triples must include x (old triples were already valid)
        # So check pairs (b, c) in S \ {a} with b + c + x ≡ 0 (mod a)
        if a == 1:
            # Any sum is divisible by 1
            if n >= 3:  # S\{a} has n-1 elements, plus x makes n. Need 3 distinct.
                return False
            continue
        rem_idx = defaultdict(list)
        for j in range(n):
            if j == i:
                continue
            rem_idx[S_list[j] % a].append(j)
        for j in range(n):
            if j == i:
                continue
            target = (-x - S_list[j]) % a
            if target in rem_idx:
                for k in rem_idx[target]:
                    if k != j:
                        return False
    
    return True

def _is_valid(S):
    """Check if set S is valid."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        # Check all 3-subsets of S \ {a}
        others = [S_list[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        if a == 1:
            return False
        for b_idx in range(len(others)):
            for c_idx in range(b_idx + 1, len(others)):
                for d_idx in range(c_idx + 1, len(others)):
                    if (others[b_idx] + others[c_idx] + others[d_idx]) % a == 0:
                        return False
    return True

def _greedy(order):
    """Greedy build using given order of candidates."""
    S = set()
    for x in order:
        if len(S) >= 48:
            break
        if _check_add(S, x):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = set()
    
    # Multiple greedy initializations
    random.seed(42)
    
    # Descending order
    S = _greedy(range(192, 0, -1))
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Ascending order
    S = _greedy(range(1, 193))
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # Several random orders
    for _ in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy(order)
        if len(S) > len(best):
            best = S
    
    # Local search on best
    S = set(best)
    unselected = set(range(1, 193)) - S
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        for x in list(unselected):
            if len(S) >= 48:
                break
            if _check_add(S, x):
                S.add(x)
                unselected.discard(x)
                improved = True
        
        # Try swaps: remove one, add one or two
        if len(S) < 48:
            for a in list(S):
                if time.time() > deadline:
                    break
                S.remove(a)
                unselected.add(a)
                added = []
                for x in list(unselected):
                    if len(S) >= 48:
                        break
                    if _check_add(S, x):
                        S.add(x)
                        unselected.discard(x)
                        added.append(x)
                if len(added) > 1:
                    improved = True
                elif len(added) == 0:
                    # Revert
                    S.add(a)
                    unselected.discard(a)
                # If len(added) == 1, we swapped a for x, net zero change in size
                # But might enable further additions, so keep it
                # Actually, let's only keep if we can add more after
                if len(added) == 1:
                    # Try to add more
                    can_add_more = False
                    for x in list(unselected):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            can_add_more = True
                            break
                    if not can_add_more:
                        # Revert the swap
                        S.discard(added[0])
                        unselected.add(added[0])
                        S.add(a)
                        unselected.discard(a)
    
    if len(S) > len(best):
        best = S
    
    if time.time() > deadline:
        return sorted(best)
    
    # More random restarts with local search
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy(order)
        
        # Local search
        unselected = set(range(1, 193)) - S
        for _ in range(100):
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
            # Try to add
            added_any = False
            for x in list(unselected):
                if len(S) >= 48:
                    break
                if _check_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    added_any = True
            if not added_any:
                # Try swap
                swapped = False
                for a in list(S):
                    S.remove(a)
                    unselected.add(a)
                    added = []
                    for x in list(unselected):
                        if len(S) >= 48:
                            break
                        if _check_add(S, x):
                            S.add(x)
                            unselected.discard(x)
                            added.append(x)
                    if len(added) >= 1:
                        swapped = True
                        break
                    else:
                        S.add(a)
                        unselected.discard(a)
                if not swapped:
                    break
        
        if len(S) > len(best):
            best = S
    
    return sorted(best)

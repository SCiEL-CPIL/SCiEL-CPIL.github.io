import time
import random

DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid(S):
    """Return True if no three distinct a,b,c in S have a+b,a+c,b+c all in S."""
    sset = set(S)
    n = len(S)
    # Iterate triples; use set lookups
    # For performance, precompute pair sums presence
    # A triple (a,b,c) is bad if a+b in sset and a+c in sset and b+c in sset
    # We can iterate over pairs and check.
    # Efficient: for each pair (a,b) with a<b, if a+b in sset, then for each c in S, c!=a,c!=b,
    # check a+c in sset and b+c in sset.
    # That's O(n^3) worst case but with early exits.
    # Alternative: for each c, for each pair (a,b) with a<b, a!=c,b!=c, check.
    # Let's do: for each c in S: for each a in S, a<c: for each b in S, b<c, b!=a:
    #   check a+b, a+c, b+c all in sset.
    lst = sorted(S)
    for ci in range(len(lst)):
        c = lst[ci]
        for ai in range(ci):
            a = lst[ai]
            ac = a + c
            if ac not in sset:
                continue
            for bi in range(ai):
                b = lst[bi]
                if a + b in sset and b + c in sset:
                    return False
    return True

def construct():
    t0 = time.time()
    # Start with the 64 large numbers
    S = list(range(65, 129))
    sset = set(S)
    # Try to add small numbers greedily
    smalls = list(range(1, 65))
    random.seed(42)
    best = S[:]
    best_size = len(S)
    
    # Greedy add: try adding each small number if it keeps validity
    # But adding one may require removing others. Do local search.
    
    # First, greedy add in random order
    order = smalls[:]
    random.shuffle(order)
    current = S[:]
    cset = set(current)
    for x in order:
        if time.time() > DEADLINE:
            break
        trial = current + [x]
        tset = cset | {x}
        # Quick check: only need to check triples involving x
        # Triple with x and two others a,b: need a+b, a+x, b+x all in tset
        # a+x and b+x: since x<=64, a,b>=65, a+x could be in tset if a+x in [65,128]
        # Let's just do full check but optimized for the new element
        ok = True
        lst = sorted(current)
        for ai in range(len(lst)):
            a = lst[ai]
            ax = a + x
            if ax not in tset:
                continue
            for bi in range(ai):
                b = lst[bi]
                if a + b in tset and b + x in tset:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            current.append(x)
            cset.add(x)
    if len(current) > best_size:
        best = current[:]
        best_size = len(current)
    
    # Local search: try to improve by swaps
    all_nums = list(range(1, 129))
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        cur_set = set(best)
        # Try adding a random small number not in best, removing a conflicting one if needed
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            candidates = [x for x in all_nums if x not in cur_set]
            if not candidates:
                break
            x = random.choice(candidates)
            trial = best + [x]
            tset = cur_set | {x}
            if _is_valid(trial):
                best = trial
                cur_set = tset
                best_size = len(best)
                improved = True
                break
            else:
                # Try removing one element that's in a bad triple with x, then re-add x
                # Find a bad triple involving x
                lst = sorted(best)
                bad_partners = None
                for ai in range(len(lst)):
                    a = lst[ai]
                    ax = a + x
                    if ax not in tset:
                        continue
                    for bi in range(ai):
                        b = lst[bi]
                        if a + b in tset and b + x in tset:
                            bad_partners = (a, b)
                            break
                    if bad_partners:
                        break
                if bad_partners:
                    a, b = bad_partners
                    for rem in (a, b, x):
                        if rem in cur_set:
                            newbest = [y for y in best if y != rem]
                            newbest.append(x)
                            if _is_valid(newbest):
                                if len(newbest) > best_size:
                                    best = newbest
                                    cur_set = set(best)
                                    best_size = len(best)
                                    improved = True
                                    break
    # Ensure return is plain ints in a list
    return [int(x) for x in best]

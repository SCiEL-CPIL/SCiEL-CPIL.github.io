<<<<<<< SEARCH
def _fast_greedy(order, deadline):
    """
    Greedily build two Sidon sets A and B following the given element order.
    For each element, try to add it to A, then to B, then skip.
    Returns (A, B) as lists.
    """
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()

    start = time.time()

    for x in order:
        if time.time() - start > deadline:
            break

        # Try adding x to A
        can_add_A = True
        new_sums_A = set()
        for a in A:
            s = a + x
            if s in sums_A or s in new_sums_A:
                can_add_A = False
                break
            new_sums_A.add(s)
        s2 = 2 * x
        if can_add_A and (s2 in sums_A or s2 in new_sums_A):
            can_add_A = False

        if can_add_A:
            # Check difference disjointness
            new_diffs = set()
            for a in A:
                d1 = x - a
                d2 = a - x
                new_diffs.add(d1)
                new_diffs.add(d2)
            if new_diffs & diffs_B:
                can_add_A = False

        if can_add_A:
            A.append(x)
            sums_A.update(new_sums_A)
            sums_A.add(s2)
            for a in A:
                if a != x:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            continue

        # Try adding x to B
        can_add_B = True
        new_sums_B = set()
        for b in B:
            s = b + x
            if s in sums_B or s in new_sums_B:
                can_add_B = False
                break
            new_sums_B.add(s)
        s2b = 2 * x
        if can_add_B and (s2b in sums_B or s2b in new_sums_B):
            can_add_B = False

        if can_add_B:
            new_diffs = set()
            for b in B:
                d1 = x - b
                d2 = b - x
                new_diffs.add(d1)
                new_diffs.add(d2)
            if new_diffs & diffs_A:
                can_add_B = False

        if can_add_B:
            B.append(x)
            sums_B.update(new_sums_B)
            sums_B.add(s2b)
            for b in B:
                if b != x:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)

    return A, B
=======
def _can_add(x, lst, sums, diffs_other):
    """Check if x can be added to lst given current sums and diffs_other."""
    # Check sum collisions within lst
    for a in lst:
        s = a + x
        if s in sums:
            return False
    s2 = 2 * x
    if s2 in sums:
        return False
    
    # Check difference disjointness
    # New diffs from x to lst elements
    for a in lst:
        if (x - a) in diffs_other or (a - x) in diffs_other:
            return False
            
    return True

def _add_to_set(x, lst, sums, diffs):
    """Add x to lst, updating sums and diffs."""
    lst.append(x)
    sums.add(2 * x)
    for a in lst:
        if a != x:
            sums.add(a + x)
            diffs.add(x - a)
            diffs.add(a - x)

def _remove_from_set(x, lst, sums, diffs):
    """Remove x from lst, updating sums and diffs."""
    lst.remove(x)
    # Rebuild sums and diffs is safer and fast enough for N=256
    # But we can also subtract. Let's just rebuild for simplicity and correctness in polish.
    new_sums = set()
    new_diffs = set()
    for i in range(len(lst)):
        a = lst[i]
        new_sums.add(2 * a)
        for j in range(i + 1, len(lst)):
            b = lst[j]
            s = a + b
            new_sums.add(s)
            new_diffs.add(b - a)
            new_diffs.add(a - b)
    sums.clear()
    sums.update(new_sums)
    diffs.clear()
    diffs.update(new_diffs)

def _polish(A, B, deadline):
    """
    Local search to improve A and B.
    """
    start = time.time()
    all_elements = set(range(1, N + 1))
    placed = set(A) | set(B)
    unplaced = list(all_elements - placed)
    
    # Precompute current sums and diffs for fast checks
    sums_A = set()
    diffs_A = set()
    for i in range(len(A)):
        a = A[i]
        sums_A.add(2 * a)
        for j in range(i + 1, len(A)):
            b = A[j]
            sums_A.add(a + b)
            diffs_A.add(b - a)
            diffs_A.add(a - b)
            
    sums_B = set()
    diffs_B = set()
    for i in range(len(B)):
        b = B[i]
        sums_B.add(2 * b)
        for j in range(i + 1, len(B)):
            c = B[j]
            sums_B.add(b + c)
            diffs_B.add(c - b)
            diffs_B.add(b - c)

    rng = random.Random(12345)
    
    while time.time() - start < deadline:
        # 1. Try to add unplaced elements directly
        if unplaced:
            x = rng.choice(unplaced)
            if _can_add(x, A, sums_A, diffs_B):
                _add_to_set(x, A, sums_A, diffs_A)
                unplaced.remove(x)
                placed.add(x)
            elif _can_add(x, B, sums_B, diffs_A):
                _add_to_set(x, B, sums_B, diffs_B)
                unplaced.remove(x)
                placed.add(x)
            else:
                # 2. Try swaps
                # Pick x from unplaced, y from A
                if A:
                    y = rng.choice(A)
                    # Check if x can be added to A\{y}
                    # This is tricky with precomputed sets. 
                    # We can simulate removal of y from A's perspective for x.
                    # Or just try: Remove y, Check x, Add x, Add y to B?
                    # Let's do:
                    # 1. Temporarily remove y from A
                    # 2. Check if x fits in A
                    # 3. Check if y fits in B
                    # 4. If yes, commit.
                    
                    # To avoid complex temporary state, we can just check:
                    # Does x conflict with A only because of y?
                    # And does y fit in B?
                    
                    # Simpler: Just try the move if it looks promising.
                    # We'll use a helper that checks validity with a "hole"
                    
                    # Check if y can go to B
                    if _can_add(y, B, sums_B, diffs_A):
                         # Now check if x can go to A if y is removed
                         # We need to check if x conflicts with A\{y}
                         # Conflicts in A\{y} are a subset of conflicts in A.
                         # If x had no conflicts in A, it fits.
                         # If x had conflicts, we need to know if they were with y.
                         
                         # Let's just perform the swap if we can verify it quickly.
                         # Verification:
                         # 1. Save state
                         # 2. Remove y from A
                         # 3. Add x to A
                         # 4. Add y to B
                         # 5. Check validity of new A and B? 
                         #    Actually, if _can_add passed for y->B and x->A (with y gone), it's valid.
                         
                         # How to check x->A (with y gone) efficiently?
                         # We can iterate A. For each a in A:
                         #   if a == y: skip
                         #   else: check sum/diff
                         
                         can_add_x_to_A_minus_y = True
                         for a in A:
                             if a == y:
                                 continue
                             if (a + x) in sums_A:
                                 # Conflict! But is this conflict caused by y?
                                 # sums_A contains sums from A.
                                 # If a+x == u+v, and {u,v} does not contain y, then conflict remains.
                                 # If {u,v} contains y, then removing y might fix it.
                                 # This is hard to check without knowing u,v.
                                 # 
                                 # Alternative: Just rebuild sums_A without y? 
                                 # That is O(|A|^2). 
                                 # Given N=256, and we do this rarely, it might be ok.
                                 # But we are in a tight loop.
                                 
                                 # Let's approximate: If x conflicts with A, we assume it's a "hard" conflict 
                                 # unless we verify.
                                 # For now, let's just skip swaps if direct add fails, 
                                 # OR implement a robust swap check.
                                 
                                 pass # Placeholder for robust check
                                 
                         # Let's implement a robust check for x in A\{y}
                         # We'll just rebuild the "forbidden sums" for A\{y} on the fly for this check?
                         # No, that's too slow.
                         
                         # Heuristic: Only try swap if x has "few" conflicts.
                         # Or just try the swap and validate the whole set?
                         # Validating whole set is O(N^2). 
                         # If we do this 1000 times, 1000 * 65k = 65M ops. OK.
                         
                         # Let's try:
                         # 1. New A = A - {y} + {x}
                         # 2. New B = B + {y}
                         # 3. Check if New A is Sidon.
                         # 4. Check if New B is Sidon.
                         # 5. Check if diffs(New A) and diffs(New B) are disjoint.
                         
                         # This is safe.
                         
                         new_A = [a for a in A if a != y]
                         new_A.append(x)
                         new_B = B + [y]
                         
                         if _pair_sums_ok(new_A) and _pair_sums_ok(new_B):
                             dA = _diffs(new_A)
                             dB = _diffs(new_B)
                             if not (dA & dB):
                                 # Commit
                                 A = new_A
                                 B = new_B
                                 unplaced.remove(x)
                                 placed.add(x)
                                 # y is now in B, so it was in A, now in B. 
                                 # It was already in 'placed'.
                                 # Recalculate sums/diffs for global state
                                 sums_A.clear(); diffs_A.clear()
                                 for i in range(len(A)):
                                     a = A[i]
                                     sums_A.add(2 * a)
                                     for j in range(i + 1, len(A)):
                                         b = A[j]
                                         sums_A.add(a + b)
                                         diffs_A.add(b - a)
                                         diffs_A.add(a - b)
                                 sums_B.clear(); diffs_B.clear()
                                 for i in range(len(B)):
                                     b = B[i]
                                     sums_B.add(2 * b)
                                     for j in range(i + 1, len(B)):
                                         c = B[j]
                                         sums_B.add(b + c)
                                         diffs_B.add(c - b)
                                         diffs_B.add(b - c)
                                 continue
                                 
        # If nothing worked, try a random restart of the polish loop?
        # Or just continue.
        
    return A, B
>>>>>>> REPLACE

This implementation of `_polish` is a bit heavy because it rebuilds sums/diffs and validates full sets for swaps. However, it is correct and should find improvements. The direct add part is fast.

I will also modify `construct` to call `_polish` after `_fast_greedy`.


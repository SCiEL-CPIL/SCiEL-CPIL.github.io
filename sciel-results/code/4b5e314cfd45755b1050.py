import time

def _check_addition(Sset, elems, new_elem):
    """Check if adding new_elem to Sset (with sorted elems) creates a forbidden triple.
    Only need to check triples where new_elem is one of a,b,c."""
    n = len(elems)
    # Case 1: new_elem is c (largest)
    # Check all pairs (a,b) in elems: if a+b, a+new_elem, b+new_elem all in Sset
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            if a + b in Sset and a + new_elem in Sset and b + new_elem in Sset:
                return False
    # Case 2: new_elem is b (middle)
    # Check pairs (a, c) where a < new_elem < c
    for i in range(n):
        a = elems[i]
        if a >= new_elem:
            break
        for k in range(n):
            c = elems[k]
            if c <= new_elem:
                continue
            if a + new_elem in Sset and a + c in Sset and new_elem + c in Sset:
                return False
    # Case 3: new_elem is a (smallest)
    # Check pairs (b, c) where new_elem < b < c
    for j in range(n):
        b = elems[j]
        if b <= new_elem:
            continue
        for k in range(j+1, n):
            c = elems[k]
            if new_elem + b in Sset and new_elem + c in Sset and b + c in Sset:
                return False
    return True

def _valid_full(Sset):
    """Full validity check for a set."""
    elems = sorted(Sset)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = elems[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def construct():
    deadline = time.time() + 85.0
    # Start from the strong sufficient-condition interval: {64..128}, size 65
    S = set(range(64, 129))
    elems = sorted(S)
    best = set(S)
    best_size = len(S)
    
    # Greedily try to add each small element 1..63 if validity preserved
    for cand in range(1, 64):
        if cand in S:
            continue
        if _check_addition(S, elems, cand):
            S.add(cand)
            elems.append(cand)
            elems.sort()
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
    
    # Local search: try adding pairs of small elements
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for x in range(1, 64):
            if x in S:
                continue
            for y in range(x+1, 64):
                if y in S:
                    continue
                # Check if adding x first, then y works
                if _check_addition(S, elems, x):
                    S.add(x)
                    elems.append(x)
                    elems.sort()
                    if _check_addition(S, elems, y):
                        S.add(y)
                        elems.append(y)
                        elems.sort()
                        if len(S) > best_size:
                            best = set(S)
                            best_size = len(S)
                        changed = True
                    else:
                        # Undo adding x
                        S.discard(x)
                        elems = sorted(S)
    
    # Try individual additions again after pair additions
    for cand in range(1, 64):
        if cand in S:
            continue
        if _check_addition(S, elems, cand):
            S.add(cand)
            elems.append(cand)
            elems.sort()
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
    
    # Final verification
    if not _valid_full(best):
        best = set(range(64, 129))
    
    # Ensure within deadline
    if time.time() > deadline - 1.0:
        best = set(range(64, 129))
    
    result = sorted(best)
    result = [int(x) for x in result]
    return result

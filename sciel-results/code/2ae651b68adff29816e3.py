import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 180
    # Precompute all reciprocal Schur triples (a,b,c) distinct in [1,180]
    # where 1/a = 1/b + 1/c, i.e., a*(b+c) = b*c
    # a = bc/(b+c), so bc must be divisible by (b+c)
    
    # Find all valid triples
    triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab)
            # c = ab/(b-a)
            if b > a:
                num = a * b
                den = b - a
                if num % den == 0:
                    c = num // den
                    if c > b and c <= n:
                        triples.append((a, b, c))
    
    # Build conflict: for each number, which other pairs form a triple with it
    # We want max subset with no triple fully contained
    
    # Greedy: try to include as many as possible
    # Start with all numbers, remove minimum to break all triples
    
    # Use a simple approach: greedy set packing
    # Try to build a large set by adding numbers one by one
    
    # Better: use the fact that if we exclude small numbers, we avoid most triples
    # Let's try: take all numbers from k to 180, and see what k minimizes exclusions
    
    # Actually, let's do a smarter greedy:
    # Sort numbers by "constraint degree" (how many triples they're in)
    # Add numbers in order of least constraint first
    
    # Count how many triples each number participates in
    triple_count = [0] * (n+1)
    for (a, b, c) in triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Greedy: process numbers from 1 to 180, add if possible
    # But we want to maximize size, so maybe process in order of least constraint
    
    # Alternative: start with all numbers, and for each triple, remove one element
    # Use a matching-like approach
    
    # Let's try: include all numbers > 90, then try to add smaller ones
    selected = set(range(91, 181))
    
    # Try to add numbers from 1 to 90
    for num in range(1, 91):
        # Check if adding num creates a triple
        can_add = True
        for (a, b, c) in triples:
            if num in (a, b, c) and (a, b, c) != (num, num, num):
                # Check if the other two are in selected
                others = [x for x in (a, b, c) if x != num]
                if len(others) == 2 and others[0] in selected and others[1] in selected:
                    can_add = False
                    break
        if can_add:
            selected.add(num)
    
    # Try to add more by removing some existing elements
    # Local search: try to swap
    for num in range(1, 91):
        if num not in selected:
            # Try removing one element from selected to allow adding num
            for (a, b, c) in triples:
                if num in (a, b, c):
                    others = [x for x in (a, b, c) if x != num]
                    if len(others) == 2 and others[0] in selected and others[1] in selected:
                        # Try removing one of others
                        for rem in others:
                            new_selected = selected - {rem}
                            new_selected.add(num)
                            # Verify no triple in new_selected
                            valid = True
                            for (x, y, z) in triples:
                                if x in new_selected and y in new_selected and z in new_selected:
                                    valid = False
                                    break
                            if valid:
                                selected = new_selected
                                break
                        if num in selected:
                            break
    
    # Final verification
    for (a, b, c) in triples:
        if a in selected and b in selected and c in selected:
            # Remove one to fix
            selected.discard(c)
    
    return sorted(selected)

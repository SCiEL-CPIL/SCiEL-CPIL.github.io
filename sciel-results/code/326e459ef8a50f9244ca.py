def construct():
    n = 256
    # Start with full set
    S = set(range(1, n+1))
    
    def has_4ap(S, x):
        # Check if x in S creates a 4-AP with other elements in S
        # x can be at position 0,1,2,3 in the 4-AP
        # Position 0: x, x+d, x+2d, x+3d all in S
        # Position 1: x-d, x, x+d, x+2d all in S
        # Position 2: x-2d, x-d, x, x+d all in S
        # Position 3: x-3d, x-2d, x-d, x all in S
        for d in range(1, n+1):
            # x is first
            if x+d in S and x+2*d in S and x+3*d in S:
                return True
            # x is second
            if x-d in S and x+d in S and x+2*d in S and x-d >= 1:
                return True
            # x is third
            if x-2*d in S and x-d in S and x+d in S and x-2*d >= 1:
                return True
            # x is fourth
            if x-3*d in S and x-2*d in S and x-d in S and x-3*d >= 1:
                return True
        return False
    
    def count_3aps(S):
        count = 0
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                c = S_list[j]
                mid = (a + c) / 2
                if mid == int(mid) and mid in S:
                    count += 1
        return count
    
    def three_ap_degree(S, x):
        # Count number of 3-APs that x participates in
        count = 0
        for d in range(1, n+1):
            # x is middle: x-d, x, x+d
            if x-d in S and x+d in S and x-d >= 1:
                count += 1
            # x is first: x, x+d, x+2d
            if x+d in S and x+2*d in S:
                count += 1
            # x is last: x-2d, x-d, x
            if x-2*d in S and x-d in S and x-2*d >= 1:
                count += 1
        return count
    
    # Phase 1: Remove elements to eliminate all 4-APs
    # Find all 4-APs and remove the least valuable element
    changed = True
    while changed:
        changed = False
        # Find a 4-AP
        found = None
        for a in range(1, n+1):
            for d in range(1, (n-a)//3 + 1):
                b = a+d
                c = a+2*d
                e = a+3*d
                if a in S and b in S and c in S and e in S:
                    found = (a, b, c, e)
                    break
            if found:
                break
        if found:
            a, b, c, e = found
            # Remove the element with lowest 3-AP degree
            degrees = [(three_ap_degree(S, x), x) for x in [a, b, c, e]]
            degrees.sort()
            S.remove(degrees[0][1])
            changed = True
    
    # Phase 2: Try to add back removed elements
    removed = set(range(1, n+1)) - S
    for x in sorted(removed):
        # Check if adding x creates a 4-AP
        if not has_4ap(S, x):
            # Check if it increases 3-AP count
            old_count = count_3aps(S)
            S.add(x)
            new_count = count_3aps(S)
            if new_count >= old_count:
                pass  # Keep it
            else:
                S.remove(x)
    
    return sorted(S)

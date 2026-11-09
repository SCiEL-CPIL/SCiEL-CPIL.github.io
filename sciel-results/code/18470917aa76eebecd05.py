def construct():
    n = 256
    
    # Precompute all 4-APs and 3-APs
    all_4aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    all_3aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            all_3aps.append((a, a + d, a + 2 * d))
    
    # Track 3-AP counts per element and which 3-APs each element belongs to
    elem_3ap_count = [0] * (n + 1)
    elem_3aps = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(all_3aps):
        elem_3ap_count[a] += 1
        elem_3ap_count[b] += 1
        elem_3ap_count[c] += 1
        elem_3aps[a].append(i)
        elem_3aps[b].append(i)
        elem_3aps[c].append(i)
    
    # Track which 4-APs each element belongs to
    elem_4aps = [[] for _ in range(n + 1)]
    for i, ap in enumerate(all_4aps):
        for x in ap:
            elem_4aps[x].append(i)
    
    A = set(range(1, n + 1))
    active_4aps = [True] * len(all_4aps)
    
    # Greedy removal: while a 4-AP exists in A, remove the element from that 4-AP
    # with the fewest 3-APs (to minimize 3-AP loss)
    while True:
        target_ap = -1
        for idx in range(len(all_4aps)):
            if active_4aps[idx] and all(x in A for x in all_4aps[idx]):
                target_ap = idx
                break
        
        if target_ap == -1:
            break
        
        ap = all_4aps[target_ap]
        min_count = float('inf')
        min_elem = None
        for x in ap:
            if elem_3ap_count[x] < min_count:
                min_count = elem_3ap_count[x]
                min_elem = x
        
        A.remove(min_elem)
        
        # Deactivate all 4-APs containing min_elem
        for idx in elem_4aps[min_elem]:
            active_4aps[idx] = False
        
        # Update 3-AP counts for neighbors
        for idx in elem_3aps[min_elem]:
            a, b, c = all_3aps[idx]
            for y in (a, b, c):
                if y != min_elem:
                    elem_3ap_count[y] -= 1
    
    return sorted(A)

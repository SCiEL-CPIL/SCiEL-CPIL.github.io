def min_span(m):
    # Precompute valid residues for each i up to 144
    residues = [[] for _ in range(13)]
    for i in range(1, 13):
        c = (-m) % i
        if c == 0:
            r = i
        else:
            r = c
        while r <= 144:
            residues[i].append(r)
            r += i

    # L must be at least 12 because we need 12 distinct integers
    for L in range(12, 145):
        # Build adjacency lists for this L
        adj = [[] for _ in range(13)]
        for i in range(1, 13):
            # Filter residues <= L
            # Since residues[i] is sorted, we can break early
            # But list is small, so simple list comprehension is fine
            adj[i] = [r for r in residues[i] if r <= L]

        # Quick check: if any i has no candidates, fail
        if any(len(adj[i]) == 0 for i in range(1, 13)):
            continue

        # Matching using DFS with bitmasks
        match = [-1] * (L + 1) # r -> i

        # Heuristic: process i with smallest degree first
        indices = list(range(1, 13))
        indices.sort(key=lambda x: len(adj[x]))

        def dfs(i, visited):
            for r in adj[i]:
                bit = 1 << (r - 1)
                if visited & bit:
                    continue
                visited |= bit
                if match[r] == -1 or dfs(match[r], visited):
                    match[r] = i
                    return True
            return False

        success = True
        for i in indices:
            if not dfs(i, 0):
                success = False
                break

        if success:
            return L
    return 144 # Fallback

def construct():
    best_m = 0
    best_val = 0
    P = 27720 # LCM(1..12)
    # Search the full period
    for m in range(P):
        val = min_span(m)
        if val > best_val:
            best_val = val
            best_m = m
            if best_val == 144:
                break
    return best_m

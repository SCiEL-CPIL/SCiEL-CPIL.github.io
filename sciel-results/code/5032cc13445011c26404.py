import time

P = 27720  # LCM(1..12)
MAX_L = 144

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    # Precompute adjacency masks for all m
    # adj_masks[m][i] = bitmask of j in [0, MAX_L) where (m+1+j) % (i+1) == 0
    adj_masks = []
    for m in range(P):
        base = m + 1
        masks = []
        for i in range(12):
            div = i + 1
            target = (-base) % div
            mask = 0
            j = target
            while j < MAX_L:
                mask |= (1 << j)
                j += div
            masks.append(mask)
        adj_masks.append(masks)

    def has_matching(m_idx, L):
        mask_limit = (1 << L) - 1
        adj = [adj_masks[m_idx][i] & mask_limit for i in range(12)]
        for i in range(12):
            if adj[i] == 0:
                return False

        match_to = [-1] * L

        def dfs(i, visited):
            mask = adj[i]
            while mask:
                lb = mask & (-mask)
                j = lb.bit_length() - 1
                mask ^= lb
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False

        for i in range(12):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    best_L = 0

    for m in range(P):
        if time.time() > deadline:
            break

        lo, hi = 12, MAX_L
        min_L = MAX_L + 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if min_L > best_L:
            best_L = min_L
            best_m = m
            if best_L == MAX_L:
                break

    return best_m

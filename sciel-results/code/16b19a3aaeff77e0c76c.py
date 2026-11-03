import itertools
import numpy as np

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # 1. Precompute Overlap Matrix
    # Map prefix tuples to list of indices
    # prefix_map[k] = { tuple: [indices] }
    prefix_map = [dict() for _ in range(n)]
    for j, p in enumerate(perms):
        for k in range(1, n):
            prefix = tuple(p[:k])
            if prefix not in prefix_map[k]:
                prefix_map[k][prefix] = []
            prefix_map[k][prefix].append(j)
            
    ov = np.zeros((N, N), dtype=np.uint8)
    for i, p in enumerate(perms):
        for k in range(n - 1, 0, -1):
            suffix = tuple(p[n - k:])
            if suffix in prefix_map[k]:
                for j in prefix_map[k][suffix]:
                    if ov[i, j] == 0:
                        ov[i, j] = k
                        # We could break if we assume we want max, 
                        # but we are iterating k descending, so first set is max.
                        # However, we need to fill the whole row.
                        # The loop over j sets ov[i, j].
                        # We don't break the k-loop because we need to check smaller k 
                        # for other j's that didn't match larger k.
                        # Wait, if j matched k=6, it won't match k=5 in a way that 
                        # increases overlap. So we can skip j if ov[i, j] > 0?
                        # Yes.
    
    # Optimization: skip if already set
    for i, p in enumerate(perms):
        for k in range(n - 1, 0, -1):
            suffix = tuple(p[n - k:])
            if suffix in prefix_map[k]:
                for j in prefix_map[k][suffix]:
                    if ov[i, j] == 0:
                        ov[i, j] = k

    # 2. Greedy Construction (Nearest Neighbor)
    unvisited = set(range(N))
    unvisited.remove(0)
    path = [0]
    current = 0
    
    for _ in range(N - 1):
        # Find j in unvisited with max ov[current, j]
        best_j = -1
        best_ov = -1
        for j in unvisited:
            o = ov[current, j]
            if o > best_ov:
                best_ov = o
                best_j = j
        path.append(best_j)
        unvisited.remove(best_j)
        current = best_j

    # 3. Local Search: 1-Opt (Swap Adjacent)
    def get_cost(i, j):
        return n - ov[i, j]

    improved = True
    passes = 0
    while improved and passes < 100: # Limit passes
        improved = False
        passes += 1
        for i in range(N - 1):
            # Consider swapping path[i] and path[i+1]
            x = path[i]
            y = path[i + 1]
            
            # Neighbors
            a = path[i - 1] if i > 0 else -1
            b = path[i + 2] if i + 2 < N else -1
            
            old_cost = 0
            new_cost = 0
            
            if a != -1:
                old_cost += get_cost(a, x)
                new_cost += get_cost(a, y)
            
            old_cost += get_cost(x, y)
            new_cost += get_cost(y, x)
            
            if b != -1:
                old_cost += get_cost(y, b)
                new_cost += get_cost(x, b)
                
            if new_cost < old_cost:
                path[i], path[i + 1] = path[i + 1], path[i]
                improved = True

    # 4. Local Search: 2-Opt (Reverse Segment) - Optional if 1-Opt is slow or insufficient
    # Let's try a few 2-Opt passes if time permits, but 1-Opt might be enough.
    # Given the gap is small, 1-Opt might suffice.
    
    # Calculate final length
    final_len = n
    for i in range(N - 1):
        final_len += get_cost(path[i], path[i + 1])
        
    # Compare with classic
    word_classic = [1]
    for m in range(2, n + 1):
        perms_local, seen = [], set()
        for i in range(len(word_classic) - m + 2):
            window = tuple(word_classic[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_local.append(window)
        new = []
        for p in perms_local:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_classic = new
    classic_len = len(word_classic)

    if final_len < classic_len:
        res = perm_strs[path[0]]
        for i in range(1, N):
            ov_val = ov[path[i - 1], path[i]]
            res += perm_strs[path[i]][ov_val:]
        return res
    else:
        return "".join(map(str, word_classic))

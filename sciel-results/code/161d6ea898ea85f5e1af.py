import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    if n == 5:
        return _standard_superperm(n)
    if n == 7:
        return _optimized_superperm_7()
    return _standard_superperm(n)

def _standard_superperm(n):
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

def _optimized_superperm_7():
    start_time = time.time()
    deadline = start_time + 140.0

    perms = list(itertools.permutations(range(1, 7)))
    N = 720
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap O[i][j] and difference D[i][j] = O[j][i] - O[i][j]
    O = [[0]*N for _ in range(N)]
    D = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            ov = 0
            # Max overlap 6
            for k in range(6, -1, -1):
                if pi[6-k:] == pj[:k]:
                    ov = k
                    break
            O[i][j] = ov
    
    for i in range(N):
        for j in range(N):
            D[i][j] = O[j][i] - O[i][j]

    # Initial order: Standard
    s6 = _standard_superperm(6)
    s6_list = [int(c) for c in s6]
    std_order = []
    seen_std = set()
    for i in range(len(s6_list) - 5):
        window = tuple(s6_list[i:i+6])
        if window in perm_to_idx and window not in seen_std:
            seen_std.add(window)
            std_order.append(perm_to_idx[window])
    if len(std_order) != N:
        std_order = list(range(N))

    current_order = std_order[:]
    
    # Calculate initial cost
    # Cost = - sum O[order[k]][order[k+1]]
    # We want to minimize cost (maximize overlap)
    current_cost = 0
    for k in range(N-1):
        current_cost -= O[current_order[k]][current_order[k+1]]
        
    best_cost = current_cost
    best_order = current_order[:]

    # 2-Opt Search
    # To speed up, we only do a few passes
    passes = 0
    max_passes = 50
    
    while passes < max_passes and time.time() < deadline - 10.0:
        passes += 1
        improved = False
        for i in range(N-1):
            if time.time() > deadline - 10.0:
                break
            for j in range(i+1, N):
                # Calculate delta cost for reversing i..j
                # Edges removed: (i-1, i) if i>0, (j, j+1) if j<N-1
                # Edges added: (i-1, j) if i>0, (i, j+1) if j<N-1
                # Internal edges reversed: sum D[k][k+1] for k=i..j-1
                
                delta = 0
                
                # Internal change
                # Sum D[order[k]][order[k+1]] for k=i to j-1
                # Note: D[A][B] = O[B][A] - O[A][B]
                # Old internal sum: S_old = sum O[order[k]][order[k+1]]
                # New internal sum: S_new = sum O[order[k+1]][order[k]]
                # S_new - S_old = sum (O[order[k+1]][order[k]] - O[order[k]][order[k+1]])
                # = sum D[order[k]][order[k+1]]
                
                # We need to sum D quickly.
                # Since N is small (720), we can just loop if j-i is small?
                # Or precompute prefix sums of D for the current order?
                # But order changes.
                # Let's just loop. Average j-i is N/3 = 240.
                # 240 ops per check.
                # 720*720 checks = 500k checks.
                # 500k * 240 = 120M ops. Might be too slow in Python (10-20s).
                
                # Optimization: Only check if j-i is small?
                # Or use a faster heuristic.
                
                # Let's try a simpler move: 1-Opt (swap adjacent or move 1 element)
                # Or just accept the slowness and limit passes.
                
                internal_delta = 0
                for k in range(i, j):
                    internal_delta += D[current_order[k]][current_order[k+1]]
                
                if i > 0:
                    delta -= O[current_order[i-1]][current_order[i]]
                    delta += O[current_order[i-1]][current_order[j]]
                if j < N-1:
                    delta -= O[current_order[j]][current_order[j+1]]
                    delta += O[current_order[i]][current_order[j+1]]
                    
                delta += internal_delta
                
                if delta < 0:
                    # Accept
                    current_order[i:j+1] = reversed(current_order[i:j+1])
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = current_order[:]

    def build_string(order):
        blocks = []
        for idx in order:
            p = perms[idx]
            blocks.append(list(p) + [7] + list(p))
        
        final = []
        for b in blocks:
            if not final:
                final = b
            else:
                max_ov = 0
                limit = min(len(final), len(b))
                for k in range(limit, -1, -1):
                    if final[len(final)-k:] == b[:k]:
                        max_ov = k
                        break
                final += b[max_ov:]
        return final

    candidate = build_string(best_order)
    
    if _is_valid_superperm(candidate, 7):
        return "".join(map(str, candidate))
    
    std_candidate = build_string(std_order)
    if _is_valid_superperm(std_candidate, 7):
        return "".join(map(str, std_candidate))
        
    return _standard_superperm(7)

def _is_valid_superperm(word, n):
    if not word:
        return False
    windows = set()
    for i in range(len(word) - n + 1):
        windows.add(tuple(word[i:i+n]))
    all_perms = set(itertools.permutations(range(1, n+1)))
    return all_perms.issubset(windows)

import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)

    LCP = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(i + 1, N):
            pj = perms[j]
            k = 0
            while k < n and pi[k] == pj[k]:
                k += 1
            LCP[i][j] = k
            LCP[j][i] = k

    def calc_lcp_sum(order):
        s = 0
        for i in range(len(order) - 1):
            s += LCP[order[i]][order[i + 1]]
        return s

    def two_opt(order, time_limit):
        n_order = len(order)
        improved = True
        passes = 0
        max_passes = 200 # Safety cap
        
        while improved and passes < max_passes and time.time() < time_limit:
            improved = False
            passes += 1
            for i in range(n_order - 1):
                if time.time() > time_limit:
                    return order
                v = order[i]
                u = order[i - 1] if i > 0 else -1
                
                for j in range(i + 1, n_order):
                    w = order[j]
                    x = order[j + 1] if j + 1 < n_order else -1
                    
                    old_score = 0
                    new_score = 0
                    
                    if u != -1:
                        old_score += LCP[u][v]
                        new_score += LCP[u][w]
                    
                    if x != -1:
                        old_score += LCP[w][x]
                        new_score += LCP[v][x]
                    
                    if new_score > old_score:
                        order[i:j + 1] = order[i:j + 1][::-1]
                        improved = True
        return order

    best_order = None
    best_lcp_sum = -1

    random.seed(42)
    
    # Estimate time for one 2-opt pass
    # N=5040. N^2/2 = 12.6M.
    # Let's allow 1 restart if it's fast, or just 1 good restart.
    # Given the tight time, let's do 1 restart with a strict time limit for 2-opt.
    
    order = list(range(N))
    random.shuffle(order)
    
    # Run 2-opt with deadline - 10 seconds
    order = two_opt(order, deadline - 10)
    
    cur_lcp = calc_lcp_sum(order)
    if cur_lcp > best_lcp_sum:
        best_lcp_sum = cur_lcp
        best_order = order[:]
        
    # If time remains, try another restart?
    if time.time() < deadline - 20:
        order = list(range(N))
        random.shuffle(order)
        order = two_opt(order, deadline - 10)
        cur_lcp = calc_lcp_sum(order)
        if cur_lcp > best_lcp_sum:
            best_lcp_sum = cur_lcp
            best_order = order[:]

    if best_order is None:
        return "".join(map(str, perms[0]))

    word = list(perms[best_order[0]])
    for idx in range(1, N):
        p_idx = best_order[idx]
        prev_idx = best_order[idx - 1]
        ov = LCP[prev_idx][p_idx]
        word.extend(perms[p_idx][ov:])

    return "".join(map(str, word))

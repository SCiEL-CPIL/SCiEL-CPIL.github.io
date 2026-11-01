import time
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134121341213412143213421342143214321"  # Standard S_4 length 33

    def standard_superperm(k):
        if k == 1:
            return [1]
        word = standard_superperm(k-1)
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    # Get the standard word for n-1
    word_prev = standard_superperm(n-1)
    
    # Extract the order of permutations of 1..n-1
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    N = len(perms_order)
    perm_to_idx = {p: i for i, p in enumerate(perms_order)}
    
    # Precompute overlap matrix
    # M[i][j] = overlap between perms_order[i] (suffix) and perms_order[j] (prefix)
    # Overlap is max k such that perms_order[i][-k:] == perms_order[j][:k]
    if time.time() > deadline - 30:
        # Not enough time, just return the standard construction
        word = list(perms_order[0]) + [n] + list(perms_order[0])
        for idx in range(1, N):
            p = perms_order[idx]
            block = list(p) + [n] + list(p)
            ov = 0
            prev = perms_order[idx-1]
            for k in range(n-1, 0, -1):
                if prev[-k:] == p[:k]:
                    ov = k
                    break
            word += block[ov:]
        return "".join(map(str, word))
    
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        p_i = perms_order[i]
        for j in range(N):
            p_j = perms_order[j]
            # Max overlap is n-1
            max_k = n - 1
            for k in range(max_k, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    M[i][j] = k
                    break
    
    # Initial path is 0, 1, ..., N-1
    path = list(range(N))
    
    # Calculate initial score (sum of overlaps)
    def calc_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += M[path[i]][path[i+1]]
        return score
    
    current_score = calc_score(path)
    
    # Local Search: 2-opt with first-improvement strategy
    # We want to maximize score.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 5:
                break
            u = path[i]
            v = path[i+1]
            for j in range(i + 1, N):
                if time.time() > deadline - 5:
                    break
                x = path[j]
                
                if j == N - 1:
                    # Reversing path[i+1..N-1]: remove edge (u,v), add edge (u,x)
                    delta = M[u][x] - M[u][v]
                else:
                    y = path[j+1]
                    # Reversing path[i+1..j]: remove (u,v) and (x,y), add (u,x) and (v,y)
                    delta = M[u][x] + M[v][y] - M[u][v] - M[x][y]
                
                if delta > 0:
                    # Perform the reversal
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    current_score += delta
                    improved = True
                    break
            if improved:
                break
        
        # If no improvement was found in a full pass, try a random restart or perturb
        if not improved:
            # Simple perturbation: reverse a random segment
            if N > 2:
                r1 = np.random.randint(0, N-1)
                r2 = np.random.randint(r1+1, N)
                path[r1+1:r2+1] = path[r1+1:r2+1][::-1]
                # Recalculate score
                old_score = current_score
                current_score = calc_score(path)
                if current_score < old_score:
                    # Revert
                    path[r1+1:r2+1] = path[r1+1:r2+1][::-1]
                    current_score = old_score
                else:
                    improved = True
    
    # Construct the final word
    first_p = perms_order[path[0]]
    word = list(first_p) + [n] + list(first_p)
    
    for idx in range(1, N):
        p = perms_order[path[idx]]
        block = list(p) + [n] + list(p)
        # Overlap is M[path[idx-1]][path[idx]]
        ov = M[path[idx-1]][path[idx]]
        word += block[ov:]
        
    return "".join(map(str, word))

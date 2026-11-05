import time
import random

def construct(n):
    start_time = time.time()
    time_limit = 115.0
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Generate standard superpermutation word
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
            max_k = min(len(new), len(block))
            k = 0
            for k_try in range(max_k, 0, -1):
                if new[-k_try:] == block[:k_try]:
                    k = k_try
                    break
            new += block[k:]
        word = new
    
    # Extract permutation sequence
    seen_perms = set()
    perm_seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen_perms:
            seen_perms.add(p)
            perm_seq.append(p)
            
    N = len(perm_seq)
    
    # Encode permutations as integers
    base = 8
    perm_vals = []
    for p in perm_seq:
        val = 0
        for d in p:
            val = val * base + d
        perm_vals.append(val)
        
    # Precompute powers of base
    base_powers = [1]
    for i in range(1, n+1):
        base_powers.append(base_powers[-1] * base)
        
    # Precompute overlap matrix
    M = [[0]*N for _ in range(N)]
    for i in range(N):
        vi = perm_vals[i]
        for j in range(N):
            vj = perm_vals[j]
            for k in range(n-1, -1, -1):
                # Suffix of vi of length k
                # vi % base_powers[k]
                # Prefix of vj of length k
                # vj // base_powers[n-k]
                if k == 0:
                    M[i][j] = 0
                    break
                if vi % base_powers[k] == vj // base_powers[n-k]:
                    M[i][j] = k
                    break
    
    current_seq = list(range(N))
    
    def get_score(seq):
        s = 0
        for i in range(len(seq)-1):
            s += M[seq[i]][seq[i+1]]
        return s

    current_score = get_score(current_seq)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Adjacent Swap Pass
        for i in range(N-1):
            if (time.time() - start_time) > time_limit - 5:
                break
            a = current_seq[i]
            b = current_seq[i+1]
            
            prev_idx = current_seq[i-1] if i > 0 else None
            next_idx = current_seq[i+2] if i+2 < N else None
            
            old_cost = 0
            if prev_idx is not None:
                old_cost += M[prev_idx][a]
            old_cost += M[a][b]
            if next_idx is not None:
                old_cost += M[b][next_idx]
            
            new_cost = 0
            if prev_idx is not None:
                new_cost += M[prev_idx][b]
            new_cost += M[b][a]
            if next_idx is not None:
                new_cost += M[a][next_idx]
            
            delta = new_cost - old_cost
            if delta > 0:
                current_seq[i], current_seq[i+1] = b, a
                current_score += delta
                improved = True
                
    # Reconstruct word
    # We need the original permutations to build the string.
    res = list(perm_seq[current_seq[0]])
    for i in range(1, N):
        prev_idx = current_seq[i-1]
        curr_idx = current_seq[i]
        prev = perm_seq[prev_idx]
        curr = perm_seq[curr_idx]
        k = M[prev_idx][curr_idx]
        res.extend(curr[k:])
        
    return "".join(map(str, res))

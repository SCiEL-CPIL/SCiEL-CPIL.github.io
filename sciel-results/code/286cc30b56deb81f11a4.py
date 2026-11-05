import time
import random

def construct(n):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
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
    
    # Extract order of first appearance of each permutation
    seen_perms = set()
    perm_seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen_perms:
            seen_perms.add(p)
            perm_seq.append(p)
    
    if len(perm_seq) != 5040:
        return "".join(map(str, word))
    
    N = len(perm_seq)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perm_seq[i][n-k:] == perm_seq[j][:k]
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perm_seq[i]
        for j in range(N):
            pj = perm_seq[j]
            # Check from n-1 down to 0
            for k in range(n - 1, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if pi[n-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Compute initial score (sum of overlaps)
    def compute_score(seq):
        total = 0
        for i in range(len(seq) - 1):
            total += overlap[seq[i]][seq[i+1]]
        return total
    
    current_seq = list(perm_seq)
    current_score = compute_score(current_seq)
    
    # 2-opt local search: reverse segment [i, j]
    # Old edges affected: (i-1,i), (i,i+1), ..., (j-1,j), (j,j+1)
    # New edges: (i-1,j), (j,j-1), ..., (i+1,i), (i,j+1)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 10:
        improved = False
        
        # Try all 2-opt moves
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 5:
                break
            for j in range(i + 1, N):
                # Calculate delta
                # Removed: overlap[seq[i-1]][seq[i]] (if i>0), 
                #          overlap[seq[i]][seq[i+1]], ..., overlap[seq[j-1]][seq[j]],
                #          overlap[seq[j]][seq[j+1]] (if j<N-1)
                # Added: overlap[seq[i-1]][seq[j]] (if i>0),
                #        overlap[seq[j]][seq[j-1]], ..., overlap[seq[i+1]][seq[i]],
                #        overlap[seq[i]][seq[j+1]] (if j<N-1)
                
                delta = 0
                
                # Boundary edges
                if i > 0:
                    delta -= overlap[current_seq[i-1]][current_seq[i]]
                    delta += overlap[current_seq[i-1]][current_seq[j]]
                if j < N - 1:
                    delta -= overlap[current_seq[j]][current_seq[j+1]]
                    delta += overlap[current_seq[i]][current_seq[j+1]]
                
                # Internal edges: old is forward, new is reversed
                for k in range(i, j):
                    delta -= overlap[current_seq[k]][current_seq[k+1]]
                    delta += overlap[current_seq[k+1]][current_seq[k]]
                
                if delta > 0:
                    # Apply move: reverse segment [i, j]
                    current_seq[i:j+1] = current_seq[i:j+1][::-1]
                    current_score += delta
                    improved = True
        
        # Also try single adjacent swaps (1-opt) for fine-tuning
        if (time.time() - start_time) > time_limit - 5:
            break
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 5:
                break
            j = i + 1
            delta = 0
            if i > 0:
                delta -= overlap[current_seq[i-1]][current_seq[i]]
                delta += overlap[current_seq[i-1]][current_seq[j]]
            if j < N - 1:
                delta -= overlap[current_seq[j]][current_seq[j+1]]
                delta += overlap[current_seq[i]][current_seq[j+1]]
            delta -= overlap[current_seq[i]][current_seq[j]]
            delta += overlap[current_seq[j]][current_seq[i]]
            
            if delta > 0:
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                current_score += delta
                improved = True
    
    # Reconstruct word
    res = list(current_seq[0])
    for i in range(1, N):
        prev = current_seq[i-1]
        curr = current_seq[i]
        k = overlap[prev][curr]
        res.extend(curr[k:])
    
    return "".join(map(str, res))

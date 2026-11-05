import time
import numpy as np
import itertools

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
    # This gives a sequence of permutations
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
    
    # Map permutation tuple to index
    perm_to_idx = {p: i for i, p in enumerate(perm_seq)}
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perm_seq[i][n-k:] == perm_seq[j][:k]
    # We use a numpy array for speed
    overlap = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: Group permutations by prefix/suffix
    # For each k from 1 to n-1, we can check matches
    # But simple loop is probably fast enough with N=5040
    # 5040*5040 = 25M. 25M * 6 checks = 150M ops. ~15s in Python.
    # Let's try to speed it up.
    
    # Create a dictionary for suffixes of length k
    # Actually, let's just do the nested loop but break early.
    
    # To speed up, we can use the fact that most overlaps are 0.
    # We only need to check j where perm_seq[j][0] == perm_seq[i][-1] for k=1
    # and so on.
    
    # Let's create lists of indices by prefix
    # prefix_map[k][prefix] = list of indices j such that perm_seq[j][:k] == prefix
    
    prefix_maps = [{} for _ in range(n)]
    for i, p in enumerate(perm_seq):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_maps[k]:
                prefix_maps[k][pref] = []
            prefix_maps[k][pref].append(i)
            
    # Now compute overlap
    for i in range(N):
        pi = perm_seq[i]
        # Check max possible overlap
        for k in range(n - 1, 0, -1):
            suffix = pi[n-k:]
            # Find all j such that perm_seq[j][:k] == suffix
            if suffix in prefix_maps[k]:
                for j in prefix_maps[k][suffix]:
                    overlap[i][j] = k
            # If we found matches for k, we don't need to check smaller k for those j?
            # No, overlap is the MAX k. So if we found k, it's the max for that i,j pair.
            # But we need to fill the matrix.
            # Wait, the loop above sets overlap[i][j] = k.
            # If we iterate k from n-1 down to 1, the first match is the max.
            # So we can break? No, we need to fill all j.
            # But for a specific i, if we find a j with overlap k, we don't care about smaller overlaps for that j.
            # However, we are filling the matrix.
            # This approach is O(N * (sum of counts)).
            # Sum of counts for a specific k is N.
            # So total ops is roughly N * N * (1/k factor)?
            # Actually, for each i, we check k=n-1, then k=n-2, etc.
            # The number of j's for a specific suffix is small.
            # This should be much faster than N^2 * n.
            
    # Compute initial score
    # score = sum of overlaps
    current_seq = list(range(N))
    current_score = 0
    for i in range(N - 1):
        current_score += overlap[current_seq[i]][current_seq[i+1]]
        
    def compute_score(seq):
        return sum(overlap[seq[i]][seq[i+1]] for i in range(len(seq) - 1))
        
    # 2-Opt Local Search
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        # First-Best strategy
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 5:
                break
            for j in range(i + 1, N):
                # Calculate delta
                # Edges removed: (i-1, i), (i, i+1) ... (j, j+1)
                # Edges added: (i-1, j), (j, j-1) ... (i, j+1)
                
                # We only care about boundary changes if we assume internal overlaps are small?
                # No, we must calculate exact delta.
                # But internal delta is sum(overlap[k][k+1] - overlap[k+1][k])
                
                # Optimization: Precompute reverse_overlap?
                # rev_overlap[i][j] = overlap[j][i]
                
                delta = 0
                if i > 0:
                    delta -= overlap[current_seq[i-1]][current_seq[i]]
                    delta += overlap[current_seq[i-1]][current_seq[j]]
                if j < N - 1:
                    delta -= overlap[current_seq[j]][current_seq[j+1]]
                    delta += overlap[current_seq[i]][current_seq[j+1]]
                
                # Internal
                # This inner loop is the bottleneck.
                # We can skip if boundary delta is very negative?
                # Or precompute prefix sums of overlap differences?
                # diff[k] = overlap[seq[k]][seq[k+1]] - overlap[seq[k+1]][seq[k]]
                # sum(diff[i:j])
                
                # Let's maintain a diff array
                # But it changes with seq.
                
                # For now, just do the loop.
                for k in range(i, j):
                    delta -= overlap[current_seq[k]][current_seq[k+1]]
                    delta += overlap[current_seq[k+1]][current_seq[k]]
                    
                if delta > 0:
                    current_seq[i:j+1] = current_seq[i:j+1][::-1]
                    current_score += delta
                    improved = True
                    # First-best: restart scan
                    break
        
    # Reconstruct word
    res = list(perm_seq[current_seq[0]])
    for i in range(1, N):
        prev = perm_seq[current_seq[i-1]]
        curr = perm_seq[current_seq[i]]
        k = overlap[current_seq[i-1]][current_seq[i]]
        res.extend(curr[k:])
        
    return "".join(map(str, res))

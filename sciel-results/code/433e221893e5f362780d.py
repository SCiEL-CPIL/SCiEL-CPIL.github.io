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
    
    # VNS with multiple neighborhood structures
    # Neighborhood 1: 2-opt (reverse segment)
    # Neighborhood 2: Or-opt (move a contiguous segment to another position)
    # Neighborhood 3: 3-opt (more complex rearrangements)
    
    # Precompute for faster delta calculation
    # For 2-opt: delta = -overlap[i-1][i] + overlap[i-1][j] - overlap[j][j+1] + overlap[i][j+1]
    #            + sum_{k=i}^{j-1} (overlap[k+1][k] - overlap[k][k+1])
    # We can precompute prefix sums of (overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]])
    
    # Maintain diff array: diff[k] = overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]] for k in 0..N-2
    diff = np.zeros(N-1, dtype=np.int32)
    for k in range(N-1):
        diff[k] = overlap[current_seq[k+1]][current_seq[k]] - overlap[current_seq[k]][current_seq[k+1]]
    
    # Prefix sums of diff for fast range sum
    diff_prefix = np.concatenate([[0], np.cumsum(diff)])
    
    def update_diff_after_move(old_seq, new_seq, changed_range):
        """Update diff array after a move."""
        for k in range(changed_range[0], changed_range[1]):
            if k < N-1:
                diff[k] = overlap[new_seq[k+1]][new_seq[k]] - overlap[new_seq[k]][new_seq[k+1]]
        diff_prefix[:] = np.concatenate([[0], np.cumsum(diff)])
    
    def range_sum_diff(i, j):
        """Sum of diff[i:j] (inclusive of i, exclusive of j)."""
        if i >= j:
            return 0
        return diff_prefix[j] - diff_prefix[i]
    
    # Variable Neighborhood Search
    improved = True
    neighborhood = 0
    
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        
        if neighborhood == 0:
            # 2-opt: reverse segment [i, j]
            for i in range(N - 1):
                if (time.time() - start_time) > time_limit - 5:
                    break
                for j in range(i + 1, N):
                    # Calculate delta for reversing [i, j]
                    delta = 0
                    if i > 0:
                        delta -= overlap[current_seq[i-1]][current_seq[i]]
                        delta += overlap[current_seq[i-1]][current_seq[j]]
                    if j < N - 1:
                        delta -= overlap[current_seq[j]][current_seq[j+1]]
                        delta += overlap[current_seq[i]][current_seq[j+1]]
                    
                    # Internal: sum of (overlap[k+1][k] - overlap[k][k+1]) for k=i to j-1
                    delta += range_sum_diff(i, j)
                    
                    if delta > 0:
                        current_seq[i:j+1] = current_seq[i:j+1][::-1]
                        current_score += delta
                        # Update diff
                        update_diff_after_move(current_seq, current_seq, (max(0, i-1), min(N-1, j+1)))
                        improved = True
                        # First-best: restart scan
                        break
                if improved:
                    break
        
        elif neighborhood == 1:
            # Or-opt: move segment [i, i+len-1] to position j
            # Try small segment lengths first
            for seg_len in [1, 2, 3]:
                if (time.time() - start_time) > time_limit - 5:
                    break
                for i in range(N - seg_len + 1):
                    if (time.time() - start_time) > time_limit - 5:
                        break
                    for j in range(N - seg_len + 1):
                        if j == i or (j > i and j < i + seg_len):
                            continue
                        # Calculate delta for moving [i, i+seg_len-1] to position j
                        # This is complex; skip for now and use a simpler move
                        # Let's try: swap two elements (1-swap)
                        pass
                # Fallback: try 1-swaps
                for i in range(N - 1):
                    if (time.time() - start_time) > time_limit - 5:
                        break
                    for j in range(i + 1, N):
                        # Swap positions i and j
                        delta = 0
                        if i > 0:
                            delta -= overlap[current_seq[i-1]][current_seq[i]]
                            delta += overlap[current_seq[i-1]][current_seq[j]]
                        if i < N - 1:
                            delta -= overlap[current_seq[i]][current_seq[i+1]]
                            if j == i + 1:
                                delta += overlap[current_seq[j]][current_seq[i]]
                            else:
                                delta += overlap[current_seq[i]][current_seq[j]]
                        if j > 0 and j != i + 1:
                            delta -= overlap[current_seq[j-1]][current_seq[j]]
                            delta += overlap[current_seq[j-1]][current_seq[i]]
                        if j < N - 1 and j != i + 1:
                            delta -= overlap[current_seq[j]][current_seq[j+1]]
                            delta += overlap[current_seq[i]][current_seq[j+1]]
                        
                        if delta > 0:
                            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                            current_score += delta
                            update_diff_after_move(current_seq, current_seq, (max(0, i-1), min(N-1, j+1)))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        else:
            # 3-opt: try all rearrangements of 3 consecutive elements
            for i in range(N - 2):
                if (time.time() - start_time) > time_limit - 5:
                    break
                a, b, c = current_seq[i], current_seq[i+1], current_seq[i+2]
                # Try 6 permutations
                perms_3 = [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]
                for perm in perms_3:
                    if perm[0] == a and perm[1] == b and perm[2] == c:
                        continue
                    # Calculate delta
                    delta = 0
                    if i > 0:
                        delta -= overlap[current_seq[i-1]][a]
                        delta += overlap[current_seq[i-1]][perm[0]]
                    if i + 2 < N - 1:
                        delta -= overlap[c][current_seq[i+3]]
                        delta += overlap[perm[2]][current_seq[i+3]]
                    # Internal edges
                    delta -= overlap[a][b] + overlap[b][c]
                    delta += overlap[perm[0]][perm[1]] + overlap[perm[1]][perm[2]]
                    
                    if delta > 0:
                        current_seq[i:i+3] = list(perm)
                        current_score += delta
                        update_diff_after_move(current_seq, current_seq, (max(0, i-1), min(N-1, i+3)))
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Move to next neighborhood
            neighborhood = (neighborhood + 1) % 3
            # After cycling through all neighborhoods, reset to 0
            if neighborhood == 0:
                # No improvement in full cycle; could try random restarts
                # But for now, just continue with 2-opt
                pass
        
        # Periodically verify score consistency
        if time.time() - start_time > time_limit - 10:
            break
    
    # Final verification and reconstruction
    # Recompute score to ensure consistency
    final_score = 0
    for i in range(N - 1):
        final_score += overlap[current_seq[i]][current_seq[i+1]]
    
    # Reconstruct word
    res = list(perm_seq[current_seq[0]])
    for i in range(1, N):
        prev = perm_seq[current_seq[i-1]]
        curr = perm_seq[current_seq[i]]
        k = overlap[current_seq[i-1]][current_seq[i]]
        res.extend(curr[k:])
        
    return "".join(map(str, res))

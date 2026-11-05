"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
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

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
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

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix using NumPy for speed
        import numpy as np
        # perms is a list of tuples of length 6
        perm_arr = np.array(perms, dtype=np.int8)  # shape (N, 6)
        
        # overlap[i][j] = longest k such that perm_arr[i][-k:] == perm_arr[j][:k]
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for k in range(1, 7):
            # Check if suffix of length k of i matches prefix of length k of j
            # suffix_k[i] = perm_arr[i, 6-k:], prefix_k[j] = perm_arr[j, :k]
            # We need to check all pairs where suffix_k[i] == prefix_k[j]
            # Group by suffix value
            suffixes = perm_arr[:, 6-k:]  # shape (N, k)
            prefixes = perm_arr[:, :k]    # shape (N, k)
            
            # For each i, find all j where prefixes[j] == suffixes[i]
            # Use a dictionary approach
            prefix_to_indices = {}
            for j in range(N):
                key = tuple(prefixes[j].tolist())
                if key not in prefix_to_indices:
                    prefix_to_indices[key] = []
                prefix_to_indices[key].append(j)
            
            for i in range(N):
                key = tuple(suffixes[i].tolist())
                if key in prefix_to_indices:
                    for j in prefix_to_indices[key]:
                        overlap_mat[i][j] = k

        # Helper to calculate total overlap
        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        # Helper to build word from path
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # 1. Generate Naive Word (standard order)
        naive_path = list(range(N))
        naive_word = build_word_from_path(naive_path)
        best_word = naive_word
        best_overlap = calc_total_overlap(naive_path)

        # 2. Simulated Annealing Optimization
        current_path = naive_path[:]
        current_overlap = calc_total_overlap(current_path)
        best_path = current_path[:]
        best_overlap = current_overlap
        
        # SA Parameters
        T = 5.0
        cooling = 0.9997
        steps = 5000000
        
        random.seed(42)
        
        for step in range(steps):
            if time.time() > deadline - 15:
                break
                
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            p_i = current_path[i]
            p_j = current_path[j]
            
            # Calculate Delta
            old_ov = 0
            new_ov = 0
            
            if abs(i - j) == 1:
                # Adjacent swap
                if i == 0:
                    # i is first, j is second
                    old_ov += overlap_mat[p_i][p_j]
                    new_ov += overlap_mat[p_j][p_i]
                    if N > 2:
                        right = current_path[2]
                        old_ov += overlap_mat[p_j][right]
                        new_ov += overlap_mat[p_i][right]
                elif i == N - 1:
                    # i is last, j is second-to-last
                    left = current_path[N-2]
                    old_ov += overlap_mat[left][p_j]
                    new_ov += overlap_mat[left][p_i]
                    old_ov += overlap_mat[p_j][p_i]
                    new_ov += overlap_mat[p_i][p_j]
                elif j == 0:
                    # j is first, i is second
                    old_ov += overlap_mat[p_j][p_i]
                    new_ov += overlap_mat[p_i][p_j]
                    if N > 2:
                        right = current_path[2]
                        old_ov += overlap_mat[p_i][right]
                        new_ov += overlap_mat[p_j][right]
                elif j == N - 1:
                    # j is last, i is second-to-last
                    left = current_path[N-2]
                    old_ov += overlap_mat[left][p_i]
                    new_ov += overlap_mat[left][p_j]
                    old_ov += overlap_mat[p_i][p_j]
                    new_ov += overlap_mat[p_j][p_i]
                else:
                    # Both in middle
                    left = current_path[i-1]
                    right = current_path[i+1]
                    old_ov += overlap_mat[left][p_i]
                    old_ov += overlap_mat[p_i][p_j]
                    old_ov += overlap_mat[p_j][right]
                    new_ov += overlap_mat[left][p_j]
                    new_ov += overlap_mat[p_j][p_i]
                    new_ov += overlap_mat[p_i][right]
            else:
                # Non-adjacent swap
                if i > 0:
                    left_i = current_path[i-1]
                    old_ov += overlap_mat[left_i][p_i]
                    new_ov += overlap_mat[left_i][p_j]
                if i < N - 1:
                    right_i = current_path[i+1]
                    old_ov += overlap_mat[p_i][right_i]
                    new_ov += overlap_mat[p_j][right_i]
                
                if j > 0:
                    left_j = current_path[j-1]
                    old_ov += overlap_mat[left_j][p_j]
                    new_ov += overlap_mat[left_j][p_i]
                if j < N - 1:
                    right_j = current_path[j+1]
                    old_ov += overlap_mat[p_j][right_j]
                    new_ov += overlap_mat[p_i][right_j]
            
            delta = new_ov - old_ov
            
            if delta > 0:
                current_path[i], current_path[j] = current_path[j], current_path[i]
                current_overlap += delta
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
            else:
                # Accept with probability exp(delta/T)
                if T > 1e-6:
                    if random.random() < math.exp(delta / T):
                        current_path[i], current_path[j] = current_path[j], current_path[i]
                        current_overlap += delta
                        if current_overlap > best_overlap:
                            best_overlap = current_overlap
                            best_path = current_path[:]
            
            T *= cooling

        # 3. Local search polish: 2-opt on best_path
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 5:
                        break
                    p_i = best_path[i]
                    p_j = best_path[j]
                    
                    old_ov = 0
                    new_ov = 0
                    
                    if j == i + 1:
                        # Adjacent
                        if i > 0:
                            left = best_path[i-1]
                            old_ov += overlap_mat[left][p_i]
                            new_ov += overlap_mat[left][p_j]
                        old_ov += overlap_mat[p_i][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                        if j < N - 1:
                            right = best_path[j+1]
                            old_ov += overlap_mat[p_j][right]
                            new_ov += overlap_mat[p_i][right]
                    else:
                        # Non-adjacent
                        if i > 0:
                            left_i = best_path[i-1]
                            old_ov += overlap_mat[left_i][p_i]
                            new_ov += overlap_mat[left_i][p_j]
                        if i < N - 1:
                            right_i = best_path[i+1]
                            old_ov += overlap_mat[p_i][right_i]
                            new_ov += overlap_mat[p_j][right_i]
                        
                        if j > 0:
                            left_j = best_path[j-1]
                            old_ov += overlap_mat[left_j][p_j]
                            new_ov += overlap_mat[left_j][p_i]
                        if j < N - 1:
                            right_j = best_path[j+1]
                            old_ov += overlap_mat[p_j][right_j]
                            new_ov += overlap_mat[p_i][right_j]
                    
                    delta = new_ov - old_ov
                    if delta > 0:
                        best_path[i], best_path[j] = best_path[j], best_path[i]
                        best_overlap += delta
                        improved = True
        
        # Build final word from best_path
        final_word = build_word_from_path(best_path)
        
        # Compare with naive
        if len(final_word) < len(naive_word):
            return final_word
        else:
            return naive_word

    # Fallback for n > 7
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
# EVOLVE-BLOCK-END

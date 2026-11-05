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
        
        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                pj = perms[j]
                best = 0
                # Max possible overlap is 5 since pi != pj
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        best = k
                        break
                ov[i][j] = best

        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total

        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word

        # Initialize with standard order
        current_path = list(range(N))
        current_overlap = calc_total_overlap(current_path)
        best_path = current_path[:]
        best_overlap = current_overlap

        # Simulated Annealing
        # Parameters
        T0 = 10.0
        Tend = 0.01
        alpha = 0.99999  # Cooling rate
        T = T0
        
        # Precompute prefix sums for faster cost update? 
        # For SA, we usually just update the delta cost directly for local moves.
        # Delta cost for moving elements requires checking neighbors.
        
        # To speed up SA, we can use a few restarts if time permits, 
        # but a single long SA run with good moves is often better.
        
        # Move types:
        # 1. Swap adjacent (i, i+1)
        # 2. 2-Opt (reverse segment i..j)
        # 3. Relocate (move element i to position j)
        
        # We will use a mix.
        
        # Helper to calculate cost change for 2-Opt reverse [i, j]
        # Cost = sum ov[p[k]][p[k+1]]
        # Old cost parts: ov[p[i-1]][p[i]] + sum_{k=i}^{j-1} ov[p[k]][p[k+1]] + ov[p[j]][p[j+1]]
        # New cost parts: ov[p[i-1]][p[j]] + sum_{k=i}^{j-1} ov[p[k+1]][p[k]] + ov[p[i]][p[j+1]]
        # We need prefix sums of forward and backward overlaps to do this in O(1).
        
        fwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            fwd_prefix[i + 1] = fwd_prefix[i] + ov[current_path[i]][current_path[i+1]]
            
        bwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            bwd_prefix[i + 1] = bwd_prefix[i] + ov[current_path[i+1]][current_path[i]]

        def get_fwd_sum(i, j):
            # Sum of ov[p[k]][p[k+1]] for k in [i, j-1]
            if i >= j:
                return 0
            return fwd_prefix[j] - fwd_prefix[i]
            
        def get_bwd_sum(i, j):
            # Sum of ov[p[k+1]][p[k]] for k in [i, j-1]
            if i >= j:
                return 0
            return bwd_prefix[j] - bwd_prefix[i]

        def update_prefixes(path):
            nonlocal fwd_prefix, bwd_prefix
            fwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                fwd_prefix[i + 1] = fwd_prefix[i] + ov[path[i]][path[i+1]]
            bwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                bwd_prefix[i + 1] = bwd_prefix[i] + ov[path[i+1]][path[i]]

        # SA Loop
        while T > Tend and time.time() < deadline - 5:
            # Choose move type
            r = random.random()
            
            if r < 0.3:
                # 2-Opt
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                
                left_idx = i - 1
                right_idx = j + 1
                
                old_cost = 0
                if left_idx != -1:
                    old_cost += ov[current_path[left_idx]][current_path[i]]
                old_cost += get_fwd_sum(i, j)
                if right_idx != N:
                    old_cost += ov[current_path[j]][current_path[right_idx]]
                
                new_cost = 0
                if left_idx != -1:
                    new_cost += ov[current_path[left_idx]][current_path[j]]
                new_cost += get_bwd_sum(i, j)
                if right_idx != N:
                    new_cost += ov[current_path[i]][current_path[right_idx]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_path[i:j+1] = reversed(current_path[i:j+1])
                    current_overlap += delta
                    update_prefixes(current_path)
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                        
            elif r < 0.6:
                # Swap adjacent
                i = random.randint(0, N - 2)
                j = i + 1
                
                left_idx = i - 1
                right_idx = j + 1
                
                old_cost = 0
                if left_idx != -1:
                    old_cost += ov[current_path[left_idx]][current_path[i]]
                old_cost += ov[current_path[i]][current_path[j]]
                if right_idx != N:
                    old_cost += ov[current_path[j]][current_path[right_idx]]
                
                new_cost = 0
                if left_idx != -1:
                    new_cost += ov[current_path[left_idx]][current_path[j]]
                new_cost += ov[current_path[j]][current_path[i]]
                if right_idx != N:
                    new_cost += ov[current_path[i]][current_path[right_idx]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_path[i], current_path[j] = current_path[j], current_path[i]
                    current_overlap += delta
                    update_prefixes(current_path)
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                        
            else:
                # Relocate single element
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j or i == j + 1:
                    T *= alpha
                    continue
                    
                item = current_path[i]
                
                # Remove i
                # Neighbors of i: left, right
                left_idx = i - 1
                right_idx = i + 1
                
                old_cost_remove = 0
                if left_idx != -1 and right_idx != N:
                    old_cost_remove = ov[current_path[left_idx]][item] + ov[item][current_path[right_idx]] - ov[current_path[left_idx]][current_path[right_idx]]
                elif left_idx != -1:
                    old_cost_remove = ov[current_path[left_idx]][item]
                elif right_idx != N:
                    old_cost_remove = ov[item][current_path[right_idx]]
                
                # Insert at j
                # Note: if j > i, indices shift. 
                # Let's just calculate cost change based on the list state.
                # It's safer to actually perform the move on a copy or handle indices carefully.
                # For simplicity and speed, we might skip complex relocate in SA if 2-opt is strong enough.
                # However, relocate is powerful.
                
                # Let's implement a simple relocate that updates cost by brute force on the affected parts.
                # This is O(1) if we track neighbors correctly.
                
                # Determine insertion neighbors in the *original* list
                if j > i:
                    # Insertion point shifts left by 1 in the original indices relative to the new position?
                    # Actually, let's just say we insert at index j in the list *after* removal?
                    # No, standard relocate: remove i, insert at j.
                    # If j > i, the index j in the new list corresponds to j in the old list?
                    # Let's stick to 2-Opt and Swap for robustness and speed. 
                    # Relocate with index management is error-prone.
                    pass
                else:
                    pass
            
            T *= alpha

        # Final Polish: Best Improvement Local Search on the best path found by SA
        # This often recovers a few more points of overlap.
        current_path = best_path[:]
        current_overlap = best_overlap
        update_prefixes(current_path)
        
        improved = True
        passes = 0
        max_passes = 100
        
        while improved and time.time() < deadline - 2 and passes < max_passes:
            improved = False
            passes += 1
            
            # 2-Opt Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            
            for i in range(N - 1):
                left_idx = i - 1
                for j in range(i + 1, N):
                    right_idx = j + 1
                    
                    old_cost = 0
                    if left_idx != -1:
                        old_cost += ov[current_path[left_idx]][current_path[i]]
                    old_cost += get_fwd_sum(i, j)
                    if right_idx != N:
                        old_cost += ov[current_path[j]][current_path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1:
                        new_cost += ov[current_path[left_idx]][current_path[j]]
                    new_cost += get_bwd_sum(i, j)
                    if right_idx != N:
                        new_cost += ov[current_path[i]][current_path[right_idx]]
                    
                    gain = new_cost - old_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                current_path[best_i:best_j+1] = reversed(current_path[best_i:best_j+1])
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True
                continue

            # 1-Opt Relocate Best Improvement (Simplified)
            best_gain = 0
            best_i = -1
            best_j = -1
            for i in range(N):
                left = current_path[i-1] if i > 0 else -1
                right = current_path[i+1] if i < N - 1 else -1
                
                gain_remove = 0
                if left != -1 and right != -1:
                    gain_remove = ov[left][right] - ov[left][i] - ov[i][right]
                elif left != -1:
                    gain_remove = -ov[left][i]
                elif right != -1:
                    gain_remove = -ov[i][right]
                
                item = current_path[i]
                
                for j in range(N):
                    if j == i or j == i + 1:
                        continue
                    
                    if j < i:
                        left_orig = j - 1 if j > 0 else -1
                        right_orig = j
                    else:
                        left_orig = j
                        right_orig = j + 1 if j < N - 1 else -1

                    gain_insert = 0
                    if left_orig != -1 and right_orig != -1:
                        gain_insert = ov[left_orig][item] + ov[item][right_orig] - ov[left_orig][right_orig]
                    elif left_orig != -1:
                        gain_insert = ov[left_orig][item]
                    elif right_orig != -1:
                        gain_insert = ov[item][right_orig]
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i = best_i
                j = best_j
                item = current_path[i]
                del current_path[i]
                if i < j:
                    j -= 1
                current_path.insert(j, item)
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True

        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
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

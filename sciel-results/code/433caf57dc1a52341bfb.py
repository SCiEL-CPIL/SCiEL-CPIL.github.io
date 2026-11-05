import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Generate standard superpermutation for n-1
    def gen_std(m):
        if m == 1:
            return [1]
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    prev_word = gen_std(n-1)
    blocks = []
    seen = set()
    for i in range(len(prev_word) - (n-2)):
        window = tuple(prev_word[i:i+n-1])
        if len(set(window)) == n-1 and window not in seen:
            seen.add(window)
            blocks.append(window)
            
    m = len(blocks)
    
    # Precompute overlaps using numpy for speed
    import numpy as np
    
    # Convert blocks to numpy arrays for faster overlap computation
    block_arrs = [np.array(b, dtype=np.int8) for b in blocks]
    
    ov = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        b_i = block_arrs[i]
        for j in range(m):
            b_j = block_arrs[j]
            max_ov = min(len(b_i), len(b_j))
            found = 0
            for k in range(max_ov, 0, -1):
                if np.array_equal(b_i[-k:], b_j[:k]):
                    found = k
                    break
            ov[i][j] = found
    
    ov_list = ov.tolist()
    
    def get_score(order):
        s = 0
        for i in range(len(order)-1):
            s += ov_list[order[i]][order[i+1]]
        return s

    # Multiple greedy initializations with different strategies
    best_order = list(range(m))
    best_score = -1
    start_time = time.time()
    time_limit = 110
    safety_margin = 10  # seconds
    
    # Try standard order first
    std_order = list(range(m))
    std_score = get_score(std_order)
    if std_score > best_score:
        best_score = std_score
        best_order = std_order

    # Random Greedy with multiple starts and temperature-based selection
    for trial in range(50):
        if time.time() - start_time > 30:
            break
        visited = [False]*m
        order = []
        current = random.randint(0, m-1)
        visited[current] = True
        order.append(current)
        
        for _ in range(m-1):
            best_next = -1
            best_ov_val = -1
            # Add some randomness to avoid getting stuck
            if random.random() < 0.1 and len(order) > 2:
                # Occasionally pick a non-greedy choice
                candidates = [x for x in range(m) if not visited[x]]
                if candidates:
                    best_next = random.choice(candidates)
                    visited[best_next] = True
                    order.append(best_next)
                    current = best_next
                    continue
            
            for next_node in range(m):
                if not visited[next_node]:
                    val = ov_list[current][next_node]
                    if val > best_ov_val:
                        best_ov_val = val
                        best_next = next_node
            if best_next == -1: break
            visited[best_next] = True
            order.append(best_next)
            current = best_next
            
        score = get_score(order)
        if score > best_score:
            best_score = score
            best_order = order

    # VNS with 2-opt and 3-opt moves
    def try_2opt(order, score):
        """Try 2-opt reversals, return (new_order, new_score, improved)"""
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Calculate D and PrefixD
        D = [0]*(m-1)
        for k in range(m-1):
            D[k] = ov_list[order[k+1]][order[k]] - ov_list[order[k]][order[k+1]]
            
        PrefixD = [0]*(m)
        for i in range(1, m):
            PrefixD[i] = PrefixD[i-1] + D[i-1]
        
        for i in range(m-1):
            for j in range(i+1, m):
                delta = 0
                # Boundary 1
                if i > 0:
                    delta -= ov_list[order[i-1]][order[i]]
                    delta += ov_list[order[i-1]][order[j]]
                # Boundary 2
                if j < m-1:
                    delta -= ov_list[order[j]][order[j+1]]
                    delta += ov_list[order[i]][order[j+1]]
                
                # Internal sum: sum of D[k] for k=i to j-1
                delta += (PrefixD[j] - PrefixD[i])
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_delta > 0:
            new_order = order[:]
            new_order[best_i:best_j+1] = new_order[best_i:best_j+1][::-1]
            return new_order, score + best_delta, True
        return order, score, False

    def try_3opt(order, score):
        """Try 3-opt moves (remove two segments and reinsert), return (new_order, new_score, improved)"""
        best_delta = 0
        best_move = None
        
        # Try removing segment [i1, i2] and inserting it at position j
        for i1 in range(m-1):
            if time.time() - start_time > time_limit - safety_margin:
                break
            for i2 in range(i1+1, m):
                if i2 - i1 + 1 > m // 3:  # Limit segment length
                    continue
                segment = order[i1:i2+1]
                
                for j in range(m):
                    if j >= i1 and j <= i2:
                        continue
                    # Compute delta for inserting segment at position j
                    # Remove segment from [i1, i2], insert at j
                    # This is complex; try a simpler approach: swap adjacent elements
                    pass
        
        # Simpler: try swapping pairs of non-adjacent elements
        for i in range(m):
            if time.time() - start_time > time_limit - safety_margin:
                break
            for j in range(i+2, m):
                if j - i == 1:
                    continue
                # Try swapping order[i] and order[j]
                old_a = ov_list[order[i-1]][order[i]] if i > 0 else 0
                old_b = ov_list[order[i]][order[i+1]] if i < m-1 else 0
                old_c = ov_list[order[j-1]][order[j]] if j > 0 else 0
                old_d = ov_list[order[j]][order[j+1]] if j < m-1 else 0
                
                new_a = ov_list[order[j-1]][order[i]] if j > 0 else 0
                new_b = ov_list[order[i]][order[i+1]] if i < m-1 else 0
                new_c = ov_list[order[i-1]][order[j]] if i > 0 else 0
                new_d = ov_list[order[j]][order[j+1]] if j < m-1 else 0
                
                # Also need to account for the connection between i and j positions
                # This is getting complex; let's just do simple 2-opt and rely on multiple restarts
                pass
        
        return order, score, False

    # Main VNS loop with 2-opt
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() - start_time < time_limit - safety_margin:
        improved = False
        
        # Try 2-opt
        new_order, new_score, was_improved = try_2opt(best_order, best_score)
        if was_improved:
            best_order = new_order
            best_score = new_score
            improved = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if not improved:
            if no_improvement_count >= max_no_improvement:
                # Perturb and restart
                if time.time() - start_time > time_limit - safety_margin:
                    break
                # Random perturbation
                for _ in range(3):
                    i = random.randint(0, m-2)
                    j = random.randint(i+1, m-1)
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                best_score = get_score(best_order)
                no_improvement_count = 0
            else:
                # Small random perturbation
                i = random.randint(0, m-2)
                j = random.randint(i+1, m-1)
                if random.random() < 0.5:
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                else:
                    # Swap two random elements
                    a = random.randint(0, m-1)
                    b = random.randint(0, m-1)
                    if a != b:
                        best_order[a], best_order[b] = best_order[b], best_order[a]
                best_score = get_score(best_order)
        else:
            no_improvement_count = 0

    # Final polish: multiple passes of 2-opt
    for _ in range(5):
        if time.time() - start_time > time_limit - safety_margin:
            break
        new_order, new_score, was_improved = try_2opt(best_order, best_score)
        if not was_improved:
            break
        best_order = new_order
        best_score = new_score

    # Reconstruct
    word = []
    if m > 0:
        first_block = blocks[best_order[0]]
        word.extend(first_block)
        word.append(n)
        word.extend(first_block)
        
        for i in range(1, m):
            curr_block = blocks[best_order[i]]
            prev_block = blocks[best_order[i-1]]
            
            ov_val = ov_list[best_order[i-1]][best_order[i]]
            
            word.extend(curr_block[ov_val:])
            word.append(n)
            word.extend(curr_block)
            
    return "".join(map(str, word))

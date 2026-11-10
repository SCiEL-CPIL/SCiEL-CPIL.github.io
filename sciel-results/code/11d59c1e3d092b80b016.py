import time
import math
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    if n <= 5:
        return _build_classic(n)

    # For n >= 6, use optimized construction
    # 1. Build classic S_{n-1}
    base_word = _build_classic(n - 1)
    
    # 2. Extract sequence of (n-1)-permutations
    base_perms = []
    seen = set()
    m = n - 1
    for i in range(len(base_word) - m + 1):
        window = tuple(base_word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            base_perms.append(window)
    
    num_perms = len(base_perms)
    
    # 3. Compute overlap matrix efficiently using string hashing
    overlap_mat = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    # Precompute prefix hashes for each permutation
    # For each k from 1 to m-1, we want to find max overlap
    # Use a dictionary approach: for each k, map prefix -> list of indices
    
    # For each k from m-1 down to 1:
    for k in range(m - 1, 0, -1):
        # Build mapping: prefix of length k -> list of perm indices
        prefix_map = {}
        for j in range(num_perms):
            prefix = tuple(base_perms[j][:k])
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(j)
        
        # For each perm i, find all j where prefix of j matches suffix of i
        for i in range(num_perms):
            suffix = tuple(base_perms[i][m-k:])
            if suffix in prefix_map:
                for j in prefix_map[suffix]:
                    # Only set if not already set (we go from high k to low k)
                    if overlap_mat[i][j] == 0:
                        overlap_mat[i][j] = k
    
    # 4. Optimize using Simulated Annealing with better moves
    # Initial order: natural order
    order = list(range(num_perms))
    
    # Calculate initial total overlap
    total_overlap = 0
    for i in range(num_perms - 1):
        total_overlap += overlap_mat[order[i]][order[i+1]]
    
    best_order = order[:]
    best_total_overlap = total_overlap
    
    # SA Parameters
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    current_order = order[:]
    current_overlap = total_overlap
    
    # Precompute for fast delta calculations
    # Store as list for faster indexing than numpy in tight loops
    overlap_list = [overlap_mat[i].tolist() for i in range(num_perms)]
    
    no_improve = 0
    max_no_improve = 50000
    
    while T > T_min and time.time() < deadline - 10:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            
            a = current_order[i]
            b = current_order[j]
            
            old_cost = 0
            new_cost = 0
            
            # Left of i
            if i > 0:
                old_cost += overlap_list[current_order[i-1]][a]
                new_cost += overlap_list[current_order[i-1]][b]
            # Right of i
            if i < num_perms - 1:
                old_cost += overlap_list[a][current_order[i+1]]
                new_cost += overlap_list[b][current_order[i+1]]
                
            # Left of j
            if j > 0:
                old_cost += overlap_list[current_order[j-1]][b]
                new_cost += overlap_list[current_order[j-1]][a]
            # Right of j
            if j < num_perms - 1:
                old_cost += overlap_list[b][current_order[j+1]]
                new_cost += overlap_list[a][current_order[j+1]]
            
            delta = new_cost - old_cost
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_order[i], current_order[j] = b, a
                current_overlap += delta
                if current_overlap > best_total_overlap:
                    best_total_overlap = current_overlap
                    best_order = current_order[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
        elif move_type < 0.7:
            # Or-opt(1): Move single element
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j or i == j - 1:
                continue
            
            a = current_order[i]
            
            # Calculate delta for moving element from i to j
            # Case 1: i < j (moving right)
            # Case 2: i > j (moving left)
            
            if i < j:
                # Remove from i, insert at j (shift i+1..j left, a goes to j)
                old_cost = 0
                new_cost = 0
                
                # Edge before i
                if i > 0:
                    old_cost += overlap_list[current_order[i-1]][a]
                    new_cost += overlap_list[current_order[i-1]][current_order[i+1]]
                
                # Edge after i (before shift)
                old_cost += overlap_list[a][current_order[i+1]]
                
                # Internal edges i+1..j-1 stay the same, no change
                
                # Edge before j (after shift, was j-1)
                # After shift: ...j-2, j-1, a, j+1...
                # Before: ...j-2, j-1, j, j+1...
                old_cost += overlap_list[current_order[j-1]][current_order[j]]
                old_cost += overlap_list[current_order[j]][current_order[j+1]] if j < num_perms - 1 else 0
                new_cost += overlap_list[current_order[j-1]][a]
                new_cost += overlap_list[a][current_order[j+1]] if j < num_perms - 1 else 0
                
                # Wait, this is getting complex. Let me just recalculate locally.
                # For Or-opt, let's use a simpler approach:
                # Save the affected segment, remove, insert, recalculate.
                
                # Actually, let's just do a local recalculation for the affected region
                # Affected positions: i-1, i, i+1, ..., j, j+1
                # Old cost in region:
                old_region_cost = 0
                for pos in range(max(0, i-1), min(num_perms-1, j+1)):
                    old_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
                
                # Perform the move
                elem = current_order.pop(i)
                current_order.insert(j, elem)
                
                # New cost in region:
                new_region_cost = 0
                for pos in range(max(0, i-1), min(num_perms-1, j+1)):
                    new_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
                
                delta = new_region_cost - old_region_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_overlap += delta
                    if current_overlap > best_total_overlap:
                        best_total_overlap = current_overlap
                        best_order = current_order[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Undo
                    current_order.pop(j)
                    current_order.insert(i, elem)
                    
            else:  # i > j
                # Remove from i, insert at j
                old_region_cost = 0
                for pos in range(max(0, j-1), min(num_perms-1, i+1)):
                    old_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
                
                elem = current_order.pop(i)
                current_order.insert(j, elem)
                
                new_region_cost = 0
                for pos in range(max(0, j-1), min(num_perms-1, i+1)):
                    new_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
                
                delta = new_region_cost - old_region_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_overlap += delta
                    if current_overlap > best_total_overlap:
                        best_total_overlap = current_overlap
                        best_order = current_order[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Undo
                    current_order.pop(j)
                    current_order.insert(i, elem)
                    
        else:
            # Short 2-opt (reverse segment of length 2-8)
            i = random.randint(0, num_perms - 2)
            length = random.randint(2, 8)
            j = i + length
            if j >= num_perms:
                continue
            
            # Calculate delta
            old_region_cost = 0
            for pos in range(max(0, i-1), min(num_perms-1, j+1)):
                old_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
            
            # Reverse
            current_order[i:j+1] = current_order[i:j+1][::-1]
            
            new_region_cost = 0
            for pos in range(max(0, i-1), min(num_perms-1, j+1)):
                new_region_cost += overlap_list[current_order[pos]][current_order[pos+1]]
            
            delta = new_region_cost - old_region_cost
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_overlap += delta
                if current_overlap > best_total_overlap:
                    best_total_overlap = current_overlap
                    best_order = current_order[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Undo
                current_order[i:j+1] = current_order[i:j+1][::-1]
        
        T *= cooling_rate
        
        # Adaptive cooling: cool faster if stuck
        if no_improve > max_no_improve:
            T *= 0.9
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5)
        
        # Periodic restart if stuck too long
        if no_improve > max_no_improve * 3:
            # Try a few random restarts
            for _ in range(5):
                random_order = list(range(num_perms))
                random.shuffle(random_order)
                random_overlap = sum(overlap_list[random_order[i]][random_order[i+1]] 
                                   for i in range(num_perms - 1))
                if random_overlap > current_overlap:
                    current_order = random_order[:]
                    current_overlap = random_overlap
                    break
            no_improve = 0
            max_no_improve = 50000
    
    # 5. Build final word using the precomputed overlap matrix
    word = []
    for idx, perm_idx in enumerate(best_order):
        p = list(base_perms[perm_idx])
        if idx == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_idx = best_order[idx-1]
            overlap = overlap_mat[prev_idx][perm_idx]
            word.append(n)
            word.extend(p[overlap:])
    
    # Quick verification
    if time.time() < deadline - 5:
        all_perms = set(itertools.permutations(range(1, n+1)))
        found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                found.add(window)
        if len(found) < math.factorial(n):
            # Fallback to classic
            return _build_classic(n)
            
    return "".join(map(str, word))

def _build_classic(n):
    if n == 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o > 0 and new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new_word.extend(block[overlap:])
        word = new_word
    return word

# EVOLVE-BLOCK-END

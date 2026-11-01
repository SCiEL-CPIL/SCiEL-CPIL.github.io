import time
import random
import math
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the word for n-1 and extract its permutations in order of appearance
    def build_word_and_perms(m):
        if m == 1:
            return [1], [(1,)]
        
        prev_word, prev_perms = build_word_and_perms(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    # Get perms of n-1
    start_time = time.time()
    deadline = start_time + 130.0 # Leave 20s for safety and final step
    
    _, perms = build_word_and_perms(n - 1)
    num_perms = len(perms)
    
    if num_perms != math.factorial(n - 1):
        # Fallback to standard if something went wrong
        word, _ = build_word_and_perms(n)
        return "".join(map(str, word))

    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap_matrix[i][j] = found

    # Initial order: 0 to num_perms-1
    order = list(range(num_perms))
    
    # Calculate total overlap (to maximize)
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list) - 1):
            total += overlap_matrix[ord_list[i]][ord_list[i + 1]]
        return total

    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # Faster Local Search Strategy:
    # 1. Run 2-opt until no improvement, but sample moves to speed up.
    # 2. Use Simulated Annealing to escape local minima.
    
    current_overlap = best_overlap
    current_order = best_order[:]
    
    T = 10.0 # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a random move: 2-opt reverse segment [i, j]
        i = random.randint(0, len(current_order) - 2)
        j = random.randint(i + 1, len(current_order) - 1)
        
        # Calculate delta
        delta = 0
        
        # Left boundary
        if i > 0:
            left_node = current_order[i - 1]
            delta -= overlap_matrix[left_node][current_order[i]]
            delta += overlap_matrix[left_node][current_order[j]]
        
        # Right boundary
        if j < len(current_order) - 1:
            right_node = current_order[j + 1]
            delta -= overlap_matrix[current_order[j]][right_node]
            delta += overlap_matrix[current_order[i]][right_node]
        
        # Internal edges
        for k in range(i, j):
            delta -= overlap_matrix[current_order[k]][current_order[k + 1]]
            delta += overlap_matrix[current_order[k + 1]][current_order[k]]
            
        # Apply move
        current_order[i:j+1] = reversed(current_order[i:j+1])
        
        # Accept/Reject
        if delta > 0:
            current_overlap += delta
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_order = current_order[:]
        else:
            if random.random() < math.exp(delta / T):
                current_overlap += delta
            else:
                current_order[i:j+1] = reversed(current_order[i:j+1])
        
        T *= cooling_rate

    # Final polish: Hill Climbing 2-opt on best_order
    order = best_order[:]
    current_overlap = best_overlap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(1000):
            if time.time() > deadline:
                break
            i = random.randint(0, len(order) - 2)
            j = random.randint(i + 1, len(order) - 1)
            
            delta = 0
            if i > 0:
                left_node = order[i - 1]
                delta -= overlap_matrix[left_node][order[i]]
                delta += overlap_matrix[left_node][order[j]]
            if j < len(order) - 1:
                right_node = order[j + 1]
                delta -= overlap_matrix[order[j]][right_node]
                delta += overlap_matrix[order[i]][right_node]
            
            for k in range(i, j):
                delta -= overlap_matrix[order[k]][order[k + 1]]
                delta += overlap_matrix[order[k + 1]][order[k]]
                
            if delta > 0:
                order[i:j+1] = reversed(order[i:j+1])
                current_overlap += delta
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]

    # Construct final string
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))

import time
import random
from itertools import permutations
from heapq import heappush, heappop
import sys
import math

def construct(n):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # Base case
    if n == 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        # Known shortest: 123121321
        return "123121321"
    
    # For n >= 4, use the recursive construction with optimization
    # The standard recursive construction gives length sum_{k=1}^n k!
    # For n=7, this is 5913. The known best is 5905.
    # We'll use a greedy approach to try to find a shorter one.
    
    # Strategy: Build S_1, S_2, ..., S_n using greedy TSP at each step
    # but with more sophisticated overlap calculation
    
    word = [1]
    
    for k in range(2, n + 1):
        time_remaining = deadline - time.time()
        
        # Extract unique (k-1)-permutations from current word
        perms = []
        seen = set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        if num_perms == 0:
            return "".join(map(str, word))
        
        # Precompute overlap matrix
        # overlap[i][j] = max overlap between end of perms[i] and start of perms[j]
        # Max possible overlap is k-2 (since each perm has length k-1)
        max_overlap_possible = k - 2
        
        # For efficiency, use suffix/prefix matching
        # Build a map from suffix of length o to list of indices
        suffix_map = {}
        for o in range(1, max_overlap_possible + 1):
            suffix_map[o] = {}
            for idx, p in enumerate(perms):
                suffix = p[-o:]
                if suffix not in suffix_map[o]:
                    suffix_map[o][suffix] = []
                suffix_map[o][suffix].append(idx)
        
        # Build prefix map
        prefix_map = {}
        for o in range(1, max_overlap_possible + 1):
            prefix_map[o] = {}
            for idx, p in enumerate(perms):
                prefix = p[:o]
                if prefix not in prefix_map[o]:
                    prefix_map[o][prefix] = []
                prefix_map[o][prefix].append(idx)
        
        # Compute overlap matrix
        # overlap[i][j] = max o such that perms[i][-o:] == perms[j][:o]
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            p_i = perms[i]
            for o in range(max_overlap_possible, 0, -1):
                suffix = p_i[-o:]
                if suffix in prefix_map[o]:
                    for j in prefix_map[o][suffix]:
                        if overlap_matrix[i][j] < o:
                            overlap_matrix[i][j] = o
        
        # Determine if we should use fast mode
        use_fast_mode = (time_remaining < 10.0) or (num_perms > 500)
        
        if use_fast_mode:
            best_order = list(range(num_perms))
        else:
            # Greedy TSP with multiple random restarts using precomputed overlaps
            best_order = None
            best_len = float('inf')
            
            # Try several random starts
            if num_perms <= 100:
                num_starts = min(num_perms, 30)
            else:
                num_starts = 15
                
            starts = random.sample(range(num_perms), num_starts)
            
            for start_idx in starts:
                if time.time() > deadline - 10.0:
                    break
                
                visited = [False] * num_perms
                order = []
                current = start_idx
                visited[current] = True
                order.append(current)
                
                for _ in range(num_perms - 1):
                    if time.time() > deadline - 10.0:
                        break
                    best_next = -1
                    best_overlap = -1
                    
                    for next_idx in range(num_perms):
                        if visited[next_idx]:
                            continue
                        ov = overlap_matrix[current][next_idx]
                        if ov > best_overlap:
                            best_overlap = ov
                            best_next = next_idx
                            
                    if best_next == -1:
                        break
                    visited[best_next] = True
                    order.append(best_next)
                    current = best_next
                    
                if len(order) < num_perms:
                    continue
                
                # Calculate total length
                total_len = 2*(k-1) + 1
                
                for i in range(1, len(order)):
                    p_prev_idx = order[i-1]
                    p_curr_idx = order[i]
                    ov = overlap_matrix[p_prev_idx][p_curr_idx]
                    total_len += (2*(k-1) + 1) - ov
                
                if total_len < best_len:
                    best_len = total_len
                    best_order = order[:][:]
                    
                # 2-opt local improvement
                if time.time() < deadline - 15.0:
                    improved = True
                    while improved:
                        improved = False
                        for i in range(1, len(order) - 1):
                            if time.time() > deadline - 10.0:
                                break
                            for j in range(i + 1, len(order)):
                                if time.time() > deadline - 10.0:
                                    break
                                # Try reversing order[i:j+1]
                                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                                
                                # Calculate delta in total length
                                old_cost = 0
                                new_cost = 0
                                
                                # Cost before segment
                                if i > 0:
                                    old_cost += (2*(k-1) + 1) - overlap_matrix[order[i-1]][order[i]]
                                if j < len(order) - 1:
                                    old_cost += (2*(k-1) + 1) - overlap_matrix[order[j]][order[j+1]]
                                for a in range(i, j):
                                    old_cost += (2*(k-1) + 1) - overlap_matrix[order[a]][order[a+1]]
                                
                                if i > 0:
                                    new_cost += (2*(k-1) + 1) - overlap_matrix[order[i-1]][new_order[i]]
                                if j < len(order) - 1:
                                    new_cost += (2*(k-1) + 1) - overlap_matrix[new_order[j]][order[j+1]]
                                for a in range(i, j):
                                    new_cost += (2*(k-1) + 1) - overlap_matrix[new_order[a]][new_order[a+1]]
                                
                                if new_cost < old_cost:
                                    order = new_order
                                    improved = True
                                    if total_len > best_len:
                                        total_len = best_len
                                    break
                            if improved:
                                break
                        
                        if total_len < best_len:
                            best_len = total_len
                            best_order = order[:][:]
                    
        if best_order is None:
            best_order = list(range(num_perms))
            
        # Construct new word
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                # Find maximum overlap between end of new_word and start of block
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word
        
        # Early exit if we're out of time
        if time.time() > deadline - 10.0:
            break
    
    # Verify we have a valid superpermutation
    # If not, fall back to a known valid construction
    def is_valid_superperm(word, n):
        if len(word) < n:
            return False
        perms_found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                perms_found.add(window)
        return len(perms_found) == math.factorial(n)
    
    if not is_valid_superperm(word, n):
        # Fallback: use the standard recursive construction
        word = [1]
        for k in range(2, n + 1):
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for idx, p in enumerate(perms):
                block = list(p) + [k] + list(p)
                if idx == 0:
                    new_word = block
                else:
                    max_overlap = min(len(new_word), len(block))
                    overlap = 0
                    for o in range(min(k, max_overlap), -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if new_word[-o:] == block[:o]:
                            overlap = o
                            break
                    new_word += block[overlap:]
            word = new_word
    
    return "".join(map(str, word))

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols using the standard
    recursive construction, with Simulated Annealing optimization for n=7
    to reorder the permutations of 1..6 to minimize the total length.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        # Optimization for m=7 (adding 7 to perms of 1..6)
        if m == 7:
            # We have 720 permutations.
            # Use Simulated Annealing to find optimal ordering.
            
            num_perms = len(perms)
            
            # Precompute overlap matrix
            def get_overlap(p1, p2):
                # p1, p2 are tuples
                for k in range(6, 0, -1):
                    if p1[-k:] == p2[:k]:
                        return k
                return 0
            
            # Overlap from start (end of S_6) to each perm
            start_end = tuple(word[-6:])
            start_ov = [0] * num_perms
            for i in range(num_perms):
                start_ov[i] = get_overlap(start_end, perms[i])
            
            # Pairwise overlaps
            ov_matrix = [[0] * num_perms for _ in range(num_perms)]
            for i in range(num_perms):
                for j in range(num_perms):
                    if i != j:
                        ov_matrix[i][j] = get_overlap(perms[i], perms[j])
            
            # Fitness function: maximize total overlap
            def calc_fitness(order):
                total = start_ov[order[0]]
                for i in range(len(order) - 1):
                    total += ov_matrix[order[i]][order[i+1]]
                return total
            
            # Initialize order (standard order)
            order = list(range(num_perms))
            current_score = calc_fitness(order)
            best_order = order[:]
            best_score = current_score
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.9995
            
            while time.time() < deadline - 5:
                if T < T_min:
                    break
                
                # Pick a random 2-opt move: reverse segment [i, j]
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                # Calculate delta score for reversing order[i:j+1]
                # Old links: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New links: (i-1, j), (j, j-1), ..., (i+1, i), (i, i-1)
                
                old_links = 0
                new_links = 0
                
                # Boundary 1: i-1 to i -> i-1 to j
                if i == 0:
                    old_links += start_ov[order[i]]
                    new_links += start_ov[order[j]]
                else:
                    old_links += ov_matrix[order[i-1]][order[i]]
                    new_links += ov_matrix[order[i-1]][order[j]]
                    
                # Boundary 2: j to j+1 -> i to j+1
                if j == num_perms - 1:
                    # No right neighbor
                    pass
                else:
                    old_links += ov_matrix[order[j]][order[j+1]]
                    new_links += ov_matrix[order[i]][order[j+1]]
                    
                # Internal links: (k, k+1) becomes (k+1, k) for k in i..j-1
                # Sum ov[order[k]][order[k+1]] for k in i..j-1
                # Subtract Sum ov[order[k+1]][order[k]] for k in i..j-1
                
                # Note: ov_matrix is not symmetric, so we must compute both sums.
                # However, we can iterate once.
                internal_old = 0
                internal_new = 0
                for k in range(i, j):
                    internal_old += ov_matrix[order[k]][order[k+1]]
                    internal_new += ov_matrix[order[k+1]][order[k]]
                    
                delta = (new_links + internal_new) - (old_links + internal_old)
                
                # Accept if improvement or with probability exp(delta/T)
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Perform the reversal
                    order[i:j+1] = reversed(order[i:j+1])
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_order = order[:]
                
                T *= cooling_rate
            
            order = best_order
            
            # Use the optimized order
            for idx in order:
                p = perms[idx]
                block = list(p) + [m] + list(p)
                
                # Find overlap
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        else:
            # Standard construction for m < 7
            for p in perms:
                block = list(p) + [m] + list(p)
                
                max_possible_overlap = min(len(new_word), len(block), m - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if new_word[len(new_word)-k:] == block[:k]:
                        overlap = k
                        break
                
                new_word.extend(block[overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

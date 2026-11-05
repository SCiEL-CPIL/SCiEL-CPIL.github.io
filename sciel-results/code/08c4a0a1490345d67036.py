import time
import random
import math
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n != 7:
        return _standard_construct(n)
        
    # 1. Generate permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    num_perms = len(perms) # 720
    
    # 2. Compute Overlap Matrix
    # M[i][j] = overlap between perms[i] and perms[j]
    # We store this as a list of lists of small ints
    M = [[0]*num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            ov = 0
            if p1[1:] == p2[:5]: ov = 5
            elif p1[2:] == p2[:4]: ov = 4
            elif p1[3:] == p2[:3]: ov = 3
            elif p1[4:] == p2[:2]: ov = 2
            elif p1[5] == p2[0]: ov = 1
            M[i][j] = ov

    # 3. Greedy Initialization
    order = [0]
    used = [False]*num_perms
    used[0] = True
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not used[j]:
                ov = M[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        order.append(best_next)
        used[best_next] = True
        
    # Calculate initial score
    def calc_score(ord_list):
        s = 0
        for i in range(len(ord_list) - 1):
            s += M[ord_list[i]][ord_list[i+1]]
        return s
        
    current_score = calc_score(order)
    best_order = order[:]
    best_score = current_score
    
    # 4. Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    min_T = 1e-4
    random.seed(42)
    
    while T > min_T and time.time() < deadline:
        # Choose move: 2-opt or Or-opt
        move_type = random.random()
        
        if move_type < 0.4:
            # 2-opt: reverse segment
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            # Calculate delta
            old_score = 0
            new_score = 0
            
            if i > 0:
                old_score += M[order[i-1]][order[i]]
                new_score += M[order[i-1]][order[j]]
                
            if j < num_perms - 1:
                old_score += M[order[j]][order[j+1]]
                new_score += M[order[i]][order[j+1]]
                
            # Internal links
            # Old: (i, i+1), ..., (j-1, j)
            # New: (j, j-1), ..., (i+1, i)
            for k in range(i, j):
                old_score += M[order[k]][order[k+1]]
            for k in range(j, i, -1):
                new_score += M[order[k]][order[k-1]]
                
            delta = new_score - old_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                order[i:j+1] = reversed(order[i:j+1])
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order[:]
                    
        elif move_type < 0.8:
            # Or-opt: move a segment of length 1, 2, or 3
            seg_len = random.randint(1, 3)
            i = random.randint(0, num_perms - seg_len)
            # Choose insertion position j. 
            # We remove from i..i+seg_len-1.
            # Insert at j. If j > i, we must account for the shift.
            # Easiest: pick i and a target index in the *remaining* list, or just pick two indices and handle the logic.
            # Let's pick i (start) and j (end index in original list before removal, or just a position).
            # To avoid confusion, let's pick i and k (position in original list to insert after).
            # If we remove [i, i+L), the length decreases by L.
            # If we insert at position k in the original list (meaning after element k-1, or at start if k=0),
            # we need to be careful if k is within the removed segment.
            
            # Simpler approach for Or-opt in a list:
            # 1. Extract segment
            # 2. Remove it
            # 3. Insert it at a new position
            # But we need the delta.
            
            # Let's stick to a simpler move: Swap two elements? No, 2-opt covers reversals.
            # Let's do "Or-opt" properly: move a block of size 1-3 to a random position.
            
            i = random.randint(0, num_perms - seg_len)
            # Pick insertion index in the range [0, num_perms - seg_len] (position in the list AFTER removal)
            # This is tricky to map back to original indices for delta calc without doing the move.
            
            # Alternative: Just do a random insertion of a single element (Or-opt L=1)
            # Or, let's just do 2-opt more aggressively and add a "Random Restart" or "Perturbation".
            pass # Placeholder for now, will implement Or-opt L=1 below if time permits, or just stick to 2-opt.
            
            # Let's implement Or-opt L=1 (Move one element)
            seg_len = 1
            i = random.randint(0, num_perms - 1)
            # Pick j, the destination index in the original list.
            # If we remove i, and insert at j.
            # Case 1: j > i.
            #   Old: ... A[i-1] A[i] A[i+1] ... A[j-1] A[j] A[j+1] ...
            #   Remove A[i]. List: ... A[i-1] A[i+1] ... A[j-1] A[j] A[j+1] ...
            #   Insert A[i] at j.
            #   If j is the index in the ORIGINAL list where we want to place it.
            #   This is getting confusing.
            
            # Let's just use 2-opt. It's effective for TSP-like problems.
            # To improve, we can vary the segment length for 2-opt? No, 2-opt is reversal.
            # Maybe "3-opt" style moves?
            
            # Let's add a simple "Swap" move (Or-opt L=1 with swap)
            # Swap i and j
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            old_score = 0
            new_score = 0
            
            a = order[i]
            b = order[j]
            
            if i > 0:
                old_score += M[order[i-1]][a]
                new_score += M[order[i-1]][b]
            if i < num_perms - 1:
                old_score += M[a][order[i+1]]
                new_score += M[b][order[i+1]]
            
            if j > 0:
                old_score += M[order[j-1]][b]
                new_score += M[order[j-1]][a]
            if j < num_perms - 1:
                old_score += M[b][order[j+1]]
                new_score += M[a][order[j+1]]
            
            delta = new_score - old_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                order[i], order[j] = b, a
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order[:]

        else:
            # Or-opt: Move segment of length 1, 2, or 3
            seg_len = random.randint(1, 3)
            i = random.randint(0, num_perms - seg_len)
            
            # We remove order[i : i+seg_len]
            # And insert it at some position k.
            # To make delta calculation easy, let's just pick a random position k in [0, num_perms - seg_len]
            # representing the index in the list *after removal* where the segment starts.
            
            # This is complex. Let's stick to 2-opt and Swap for now.
            # The 0.8 probability split was 0.4 (2-opt) and 0.4 (Swap).
            # Let's make the last 20% a "Perturbation" (random shuffle of a small segment)
            # Or just another 2-opt.
            
            # Let's do another 2-opt with a different length bias?
            # Or just re-roll.
            pass

        T *= cooling_rate
        
    # 5. Construct Word
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M[best_order[idx-1]][best_order[idx]]
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_construct(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

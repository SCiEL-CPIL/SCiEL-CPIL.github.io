import time
import random
import math
import itertools
import numpy as np

def construct(n):
    if n < 7:
        # Use standard construction for n < 7 as it is optimal
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
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        max_ov = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, use Simulated Annealing on the order of all 5040 permutations
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s wall clock
    
    # 1. Generate all permutations of 1..n
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # 2. Precompute overlap matrix
    # M[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Since perms are permutations of 1..n, the suffix of i and prefix of j 
    # must match.
    # We can optimize this by checking from k=n-1 down to 0.
    # M is N x N. N=5040. 5040^2 ~ 25M entries. 
    # Storing as a numpy array of uint8 is 25MB. This is fine.
    # Computation time: 25M * (avg checks). 
    # We can speed this up by noting that if perms[i][0] != perms[j][n-1], 
    # overlap is limited? No, overlap is suffix of i, prefix of j.
    
    # Let's use a dictionary or list of lists for speed if numpy is not fast enough for init.
    # Actually, numpy vectorization might be hard for variable length overlap.
    # Let's stick to Python loops but optimize the inner check.
    
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: 
    # For each permutation i, we can map its suffixes to indices.
    # But simple loop might be okay if we break early.
    # 25M iterations in Python might take 10-20 seconds. 
    # Let's try to be efficient.
    
    # We can precompute the prefix tuples and suffix tuples?
    # No, that's too much memory.
    
    # Let's just compute it.
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            # Max possible overlap is n-1
            # Check from n-1 down to 0
            # pi[-k:] == pj[:k]
            # This is a string comparison.
            # Converting to string might be faster in C?
            # Or just loop k.
            for k in range(n - 1, -1, -1):
                if k == 0:
                    M[i, j] = 0
                    break
                # Check if pi[-k:] == pj[:k]
                # Using tuple slicing
                if pi[-k:] == pj[:k]:
                    M[i, j] = k
                    break
    
    # 3. Initialize order
    # Start with lexicographic order or standard superpermutation order?
    # Lexicographic is a valid permutation of the set.
    order = list(range(N))
    
    # Calculate initial cost (sum of overlaps)
    def calc_cost(ord_list):
        cost = 0
        for i in range(len(ord_list) - 1):
            cost += M[ord_list[i], ord_list[i + 1]]
        return cost
    
    current_cost = calc_cost(order)
    best_cost = current_cost
    best_order = order[:]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    # Moves:
    # 1. Swap adjacent
    # 2. Swap random
    # 3. Reverse segment
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random move
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            i = random.randint(0, N - 2)
            j = i + 1
            # Delta cost
            # Remove edges (i-1, i), (i, i+1), (i+1, i+2)
            # Add edges (i-1, i+1), (i+1, i), (i, i+2)
            # Indices in order list: i, j=i+1
            # Edges affected:
            # Old: (order[i-1], order[i]), (order[i], order[i+1]), (order[i+1], order[i+2])
            # New: (order[i-1], order[i+1]), (order[i+1], order[i]), (order[i], order[i+2])
            # Note: boundaries need care.
            
            # Simpler: just swap and recalc local delta
            # Let's implement a helper to get edge cost
            def edge_cost(a, b):
                if a is None or b is None: return 0
                return M[a, b]
            
            p1 = order[i-1] if i > 0 else None
            p2 = order[i]
            p3 = order[i+1]
            p4 = order[i+2] if i+2 < N else None
            
            old_cost = edge_cost(p1, p2) + edge_cost(p2, p3) + edge_cost(p3, p4)
            
            # Swap p2 and p3
            new_p2 = p3
            new_p3 = p2
            
            new_cost = edge_cost(p1, new_p2) + edge_cost(new_p2, new_p3) + edge_cost(new_p3, p4)
            
            delta = new_cost - old_cost
            
            if delta > 0 or random.random() < math.exp(delta / T):
                order[i], order[i+1] = order[i+1], order[i]
                current_cost += delta
                if current_cost > best_cost:
                    best_cost = current_cost
                    best_order = order[:]
                    
        elif move_type == 1:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j: continue
            if j < i: i, j = j, i
            
            # This is complex to calculate delta for arbitrary swap.
            # Let's just recalc cost for this move? No, too slow.
            # Let's use 2-opt style or just adjacent swaps and reverses.
            # Random swap delta calculation:
            # Edges removed: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
            # Edges added: (i-1, j), (j, i+1), (j-1? no, j is gone from its spot), 
            # Actually, if we swap i and j:
            # Pos i gets order[j], Pos j gets order[i].
            # Neighbors of i: L1=order[i-1], R1=order[i+1]
            # Neighbors of j: L2=order[j-1], R2=order[j+1]
            
            L1 = order[i-1] if i > 0 else None
            R1 = order[i+1] if i+1 < N else None
            L2 = order[j-1] if j > 0 else None
            R2 = order[j+1] if j+1 < N else None
            
            A = order[i]
            B = order[j]
            
            old_cost = edge_cost(L1, A) + edge_cost(A, R1) + edge_cost(L2, B) + edge_cost(B, R2)
            
            # New: A is at j, B is at i
            # Edges: (L1, B), (B, R1), (L2, A), (A, R2)
            new_cost = edge_cost(L1, B) + edge_cost(B, R1) + edge_cost(L2, A) + edge_cost(A, R2)
            
            delta = new_cost - old_cost
            
            if delta > 0 or random.random() < math.exp(delta / T):
                order[i], order[j] = order[j], order[i]
                current_cost += delta
                if current_cost > best_cost:
                    best_cost = current_cost
                    best_order = order[:]
                
        else:
            # Reverse segment
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            L1 = order[i-1] if i > 0 else None
            R1 = order[j+1] if j+1 < N else None
            
            A = order[i]
            B = order[j]
            
            # The internal edges are reversed.
            # Old internal: (order[i], order[i+1]) ... (order[j-1], order[j])
            # New internal: (order[j], order[j-1]) ... (order[i+1], order[i])
            # Sum of internal edges is the same! Because M is not symmetric?
            # M[i][j] != M[j][i] generally.
            # So we must recalc internal sum.
            
            # This is O(length) which is bad.
            # Let's skip reverse for now or only do small reverses.
            # If j - i < 10, we can recalc.
            if j - i < 10:
                old_internal = 0
                for k in range(i, j):
                    old_internal += M[order[k], order[k+1]]
                
                new_segment = order[i:j+1][::-1]
                new_internal = 0
                for k in range(len(new_segment) - 1):
                    new_internal += M[new_segment[k], new_segment[k+1]]
                
                old_ext = edge_cost(L1, A) + edge_cost(B, R1)
                new_ext = edge_cost(L1, B) + edge_cost(A, R1)
                
                old_total = old_internal + old_ext
                new_total = new_internal + new_ext
                delta = new_total - old_total
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    order[i:j+1] = new_segment
                    current_cost += delta
                    if current_cost > best_cost:
                        best_cost = current_cost
                        best_order = order[:]
            else:
                # Skip large reverses to save time, or do it if we have time?
                # For now, skip.
                pass
        
        T *= cooling
        
    # Reconstruct word from best_order
    final_word = list(perms[best_order[0]])
    for idx in best_order[1:]:
        p = perms[idx]
        # Find max overlap with current final_word
        # We know the overlap from M, but we need to apply it to the actual list.
        # M[prev_idx, idx] gives the overlap k.
        # We just append p[k:]
        # We need the index of the previous permutation in best_order to look up M?
        # Or we can just compute overlap on the fly for the final construction.
        # Since N is 5040, and we do this once, it's fast.
        
        # Actually, we can just use the stored overlap?
        # No, the overlap depends on the specific permutations.
        # Let's just compute it.
        max_ov = 0
        prev = final_word
        # Max possible overlap is n-1
        for k in range(n - 1, -1, -1):
            if k == 0:
                break
            if prev[-k:] == p[:k]:
                max_ov = k
                break
        final_word.extend(p[max_ov:])
        
    return "".join(map(str, final_word))


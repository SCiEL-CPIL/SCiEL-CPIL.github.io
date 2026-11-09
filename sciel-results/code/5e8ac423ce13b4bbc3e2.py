import time
import sys
import itertools
import random

def construct(n):
    if n <= 6:
        # Use standard recursive construction for small n
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n != 7:
        # Fallback for other n, though task specifies n=7
        return "".join(map(str, list(range(1, n+1))))

    # n = 7 specific optimization
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate all permutations of 1..7
    perms = list(itertools.permutations(range(1, 8)))
    N_PERMS = len(perms) # 5040
    
    # Map permutation to index for fast lookup?
    # Not strictly needed if we work with tuples directly in sets,
    # but useful for DP state indexing if we map cycles to IDs.
    
    # 2. Identify cycles of left rotations
    # L(p) = (p[1], p[2], p[3], p[4], p[5], p[6], p[0])
    visited = set()
    cycles = []
    
    for p in perms:
        if p in visited:
            continue
        cycle = []
        cur = p
        while cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            cur = (cur[1], cur[2], cur[3], cur[4], cur[5], cur[6], cur[0])
        cycles.append(cycle)
        
    N_CYCLES = len(cycles) # Should be 720
    
    # 3. Precompute M_max[A][B]
    # M_max[A][B] = max overlap between exit of A and entry of B
    # exit_A(x) is the last node in the cycle if we start at x?
    # If cycle is [c0, c1, ..., c6], and we enter at ci, we visit ci, ci+1, ..., c6, c0, ..., ci-1.
    # The last visited is c(i-1) % 7.
    # So exit_A(ci) = c(i-1)%7.
    
    # Let's store cycles as lists of permutations.
    # To speed up overlap calculation, we can precompute a dictionary of overlaps?
    # Or just calculate on the fly.
    
    # Let's calculate M_max matrix
    # This might be slow. Let's try to optimize.
    # We can precompute overlap for all pairs of permutations?
    # 5040x5040 is too big.
    
    # Let's just compute M_max directly.
    # 720x720 matrix.
    # For each A, B:
    #   max_ov = 0
    #   for i in 7:
    #     exit_p = cycles[A][(i-1)%7]
    #     for j in 7:
    #       entry_q = cycles[B][j]
    #       ov = overlap(exit_p, entry_q)
    #       if ov > max_ov: max_ov = ov
    
    # To make this faster, we can precompute overlaps for all pairs of permutations that are in the same "class"?
    # No, just run it.
    
    # Helper function for overlap
    def calc_overlap(p, q):
        # p and q are tuples of length 7
        # Max overlap 6
        for k in range(6, 0, -1):
            if p[7-k:] == q[:k]:
                return k
        return 0
        
    M_max = [[0]*N_CYCLES for _ in range(N_CYCLES)]
    
    # Optimize by precomputing exits and entries
    exits = []
    entries = []
    for c in cycles:
        # c is [c0, ..., c6]
        # If we enter at ci, exit is c(i-1)%7
        # So the set of possible exits is just the cycle itself.
        # The set of possible entries is just the cycle itself.
        pass
        
    # Actually, the set of exits for cycle A is exactly the set of elements in cycle A.
    # The set of entries for cycle B is exactly the set of elements in cycle B.
    # So M_max[A][B] = max(overlap(p, q) for p in cycle A, q in cycle B).
    
    # This is simpler.
    for i in range(N_CYCLES):
        ci = cycles[i]
        for j in range(N_CYCLES):
            cj = cycles[j]
            max_ov = 0
            for p in ci:
                for q in cj:
                    ov = calc_overlap(p, q)
                    if ov > max_ov:
                        max_ov = ov
                        if max_ov == 6: # Max possible
                            break
                if max_ov == 6:
                    break
            M_max[i][j] = max_ov
            
    if time.time() > deadline - 30:
        # If precomputation took too long, fallback to standard
        pass

    # 4. TSP Heuristic on N_CYCLES nodes
    # Greedy initialization
    order = [0]
    visited_c = [False]*N_CYCLES
    visited_c[0] = True
    
    for _ in range(N_CYCLES - 1):
        last = order[-1]
        best_next = -1
        best_cost = -1
        for j in range(N_CYCLES):
            if not visited_c[j]:
                cost = M_max[last][j]
                if cost > best_cost:
                    best_cost = cost
                    best_next = j
        order.append(best_next)
        visited_c[best_next] = True
        
    # 2-Opt improvement on cycles
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 20:
        improved = False
        iterations += 1
        for i in range(1, N_CYCLES - 1):
            if time.time() > deadline - 20:
                break
            for j in range(i + 1, N_CYCLES):
                if time.time() > deadline - 20:
                    break
                # Reverse segment i..j
                a = order[i-1]
                b = order[i]
                c = order[j]
                d = order[j+1] if j+1 < N_CYCLES else -1
                
                if d == -1:
                    old_cost = M_max[a][b]
                    new_cost = M_max[a][c]
                    if new_cost > old_cost:
                        order[i:] = order[i:][::-1]
                        improved = True
                else:
                    old_cost = M_max[a][b] + M_max[c][d]
                    new_cost = M_max[a][c] + M_max[b][d]
                    if new_cost > old_cost:
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True
                        
    # 5. DP to find optimal entry points
    # DP[k][y] = max score for first k cycles, ending with entry y in cycle k
    # Score is sum of overlaps.
    
    # We need W[A][B](x, y) = overlap(exit_A(x), y)
    # exit_A(x) is the node in cycle A that precedes x in the cycle order?
    # If cycle is [c0, c1, c2, c3, c4, c5, c6].
    # If we enter at c_i, we visit c_i, c_{i+1}, ..., c_6, c_0, ..., c_{i-1}.
    # The last visited is c_{i-1} (mod 7).
    # So exit_A(c_i) = c_{i-1}.
    
    # Let's precompute the exit for each node in each cycle.
    # exit_map[cycle_idx][node_idx] = node_idx_of_exit
    # Actually, we just need the permutation tuple.
    
    # DP Array: list of lists, DP[k] is a dict or list of 7 floats.
    # Let's use lists of length 7.
    
    DP = []
    # For the first cycle, score is 0 (no incoming edge)
    # But we need to track the predecessor to reconstruct?
    # For now, just find the max score.
    
    # DP[0][j] = 0 for all j in 0..6
    DP.append([0.0] * 7)
    prev_choice = [None] * 7 # For backtracking, store which j in prev cycle led to this
    
    for k in range(1, N_CYCLES):
        curr_cycle_idx = order[k]
        prev_cycle_idx = order[k-1]
        
        curr_cycle = cycles[curr_cycle_idx]
        prev_cycle = cycles[prev_cycle_idx]
        
        curr_DP = [0.0] * 7
        curr_prev = [0] * 7
        
        for y_idx in range(7):
            y = curr_cycle[y_idx]
            max_val = -1.0
            best_x_idx = 0
            
            for x_idx in range(7):
                x = prev_cycle[x_idx]
                # Exit from x in prev cycle
                # The node preceding x in the cycle list
                # Cycle list is [c0, c1, c2, c3, c4, c5, c6]
                # If x is c_i, exit is c_{i-1}
                exit_idx = (x_idx - 1) % 7
                exit_p = prev_cycle[exit_idx]
                
                ov = calc_overlap(exit_p, y)
                val = DP[k-1][x_idx] + ov
                
                if val > max_val:
                    max_val = val
                    best_x_idx = x_idx
                    
            curr_DP[y_idx] = max_val
            curr_prev[y_idx] = best_x_idx
            
        DP.append(curr_DP)
        # Store prev choices for backtracking?
        # We need to store the choice for each step.
        # Let's store a list of lists of best_x_idx.
        if k == 1:
            backtrack_choices = [curr_prev]
        else:
            backtrack_choices.append(curr_prev)
            
    # Reconstruct path
    # Start from the last cycle, pick the best entry
    last_k = N_CYCLES - 1
    best_last_entry_idx = 0
    max_final_score = -1.0
    for i in range(7):
        if DP[last_k][i] > max_final_score:
            max_final_score = DP[last_k][i]
            best_last_entry_idx = i
            
    entries = [0] * N_CYCLES
    entries[last_k] = best_last_entry_idx
    
    for k in range(N_CYCLES - 1, 0, -1):
        prev_idx = backtrack_choices[k-1][entries[k]] # backtrack_choices[k-1] corresponds to step k-1 -> k
        entries[k-1] = prev_idx
        
    # Construct the final permutation sequence
    seq = []
    for k in range(N_CYCLES):
        cycle_idx = order[k]
        entry_idx = entries[k]
        cycle = cycles[cycle_idx]
        # Traverse the cycle starting from entry_idx
        for step in range(7):
            idx = (entry_idx + step) % 7
            seq.append(cycle[idx])
            
    # Convert seq to word
    word = []
    for i, p in enumerate(seq):
        if i == 0:
            word.extend(list(p))
        else:
            # Overlap with previous
            prev_p = seq[i-1]
            k = calc_overlap(prev_p, p)
            word.extend(list(p[k:]))
            
    return "".join(map(str, word))

import time
import random
from itertools import permutations
import numpy as np
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit

    # For small n, use the standard construction which is optimal
    if n <= 3:
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
                for k in range(min(len(new), len(block)), 0, -1):
                    if new[len(new) - k:] == block[:k]:
                        overlap = k
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 4, use optimization
    # We model the problem as finding an ordering of permutations of 1..n-1
    # such that the total overlap between consecutive blocks is maximized.
    # Block for permutation p of 1..n-1 is: p + [n] + p
    
    # 1. Generate all permutations of 1..n-1
    base_symbols = list(range(1, n))
    perms = list(permutations(base_symbols))
    M = len(perms)
    
    # 2. Convert to tuples for hashing and comparison
    perm_tuples = [tuple(p) for p in perms]
    
    # 3. Compute overlap matrix efficiently
    # O[i][j] = max k such that perm_tuples[i][-k:] == perm_tuples[j][:k]
    # We compute this by checking each possible overlap length from longest to shortest
    # and using a dictionary to group permutations by their prefix/suffix of that length.
    
    overlap_matrix = np.zeros((M, M), dtype=np.int8)
    
    for k in range(n - 1, 0, -1):
        suffix_map = {}
        prefix_map = {}
        
        for idx, p in enumerate(perm_tuples):
            suff = p[-k:]
            pref = p[:k]
            if suff not in suffix_map:
                suffix_map[suff] = []
            suffix_map[suff].append(idx)
            
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(idx)
        
        for suff, i_list in suffix_map.items():
            if suff in prefix_map:
                j_list = prefix_map[suff]
                for i in i_list:
                    for j in j_list:
                        overlap_matrix[i][j] = k

    # 4. Lin-Kernighan Heuristic
    
    def get_total_overlap(order):
        total = 0
        for i in range(len(order) - 1):
            total += overlap_matrix[order[i]][order[i+1]]
        return total

    def lin_kernighan(order, time_limit):
        """
        Applies Lin-Kernighan style 2-opt moves to maximize total overlap.
        """
        current_order = order[:]
        current_overlap = get_total_overlap(current_order)
        best_order = current_order[:]
        best_overlap = current_overlap
        improved = True
        no_improve_count = 0
        
        while time.time() < time_limit and improved:
            improved = False
            no_improve_count = 0
            # Iterate over all pairs (i, j) with i < j
            # To avoid O(M^2) which is too slow for M=720 (720^2 = 518400, feasible if optimized)
            # But we need to be careful with Python loops.
            # Let's try a subset of moves or a more efficient iteration if M is large.
            # For n=7, M=720. 720^2 is ~500k. 500k iterations in Python might take ~1-2s.
            # We can do a few passes.
            
            for i in range(M - 1):
                if time.time() > time_limit:
                    break
                node_i = current_order[i]
                node_j = current_order[i+1]
                
                for j in range(i + 2, M):
                    node_k = current_order[j]
                    node_l = current_order[j+1]
                    
                    # We are considering removing edges (i, i+1) and (j, j+1)
                    # Old edges: (node_i, node_j) and (node_k, node_l)
                    old_cost = overlap_matrix[node_i][node_j] + overlap_matrix[node_k][node_l]
                    
                    # Option 1: Standard 2-opt reconnection
                    # New edges: (node_i, node_k) and (node_j, node_l)
                    new_cost_1 = overlap_matrix[node_i][node_k] + overlap_matrix[node_j][node_l]
                    
                    # Option 2: Lin-Kernighan swap (reverse segment i+1 to j)
                    # Actually, standard 2-opt on a path reverses the segment between i+1 and j.
                    # The edges become (node_i, node_k) and (node_j, node_l) is NOT correct for reversal.
                    # If we reverse segment [i+1, j], the edges are:
                    # Left boundary: node_i connects to node_k (was node_j)
                    # Right boundary: node_j connects to node_l (was node_k)
                    # Wait, the segment is i+1 ... j.
                    # Original: ... node_i -> node_{i+1} ... node_j -> node_l ...
                    # Reversed: ... node_i -> node_j ... node_{i+1} -> node_l ...
                    # So new edges are (node_i, node_j) and (node_{i+1}, node_l).
                    # This is exactly Option 1 above? No.
                    # Original edges: (node_i, node_{i+1}) and (node_j, node_l).
                    # New edges: (node_i, node_j) and (node_{i+1}, node_l).
                    
                    new_cost_2 = overlap_matrix[node_i][node_j] + overlap_matrix[current_order[i+1]][node_l]
                    
                    # We want to maximize overlap, so we want to maximize new_cost.
                    # But we also need to account for the internal changes?
                    # In TSP, 2-opt delta calculation usually only looks at boundary edges if the internal
                    # structure is preserved (which it is, just reversed).
                    # However, the "cost" of internal edges changes direction.
                    # Overlap is not symmetric! O[u][v] != O[v][u].
                    # So reversing the segment changes the cost of all internal edges.
                    # This makes 2-opt expensive to evaluate exactly in O(1).
                    # For M=720, O(L) evaluation where L is segment length is too slow for many iterations.
                    
                    # Alternative: Just use Swap moves which are O(1) to evaluate delta.
                    # Or, use a simpler heuristic: Random Restart Local Search with Swaps.
                    # The current program uses swaps. Let's stick to swaps but optimize the search.
                    # The "Lin-Kernighan" inspiration might be misleading if we can't afford O(L) 2-opt.
                    # Let's stick to optimized Swap moves.
                    
                    # Actually, let's try a different approach: Simulated Annealing or Tabu Search?
                    # Given time constraints, a robust implementation of Swap-based Local Search
                    # with multiple restarts is safer.
                    
                    pass # Skip 2-opt for now due to O(L) cost in Python

            # If we didn't improve with 2-opt (skipped), do swaps
            # Let's do a pass of random swaps
            num_swaps = 10000
            for _ in range(num_swaps):
                if time.time() > time_limit:
                    break
                i, j = random.sample(range(M), 2)
                if i == j: continue
                
                # Calculate delta for swap i, j
                # We already have a delta_swap function in the previous version, let's reuse the logic
                # But to keep it simple and fast, we can just compute the change in the 4 affected edges.
                
                a = current_order[i]
                b = current_order[j]
                
                # Neighbors of i
                prev_i = current_order[i-1] if i > 0 else None
                next_i = current_order[i+1] if i < M-1 else None
                
                # Neighbors of j
                prev_j = current_order[j-1] if j > 0 else None
                next_j = current_order[j+1] if j < M-1 else None
                
                # Be careful if i and j are adjacent
                if abs(i - j) == 1:
                    # Handle adjacent swap separately or skip for simplicity in this pass?
                    # Let's just skip adjacent swaps in the random pass to avoid complex boundary logic
                    # or implement it correctly.
                    # For now, let's just do non-adjacent swaps to keep code simple and bug-free.
                    continue

                old_ov = 0
                if prev_i is not None: old_ov += overlap_matrix[prev_i][a]
                if next_i is not None: old_ov += overlap_matrix[a][next_i]
                if prev_j is not None: old_ov += overlap_matrix[prev_j][b]
                if next_j is not None: old_ov += overlap_matrix[b][next_j]
                
                new_ov = 0
                if prev_i is not None: new_ov += overlap_matrix[prev_i][b]
                if next_i is not None: new_ov += overlap_matrix[b][next_i]
                if prev_j is not None: new_ov += overlap_matrix[prev_j][a]
                if next_j is not None: new_ov += overlap_matrix[a][next_j]
                
                delta = new_ov - old_ov
                if delta > 0:
                    current_order[i], current_order[j] = b, a
                    current_overlap += delta
                    improved = True
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_order = current_order[:]
        
        return best_order, best_overlap

    # Initial Solution: Greedy
    # Start with a random node, then always pick the unvisited node with max overlap.
    # Do this a few times to find a good start.
    
    best_initial_order = None
    best_initial_overlap = -1
    
    for _ in range(5): # 5 restarts for initial solution
        if time.time() > deadline: break
        
        start_node = random.randint(0, M - 1)
        unvisited = set(range(M))
        unvisited.remove(start_node)
        current_order = [start_node]
        last = start_node
        
        while unvisited:
            max_ov = -1
            best_next = None
            # To speed up, we could pre-sort or use a heap, but M is small enough
            # However, checking all unvisited is O(M^2) total.
            # M=720, so 720*720 = 518k operations. This is fast in numpy/python if optimized.
            # Let's just do a linear scan.
            for j in unvisited:
                ov = overlap_matrix[last][j]
                if ov > max_ov:
                    max_ov = ov
                    best_next = j
            
            current_order.append(best_next)
            unvisited.remove(best_next)
            last = best_next
            
        ov = get_total_overlap(current_order)
        if ov > best_initial_overlap:
            best_initial_overlap = ov
            best_initial_order = current_order

    # Run Lin-Kernighan (or Swap-based LS) on the best initial solution
    final_order, final_overlap = lin_kernighan(best_initial_order, deadline)
    
    # Reconstruct string from final_order
    # Block for permutation p is p + [n] + p
    # We need to concatenate them with overlaps.
    
    if not final_order:
        return ""
    
    # Build the word by concatenating blocks with overlaps
    result_list = []
    
    # First block
    first_p = perm_tuples[final_order[0]]
    first_block = list(first_p) + [n] + list(first_p)
    result_list.extend(first_block)
    
    for idx in range(1, M):
        prev_idx = final_order[idx-1]
        curr_idx = final_order[idx]
        
        curr_p = perm_tuples[curr_idx]
        curr_block = list(curr_p) + [n] + list(curr_p)
        
        ov = overlap_matrix[prev_idx][curr_idx]
        
        # Append curr_block[ov:]
        result_list.extend(curr_block[ov:])
        
    result_str = "".join(map(str, result_list))
    
    # Verify validity
    seen = set()
    for i in range(len(result_str) - n + 1):
        window = tuple(map(int, result_str[i:i+n]))
        if len(set(window)) == n:
            seen.add(window)
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
    
    if len(seen) == target_count:
        return result_str
    else:
        # Fallback to standard construction if something went wrong
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen_p = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_p:
                    seen_p.add(window)
                    perms_list.append(window)
            new_word = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = 0
                for k in range(min(len(new_word), len(block)), 0, -1):
                    if new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word
        return "".join(map(str, word))


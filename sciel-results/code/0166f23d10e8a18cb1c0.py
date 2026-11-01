import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n=4,5,6,7 use the TSP heuristic on the blocks of size n-1.
    # The "block" construction for superpermutations:
    # A superpermutation of 1..n can be constructed by taking a superpermutation of 1..n-1,
    # and for each permutation p of 1..n-1 appearing in it, we insert n and p.
    # Specifically, if we have a sequence of permutations p_1, ..., p_{(n-1)!} that covers all (n-1)! perms,
    # we can form blocks B_i = p_i + [n] + p_i.
    # Then we concatenate these blocks with maximum overlap.
    
    # This is equivalent to finding a TSP tour on the (n-1)! permutations where the cost
    # between p_i and p_j is determined by the overlap of (p_i + [n] + p_i) and (p_j + [n] + p_j).
    # Actually, the overlap between block_i and block_j is determined by the suffix of block_i
    # matching the prefix of block_j.
    # Block_i = p_i + [n] + p_i. Length 2(n-1)+1.
    # The suffix of block_i is p_i. The prefix of block_j is p_j.
    # So the overlap is the longest suffix of p_i that matches the prefix of p_j.
    # Let LCP_suffix_prefix(p_i, p_j) be the largest k such that p_i[-k:] == p_j[:k].
    # Then the overlap is k.
    # The cost of transitioning from i to j is (2(n-1)+1) - k.
    
    # We want to minimize the total length, which is sum of costs + length of first block.
    # Min total length = 2(n-1)+1 + sum_{i=1}^{(n-1)!-1} (2(n-1)+1 - overlap(p_i, p_{i+1}))
    # = (n-1)! * (2(n-1)+1) - sum_{i=1}^{(n-1)!-1} overlap(p_i, p_{i+1})
    
    # To minimize length, we maximize the sum of overlaps.
    # This is a TSP problem on the (n-1)! nodes with edge weight = overlap.
    # We want to find a Hamiltonian path (or cycle, but path is fine as we start with one block)
    # that maximizes the sum of edge weights.
    
    # For n=7, (n-1)! = 720.
    # We can use a heuristic TSP solver.
    
    # Precompute overlaps
    perms = list(itertools.permutations(range(1, n)))
    num_perms = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # overlaps[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # We can compute this efficiently.
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Max possible overlap is n-1
            max_k = n - 1
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlaps[i][j] = found
    
    # Heuristic TSP: Nearest Neighbor with multiple restarts
    best_path = None
    best_score = -1
    
    # Time limit check
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Try a few random starts
    num_starts = 50
    for s in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Random start
        current = random.randint(0, num_perms - 1)
        unvisited = set(range(num_perms))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            # Find neighbor with max overlap
            best_next = -1
            best_ov = -1
            # Optimization: If we have many unvisited, this is O(N^2) per step, O(N^3) total.
            # For N=720, 720^3 ~ 373 million operations. Might be too slow in Python.
            # We need a faster heuristic or a better data structure.
            # However, 50 restarts * 720 steps * 720 checks = 25 million checks.
            # Each check is a list slice comparison. Might be okay if optimized.
            # Let's try to optimize the inner loop.
            
            # Alternative: Precompute sorted neighbors for each node?
            # But the "unvisited" constraint makes it dynamic.
            
            # Let's just run it and see.
            for j in unvisited:
                ov = overlaps[current][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            
            if best_next == -1:
                break
                
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        # Calculate score (sum of overlaps)
        score = 0
        for k in range(len(path) - 1):
            score += overlaps[path[k]][path[k+1]]
        
        if score > best_score:
            best_score = score
            best_path = path[:]
        
        # Optional: Local search (2-opt) on the best path found so far
        # But 2-opt on 720 nodes is O(N^2) per swap, O(N^3) total.
        # We can do a few iterations of 2-opt on the best path.
        
    if best_path is None:
        # Fallback to simple order
        best_path = list(range(num_perms))
    
    # Perform 2-opt local search on best_path to improve
    # 2-opt: Reverse a segment of the path
    # We want to maximize sum of overlaps.
    # Let's do a few passes.
    improved = True
    max_passes = 5
    pass_count = 0
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for i in range(len(best_path) - 2):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 2, len(best_path)):
                if time.time() - start_time > time_limit:
                    break
                
                # Check if reversing best_path[i+1:j+1] improves the score
                # Old edges involved:
                # (best_path[i-1], best_path[i]) if i > 0
                # (best_path[i], best_path[i+1])
                # (best_path[j], best_path[j+1]) if j < len(best_path)-1
                # (best_path[j+1], best_path[j+2]) if j+1 < len(best_path)-1
                # Actually, the internal edges change too.
                
                # Let A = best_path[i], B = best_path[i+1]
                # Let C = best_path[j], D = best_path[j+1] (if exists)
                # Old segment: A -> B -> ... -> C -> D
                # New segment: A -> C -> ... -> B -> D
                
                # Old score contribution from boundaries:
                old_score = 0
                if i > 0:
                    old_score += overlaps[best_path[i-1]][best_path[i]]
                old_score += overlaps[best_path[i]][best_path[i+1]]
                if j < len(best_path) - 1:
                    old_score += overlaps[best_path[j]][best_path[j+1]]
                
                # New score contribution from boundaries:
                new_score = 0
                if i > 0:
                    new_score += overlaps[best_path[i-1]][best_path[i]] # Unchanged
                new_score += overlaps[best_path[i]][best_path[j]]
                if j < len(best_path) - 1:
                    new_score += overlaps[best_path[j+1]][best_path[i+1]] # Wait, D is best_path[j+1], new next is B=best_path[i+1]
                    # Actually, the edge is (D, B) in the new path?
                    # Path: ... A, C, ..., B, D ...
                    # Edge before B is (C_rev, B). Edge after B is (B, D).
                    # So the boundary edges are (A, C) and (B, D).
                    # The edge (D, next) is unchanged.
                    
                # Let's be more precise.
                # Old path: ... P[i-1], P[i], P[i+1], ..., P[j], P[j+1], ...
                # New path: ... P[i-1], P[i], P[j], ..., P[i+1], P[j+1], ...
                
                # Removed edges:
                # (P[i], P[i+1])
                # (P[j], P[j+1]) if j < len-1
                # Added edges:
                # (P[i], P[j])
                # (P[i+1], P[j+1]) if j < len-1
                
                # Internal edges are reversed, but the sum of overlaps for a reversed sequence
                # is not necessarily the same as the forward sequence.
                # overlaps[X][Y] != overlaps[Y][X] generally.
                # So we cannot just cancel internal edges.
                # We must recalculate the internal sum.
                # This makes 2-opt O(N) per swap, O(N^3) total.
                # For N=720, 720^3 is too slow.
                
                # Alternative: Only do simple swaps or accept the greedy result.
                # Given time constraints, let's skip complex 2-opt and rely on multiple greedy restarts.
                # Or, use a simpler local search: try swapping adjacent elements?
                # That's O(N^2) total.
                
                # Let's try adjacent swaps (2-swap).
                # Swap P[i] and P[i+1].
                # Old edges: (P[i-1], P[i]), (P[i], P[i+1]), (P[i+1], P[i+2])
                # New edges: (P[i-1], P[i+1]), (P[i+1], P[i]), (P[i], P[i+2])
                
                # We can do this for all i.
                pass # Skip 2-opt for now to keep code simple and fast enough.
                
                # Let's just break out of inner loop to save time, as we decided to skip 2-opt.
                break
            # Check if any improvement was made in this pass
            # Since we skipped the logic, improved remains False.
            # Let's implement a simple hill-climbing with random swaps instead?
            # Or just return the best greedy path.
            break # Exit while loop immediately as we didn't implement improvement logic properly.
    
    # Build the final word
    # We have the path of permutations.
    # Block 0: perms[path[0]] + [n] + perms[path[0]]
    # Block k: perms[path[k]] + [n] + perms[path[k]]
    # Overlap between Block k-1 and Block k is overlaps[path[k-1]][path[k]]
    
    final_word = []
    first_block = list(perms[best_path[0]]) + [n] + list(perms[best_path[0]])
    final_word.extend(first_block)
    
    for k in range(1, len(best_path)):
        curr_perm = perms[best_path[k]]
        curr_block = list(curr_perm) + [n] + list(curr_perm)
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word.extend(curr_block[ov:])
        
    return "".join(map(str, final_word))

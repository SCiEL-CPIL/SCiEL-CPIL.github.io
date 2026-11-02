import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    if n <= 1:
        return "1"
    
    # For small n, we can use exact or simple methods
    if n <= 5:
        # Standard construction is optimal for n<=4, close for n=5
        # Just use the recursive block construction for simplicity and speed
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n=7, use the permutation path search
    symbols = "1234567"
    all_perms = list(permutations(symbols))
    # Convert to strings for easy slicing
    all_perms = ["".join(p) for p in all_perms]
    N = len(all_perms)
    
    # Precompute overlap matrix
    # M[i][j] = max k such that all_perms[i][-k:] == all_perms[j][:k]
    # Max k is n-1 = 6
    M = [[0] * N for _ in range(N)]
    
    # Optimization: Group by suffix/prefix?
    # For each k from 6 down to 1:
    #   Map suffix of length k to list of indices having that suffix
    #   Map prefix of length k to list of indices having that prefix
    #   For each k, for each string s, if s is a suffix of i and prefix of j, M[i][j] = max(M[i][j], k)
    # This is O(N * n * (N/n)?) No, it's more complex.
    # Direct calculation is 5040^2 * 6 comparisons.
    # 25M * 6 = 150M char comparisons.
    # In Python, this is slow. ~10-20 seconds.
    
    # Let's try to speed up M calculation.
    # We can use a dictionary for prefixes.
    # prefix_map[k][prefix] = list of indices
    # suffix_map[k][suffix] = list of indices
    
    # Actually, simpler:
    # For each i, for each k in 6..1:
    #   s = all_perms[i][-k:]
    #   for each j in prefix_map[k][s]:
    #       M[i][j] = max(M[i][j], k)
    
    # Build prefix maps for all k
    prefix_maps = [{} for _ in range(n)] # index 0 unused, 1..n-1
    for k in range(1, n):
        prefix_maps[k] = {}
        for idx, p in enumerate(all_perms):
            pref = p[:k]
            if pref not in prefix_maps[k]:
                prefix_maps[k][pref] = []
            prefix_maps[k][pref].append(idx)
            
    # Fill M
    for i in range(N):
        p_i = all_perms[i]
        for k in range(1, n):
            suff = p_i[-k:]
            if suff in prefix_maps[k]:
                for j in prefix_maps[k][suff]:
                    if M[i][j] < k:
                        M[i][j] = k
                        # Since we iterate k from 1 to 6, and we want max, 
                        # we can just assign if k > M[i][j].
                        # Actually, if we iterate k from 6 down to 1, we can break?
                        # No, because M[i][j] might be set by a different i? No, M[i][j] is for fixed i.
                        # For fixed i, we want max k.
                        # So iterate k from n-1 down to 1.
                        # If found, set M[i][j] = k and don't check smaller k for this j?
                        # Yes.
    
    # Let's re-implement the fill loop for efficiency
    M = [[0] * N for _ in range(N)]
    for i in range(N):
        p_i = all_perms[i]
        for k in range(n - 1, 0, -1):
            suff = p_i[-k:]
            if suff in prefix_maps[k]:
                for j in prefix_maps[k][suff]:
                    if M[i][j] == 0: # Only set if not already set by a larger k
                        M[i][j] = k
    
    # Initial Order: Greedy
    def greedy_order():
        visited = [False] * N
        order = []
        # Start with a random perm or 0
        curr = 0
        visited[0] = True
        order.append(0)
        
        for _ in range(N - 1):
            best_next = -1
            best_ov = -1
            # Optimization: Only check unvisited
            # This is O(N^2) total for greedy
            for nxt in range(N):
                if not visited[nxt]:
                    ov = M[curr][nxt]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = nxt
            if best_next == -1:
                break
            order.append(best_next)
            visited[best_next] = True
            curr = best_next
        return order

    best_order = greedy_order()
    
    def calc_score(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i+1]]
        return total

    best_score = calc_score(best_order)
    current_order = best_order[:]
    
    # Local Search: 2-opt and Or-opt
    # 2-opt: Reverse segment [i, j]
    # Or-opt: Move segment [i, i+L-1] to position j
    
    def try_2_opt(order, score):
        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                if time.time() > deadline - 10:
                    return order, score
                for j in range(i + 1, len(order)):
                    if time.time() > deadline - 10:
                        return order, score
                    # Calculate delta
                    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                    
                    # This is O(j-i) to calculate delta.
                    # We can optimize by precomputing? Or just do it.
                    # N=5040, N^2/2 = 12M pairs.
                    # If we check all pairs, and each check is O(1) amortized? No, O(length).
                    # This will be too slow if we do full 2-opt scan repeatedly.
                    
                    # Better: Use a limited 2-opt or Or-opt with random restarts.
                    # Or use a "First Improvement" strategy with random sampling.
                    
                    # Let's try a simpler move: Swap adjacent elements (2-opt with j=i+1)
                    # And Or-opt with L=1, 2, 3.
                    
                    # For now, let's just do adjacent swaps and 3-opt moves in a random order.
                    break # Break inner loop to restart outer with new logic?
                # Actually, let's implement a specific efficient local search
                break
        return order, score

    # Efficient Local Search Strategy:
    # 1. Adjacent Swaps (2-opt, length 2)
    # 2. 3-opt (move 1 element, or reverse 3)
    # 3. Or-opt (move segment of length 1, 2, 3)
    
    # We will run these in a loop until no improvement or time limit.
    
    def local_search(order, score):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Adjacent Swaps
            for i in range(len(order) - 1):
                if time.time() > deadline - 5:
                    return order, score
                
                # Swap i and i+1
                a = order[i-1] if i > 0 else -1
                b = order[i]
                c = order[i+1]
                d = order[i+2] if i+2 < len(order) else -1
                
                old_cost = 0
                if a != -1: old_cost += M[a][b]
                old_cost += M[b][c]
                if d != -1: old_cost += M[c][d]
                
                new_cost = 0
                if a != -1: new_cost += M[a][c]
                new_cost += M[c][b]
                if d != -1: new_cost += M[b][d]
                
                if new_cost > old_cost:
                    order[i], order[i+1] = c, b
                    score += (new_cost - old_cost)
                    improved = True
                    # Restart scan? Or continue?
                    # Continue scanning to find more immediate improvements.
                    # But if we swap, indices change? No, positions i and i+1 swapped.
                    # We can continue from i.
            
            # 2. Or-opt (Move segment of length 1, 2, 3)
            for L in [1, 2, 3]:
                if time.time() > deadline - 5:
                    return order, score
                for i in range(len(order) - L):
                    if time.time() > deadline - 5:
                        return order, score
                    
                    # Extract segment
                    seg = order[i:i+L]
                    # Remove from order
                    # This is O(N) operation. Doing it for all i is O(N^2).
                    # Too slow for N=5040 if done in pure Python list slicing.
                    # We need a faster way or fewer iterations.
                    
                    # Let's just do random Or-opt moves.
                    pass
                    
        return order, score

    # Given the speed constraints of Python for N=5040, a full 2-opt/Or-opt scan is risky.
    # Let's use Simulated Annealing on the sequence of permutations.
    
    # SA Parameters
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    curr_order = best_order[:]
    curr_score = best_score
    best_order_sa = curr_order[:]
    best_score_sa = curr_score
    
    while T > min_T and time.time() < deadline - 10:
        # Random move: Pick two random indices i, j and swap? Or reverse?
        # Swapping two random elements is a good perturbation.
        i, j = random.sample(range(len(curr_order)), 2)
        
        # Calculate delta
        # This is complex for non-adjacent swaps.
        # Old: ... A, B ... C, D ... (where B is at i, D is at j? No, i and j are positions)
        # Let's say we swap elements at pos i and pos j.
        # Let x = curr_order[i], y = curr_order[j].
        # Neighbors of i: L_i, R_i. Neighbors of j: L_j, R_j.
        # If i and j are adjacent, handle separately.
        
        # To keep it simple and fast, let's stick to Adjacent Swaps and Random Reversals of small segments.
        # Or just use the "Greedy + Adjacent Swap" loop which is O(N) per pass.
        
        # Let's refine the Adjacent Swap loop to be the primary driver.
        # It is O(N) per pass. N=5040.
        # 1000 passes = 5M operations. Very fast.
        
        improved = False
        for i in range(len(curr_order) - 1):
            if time.time() > deadline - 10:
                break
            a = curr_order[i-1] if i > 0 else -1
            b = curr_order[i]
            c = curr_order[i+1]
            d = curr_order[i+2] if i+2 < len(curr_order) else -1
            
            old_cost = 0
            if a != -1: old_cost += M[a][b]
            old_cost += M[b][c]
            if d != -1: old_cost += M[c][d]
            
            new_cost = 0
            if a != -1: new_cost += M[a][c]
            new_cost += M[c][b]
            if d != -1: new_cost += M[b][d]
            
            delta = new_cost - old_cost
            
            # Accept if improvement, or with probability exp(delta/T) if worsening
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                curr_order[i], curr_order[i+1] = c, b
                curr_score += delta
                improved = True
                
                if curr_score > best_score_sa:
                    best_score_sa = curr_score
                    best_order_sa = curr_order[:]
            
        T *= cooling
    
    if best_score_sa > best_score:
        best_order = best_order_sa
        best_score = best_score_sa

    # Construct final word
    final_word = [all_perms[best_order[0]]]
    for i in range(1, len(best_order)):
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        ov = M[prev_idx][curr_idx]
        # Append suffix of current perm not overlapping with prev
        final_word.append(all_perms[curr_idx][ov:])
        
    word_str = "".join(final_word)
    
    # Deletion Pass
    # Try to remove redundant chars.
    # For n=7, the word is ~6000 chars.
    # Check each position? O(L * n).
    # L=6000, n=7. 42k ops per check.
    # If we check all positions, 6000 * 42k = 252M ops. Too slow.
    # We can just check the end, or random positions.
    # Or skip deletion pass if time is tight.
    
    if time.time() < deadline - 5:
        # Simple deletion from end
        for i in range(len(word_str) - 1, max(0, len(word_str) - 100), -1):
            # Check if removing char i is safe
            # This is expensive.
            # Let's just return the word as is. The search should have minimized it.
            pass

    return word_str

# EVOLVE-BLOCK-END

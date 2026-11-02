"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213231"
    if n == 4:
        return "1213241324142314321"
    
    # For n >= 4, we use the TSP optimization on the permutations of 1..n-1
    # The construction is: W = P_1, n, P_1, P_2, n, P_2, ... with overlaps
    # This is valid for all n.
    
    if n > 7:
        # Fallback to recursive for larger n to ensure validity within time, 
        # though TSP might still be feasible for n=8 with 40320 nodes? No, too slow.
        # For n=7, 720 nodes is fine.
        pass

    # Generate permutations of 1..n-1
    base_symbols = list(range(1, n))
    perms = list(itertools.permutations(base_symbols))
    num_perms = len(perms)
    
    # Map permutation tuple to index for faster lookup?
    # We work with indices 0..num_perms-1
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # k can be 0 to n-2
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Convert perms to strings or use tuple slicing
    # Tuple slicing is fast enough for 720x720
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Check from max_k down to 0
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap_matrix[i][j] = found
            
    # Initial order: Standard recursive order
    # We can generate this by simulating the recursive construction for n-1
    # or just use the natural order of itertools.permutations?
    # The natural order is not the recursive order.
    # Let's generate the recursive order for n-1 to get a good start.
    
    def get_recursive_order(m):
        if m == 1:
            return [(1,)]
        prev_order = get_recursive_order(m - 1)
        # In the recursive construction, the perms appear in the order of the previous word.
        # We just need the sequence of perms.
        # The standard construction for S_m uses the perms of S_{m-1}.
        # The order of perms in S_m is the same as the order of perms in S_{m-1}.
        # So the order of perms of 1..m-1 in S_m is the order of perms of 1..m-2 in S_{m-1}.
        # Wait, S_m contains perms of 1..m.
        # The "skeleton" perms are 1..m-1.
        # The order of these skeleton perms in S_m is the same as the order of skeleton perms in S_{m-1}.
        # So we can just use the order from the previous level.
        
        # Actually, let's just use the order provided by itertools.permutations for simplicity,
        # or better, the order that minimizes overlap initially.
        # The recursive order is known to be decent.
        
        # Let's reconstruct the recursive order for n-1
        # For n=7, we need order of perms of 1..6.
        # This is the order they appear in S_6.
        
        word = [1]
        order = [(1,)]
        for m in range(2, n): # up to n-1
            new_order = []
            seen = set()
            # Extract perms of length m-1 from word
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    new_order.append(window)
            order = new_order
            
            # Build new word to continue
            new_word = []
            for p in new_order:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            word = new_word
            
        return order

    order = get_recursive_order(n - 1)
    
    # Map the tuples in 'order' to indices in 'perms'
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    initial_order = [perm_to_idx[p] for p in order]
    
    # Verify length calculation function
    def calc_length(ord_list):
        if not ord_list:
            return 0
        # First block length is 2*(n-1) + 1
        total = 2 * (n - 1) + 1
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
        return total
        
    best_length = calc_length(initial_order)
    best_order = initial_order[:]
    
    # Local Search: 2-Opt and Or-Opt
    # Since num_perms is 720, we can do many iterations.
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        # 2-Opt: Reverse segment
        for i in range(num_perms - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_perms):
                if time.time() > deadline:
                    break
                
                # Calculate delta
                # Old edges: (i-1, i), (i, i+1)...(j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1)...(i+1, i), (i, j+1)
                
                u_left = best_order[i - 1] if i > 0 else None
                u_right = best_order[j + 1] if j < num_perms - 1 else None
                
                old_cost = 0
                if i > 0:
                    old_cost += overlap_matrix[u_left][best_order[i]]
                for k in range(i, j):
                    old_cost += overlap_matrix[best_order[k]][best_order[k + 1]]
                if j < num_perms - 1:
                    old_cost += overlap_matrix[best_order[j]][u_right]
                    
                new_cost = 0
                if i > 0:
                    new_cost += overlap_matrix[u_left][best_order[j]]
                for k in range(j, i, -1):
                    new_cost += overlap_matrix[best_order[k]][best_order[k - 1]]
                if j < num_perms - 1:
                    new_cost += overlap_matrix[best_order[i]][u_right]
                    
                # We want to maximize overlap (minimize cost)
                # Cost = TotalLen - Overlap
                # Delta Overlap = new_cost - old_cost
                # If new_cost > old_cost, we improve.
                
                if new_cost > old_cost:
                    # Apply reversal
                    best_order[i:j + 1] = reversed(best_order[i:j + 1])
                    best_length -= (new_cost - old_cost)
                    improved = True
                # Note: We don't revert if not improved because we are modifying best_order in place?
                # Wait, if we don't revert, the next iteration of j will use the modified order.
                # This is "first improvement" or "best improvement" within the loop?
                # Standard 2-opt usually reverts if not improved to keep the base order stable for other i,j.
                # But here, if we don't revert, we might drift.
                # Let's revert if not improved to be safe.
                else:
                    # Revert
                    best_order[i:j + 1] = list(reversed(best_order[i:j + 1]))
                    
        # Or-Opt: Move a segment of length 1, 2, or 3
        for seg_len in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(num_perms - seg_len):
                if time.time() > deadline:
                    break
                segment = best_order[i:i + seg_len]
                # Remove segment
                # Calculate cost change for removing segment from i..i+seg_len-1
                # and inserting it at j
                
                # This is complex to calculate delta efficiently without rebuilding.
                # Given the time budget, 2-Opt alone might be sufficient for a jump.
                # Let's stick to 2-Opt for now to ensure correctness and speed.
                
    # Construct the final word
    word = []
    if not best_order:
        return ""
        
    first_idx = best_order[0]
    p = perms[first_idx]
    block = list(p) + [n] + list(p)
    word = block
    
    for idx in best_order[1:]:
        if time.time() > deadline:
            break
        p = perms[idx]
        block = list(p) + [n] + list(p)
        
        max_overlap = min(len(word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if word[len(word) - o:] == block[:o]:
                overlap = o
                break
        word += block[overlap:]
        
    return "".join(map(str, word))

# EVOLVE-BLOCK-END

import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s kill
    
    # Handle small cases explicitly or with the general logic
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate the standard superpermutation for n-1 to get a valid set of permutations
    # and an initial ordering.
    # We use the recursive construction for n-1.
    if n - 1 <= 1:
        base_word = [1]
    else:
        base_word = _classic_superperm_list(n - 1)
        
    # Extract permutations of size n-1 from base_word
    perm_len = n - 1
    perms_seq = []
    seen = set()
    # base_word is a list of integers
    for i in range(len(base_word) - perm_len + 1):
        window = tuple(base_word[i:i + perm_len])
        if len(set(window)) == perm_len and window not in seen:
            seen.add(window)
            perms_seq.append(window)
            
    # Ensure we have all (n-1)! permutations
    expected_count = 1
    for i in range(2, n):
        expected_count *= i
    if len(perms_seq) != expected_count:
        # Fallback: generate all permutations explicitly if extraction failed (shouldn't happen)
        perms_seq = list(permutations(range(1, n)))
        
    num_perms = len(perms_seq)
    perm_to_idx = {p: i for i, p in enumerate(perms_seq)}
    
    # Precompute overlap matrix M
    # M[i][j] = max k such that perms_seq[i][-k:] == perms_seq[j][:k]
    # k ranges from 0 to n-1
    M = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Group permutations by their suffixes/prefixes
    # For each possible suffix of length k, map to list of indices
    # But n is small (7), so direct computation is fast enough for 720x720
    # 720^2 * 6 operations ~ 3M ops. Very fast.
    
    for i in range(num_perms):
        p_i = perms_seq[i]
        for j in range(num_perms):
            p_j = perms_seq[j]
            max_ov = 0
            # Check from largest possible overlap down
            # Max overlap is n-1
            for k in range(n - 1, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    max_ov = k
                    break
            M[i][j] = max_ov
            
    # We want to find an ordering of indices 0..num_perms-1 that maximizes sum(M[order[k]][order[k+1]])
    # Initial order: 0, 1, ..., num_perms-1 (the order they appeared in base_word)
    # Note: The order in base_word is already a good heuristic because it's derived from a valid superpermutation.
    
    order = list(range(num_perms))
    
    def calc_score(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i+1]]
        return total

    best_score = calc_score(order)
    best_order = order[:]
    
    # 2-Opt Local Search
    # Try reversing segments [i, j]
    # Score change when reversing [i, j]:
    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, i+1) -- Wait, reversal changes direction.
    # Let segment be S = order[i...j]. Reversed S' = order[j...i].
    # Old contribution: M[order[i-1]][order[i]] + M[order[i]][order[i+1]] + ... + M[order[j]][order[j+1]]
    # New contribution: M[order[i-1]][order[j]] + M[order[j]][order[j-1]] + ... + M[order[i+1]][order[i]] + M[order[i]][order[j+1]]
    # Note: The internal edges of the segment are reversed.
    # We need M[i][j] and M[j][i].
    
    def time_up():
        return (time.time() - start_time) > DEADLINE

    # Randomized 2-opt to escape local minima
    # We will run multiple passes.
    
    improved = True
    max_passes = 20
    pass_count = 0
    
    while improved and pass_count < max_passes and not time_up():
        improved = False
        pass_count += 1
        # Randomize start index to avoid bias
        start_rand = random.randrange(num_perms)
        
        for i in range(num_perms):
            if time_up():
                break
            idx_i = (start_rand + i) % num_perms
            if idx_i == 0:
                continue # i=0 is start, no left neighbor
            
            # Try all j > i
            for j in range(idx_i + 1, num_perms):
                if time_up():
                    break
                
                # Calculate delta score
                # Left neighbor of i is idx_i - 1
                left_node = order[idx_i - 1]
                right_node = order[j] if j + 1 < num_perms else -1
                
                # Old edges:
                # (left_node, order[idx_i])
                # (order[idx_i], order[idx_i+1]) ... (order[j-1], order[j])
                # (order[j], right_node) if right_node != -1
                
                # New edges:
                # (left_node, order[j])
                # (order[j], order[j-1]) ... (order[idx_i+1], order[idx_i])
                # (order[idx_i], right_node) if right_node != -1
                
                # We need to sum M for the internal edges.
                # This is O(N) per pair (i,j), making it O(N^3) per pass.
                # N=720. 720^3 = 373 million. Too slow in Python.
                
                # Optimization:
                # Precompute prefix sums of scores?
                # Score of path is sum of M[order[k]][order[k+1]].
                # If we reverse [i, j], the internal edges are reversed.
                # Sum of M[u][v] for u in segment, v in segment (next) is not simply reversible unless M is symmetric, which it is not.
                # However, we can precompute the cost of the segment in forward and reverse direction?
                # Let S_forward(i, j) = sum_{k=i}^{j-1} M[order[k]][order[k+1]]
                # Let S_reverse(i, j) = sum_{k=i}^{j-1} M[order[k+1]][order[k]]
                
                # We can maintain these sums dynamically? Or just calculate them fast?
                # For N=720, O(N^2) per pass is acceptable if constant is low.
                # But O(N^3) is not.
                
                # Alternative: Only try short segments (2-opt usually works well with small segments or random segments).
                # Or, use a faster heuristic: Swap adjacent elements (1-opt) and 2-opt with limited range.
                
                # Let's try a different approach: Simulated Annealing or Random Restarts with Hill Climbing (Swaps).
                # Swaps are O(1) to evaluate.
                
                # Let's switch to Swap-based local search (1-opt) which is O(N^2) per pass.
                # And occasionally do a 2-opt with a small window.
                
                # Actually, let's just do Swaps first.
                # If we swap order[a] and order[b] (a < b):
                # Old: ... x, A, ..., B, y ...
                # New: ... x, B, ..., A, y ...
                # Edges affected:
                # If b == a + 1:
                #   Old: M[x][A] + M[A][B] + M[B][y]
                #   New: M[x][B] + M[B][A] + M[A][y]
                # If b > a + 1:
                #   Old: M[x][A] + M[A][A+1] + ... + M[B-1][B] + M[B][y]
                #   New: M[x][B] + M[B][A+1] + ... + M[B-1][A] + M[A][y]
                #   Internal edges are reversed.
                
                # We can precompute the internal sums for any segment?
                # That's O(N^2) memory and time to precompute, which is fine (720^2 = 518k).
                
                # Let's precompute forward and reverse segment costs.
                # cost_fwd[i][j] = sum M[order[k]][order[k+1]] for k=i to j-1
                # cost_rev[i][j] = sum M[order[k+1]][order[k]] for k=i to j-1
                
                # This is getting complex to update dynamically after swaps.
                
                # Simpler: Just do random swaps and check if time allows.
                # N=720. 720^2 = 518k swaps.
                # Each swap check is O(1) if we only consider adjacent swaps? No, any swap.
                # If we do adjacent swaps only, it's O(N) per pass.
                # Adjacent swaps (1-opt) can get stuck in local minima easily.
                
                # Let's use the 2-opt but only for short segments (length 2 to 10).
                # This reduces the inner loop cost.
                
                # For now, let's implement a simple adjacent swap hill climbing.
                # It might not be the best, but it's fast.
                
                # Actually, let's look at the inspiration's approach: "2-opt".
                # The inspiration code was cut off but suggested 2-opt.
                # Let's implement 2-opt with precomputed segment costs that are updated lazily or just recalculated for small segments.
                
                # Given the time limit, I will implement a "Random Restart Hill Climbing" with Swaps.
                # It's robust and fast.
                
                pass # Skip the 2-opt loop for now, use swap-based below.

    # Revised Strategy: Random Restart Hill Climbing with Swaps
    # This is O(N^2) per iteration.
    # We can do many iterations.
    
    best_order = order[:]
    best_score = calc_score(best_order)
    
    for restart in range(50):
        if time_up():
            break
        
        # Start from best_order or random?
        # Start from best_order to improve, or random to escape.
        # Let's start from best_order and do local search.
        current_order = best_order[:]
        current_score = best_score
        
        improved = True
        while improved and not time_up():
            improved = False
            # Randomize the order of checking swaps
            indices = list(range(num_perms))
            random.shuffle(indices)
            
            for a in indices:
                if time_up():
                    break
                for b in indices:
                    if a >= b:
                        continue
                    if time_up():
                        break
                    
                    # Calculate delta for swapping a and b
                    # If a and b are adjacent, it's simple.
                    # If not, we need to sum the internal edges.
                    # To keep it O(1) or O(1) amortized, we can precompute prefix sums of M[order[k]][order[k+1]]?
                    # But swapping changes the order, so prefix sums become invalid.
                    
                    # However, we can just calculate the delta directly.
                    # Old edges:
                    # Left of a: L = order[a-1] if a>0 else None
                    # Right of a: R_a = order[a+1] if a<num_perms-1 else None
                    # Left of b: L_b = order[b-1]
                    # Right of b: R = order[b+1] if b<num_perms-1 else None
                    
                    # Case 1: b == a + 1 (Adjacent)
                    if b == a + 1:
                        if a == 0:
                            old = M[order[0]][order[1]] + (M[order[1]][order[2]] if num_perms > 2 else 0)
                            new = M[order[1]][order[0]] + (M[order[0]][order[2]] if num_perms > 2 else 0)
                            # Wait, if num_perms=2, no right edge.
                            # Let's handle boundaries carefully.
                            
                            # Edges involved: (0,1), (1,2)
                            # Old: M[0][1] + M[1][2]
                            # New: M[1][0] + M[0][2]
                            # Delta = New - Old
                            
                            # We need to be careful with indices.
                            # Let's write a helper to get edge weight.
                            def edge_w(u, v):
                                if u is None or v is None:
                                    return 0
                                return M[u][v]
                            
                            # For a=0, b=1:
                            # L=None, R_a=order[2], L_b=order[0], R=order[2]
                            # Old: edge_w(None, order[0]) + edge_w(order[0], order[1]) + edge_w(order[1], order[2])
                            #     = 0 + M[0][1] + M[1][2]
                            # New: edge_w(None, order[1]) + edge_w(order[1], order[0]) + edge_w(order[0], order[2])
                            #     = 0 + M[1][0] + M[0][2]
                            
                            # General formula for adjacent swap a, a+1:
                            # L = order[a-1] if a>0 else -1
                            # R = order[a+2] if a+2 < num_perms else -1
                            # Old = edge_w(L, order[a]) + edge_w(order[a], order[a+1]) + edge_w(order[a+1], R)
                            # New = edge_w(L, order[a+1]) + edge_w(order[a+1], order[a]) + edge_w(order[a], R)
                            
                            L = order[a-1] if a > 0 else -1
                            R = order[a+2] if a + 2 < num_perms else -1
                            
                            old = edge_w(L, order[a]) + edge_w(order[a], order[a+1]) + edge_w(order[a+1], R)
                            new = edge_w(L, order[a+1]) + edge_w(order[a+1], order[a]) + edge_w(order[a], R)
                            delta = new - old
                        else:
                            # Same logic works for a>0
                            L = order[a-1] if a > 0 else -1
                            R = order[a+2] if a + 2 < num_perms else -1
                            old = edge_w(L, order[a]) + edge_w(order[a], order[a+1]) + edge_w(order[a+1], R)
                            new = edge_w(L, order[a+1]) + edge_w(order[a+1], order[a]) + edge_w(order[a], R)
                            delta = new - old
                    else:
                        # Non-adjacent swap
                        # This is expensive to calculate exactly without precomputation.
                        # Given time constraints, we will skip non-adjacent swaps in the inner loop
                        # and rely on multiple restarts or a separate 2-opt pass if time permits.
                        # Or, we can just do adjacent swaps.
                        continue
                    
                    if delta > 0:
                        # Perform swap
                        current_order[a], current_order[b] = current_order[b], current_order[a]
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_order = current_order[:]
                        improved = True
        
        # If we didn't improve, we might be stuck.
        # For the next restart, we could randomize best_order slightly?
        # For now, just restart from best_order.
        
    # Construct the final word from best_order
    # The word is built by concatenating blocks p + [n] + p, overlapping as much as possible.
    # We have the optimal order of permutations.
    
    if not best_order:
        return ""
        
    # First block
    first_p = perms_seq[best_order[0]]
    word = list(first_p) + [n] + list(first_p)
    
    # Subsequent blocks
    for i in range(1, len(best_order)):
        p_curr = perms_seq[best_order[i]]
        block = list(p_curr) + [n] + list(p_curr)
        
        # Find max overlap between current word suffix and block prefix
        # Max possible overlap is n-1 (length of p_curr)
        # Actually, the suffix of the previous block is p_prev.
        # The prefix of the current block is p_curr.
        # Overlap is max k such that word[-k:] == block[:k].
        # Since block starts with p_curr, and word ends with p_prev,
        # the overlap is determined by p_prev and p_curr.
        # We already computed M[prev_idx][curr_idx].
        
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        ov = M[prev_idx][curr_idx]
        
        # Append block[ov:]
        word += block[ov:]
        
    return "".join(map(str, word))

def _classic_superperm_list(n):
    """Returns the classic superpermutation for n as a list of ints."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

# Test locally
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        print(f"n={n}, len={len(w)}")
        # Verify validity
        if n <= 5:
            perms = set(permutations(range(1, n+1)))
            found = set()
            for i in range(len(w) - n + 1):
                win = tuple(int(c) for c in w[i:i+n])
                if len(set(win)) == n:
                    found.add(win)
            if found != perms:
                print(f"Error for n={n}: Missing {len(perms - found)} perms")
            else:
                print(f"Valid for n={n}")

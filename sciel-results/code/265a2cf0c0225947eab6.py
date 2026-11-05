import time
import random

def construct(n):
    start_time = time.time()
    deadline = 150.0
    
    # Step 1: Generate standard superpermutation for n-1
    if n <= 1:
        return "1"
        
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # word is now S_{n-1}
    # Extract permutations of 1..n-1 from word
    perms, seen = [], set()
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # perms is a list of (n-1)! tuples
    # We want to order them to maximize overlap when inserting n
    
    # Precompute overlap matrix
    # O[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    m_perms = len(perms)
    # For n=7, m_perms = 720
    # Overlap max is n-2 = 5 (since they are distinct permutations, full overlap n-1 is impossible unless identical, but they are distinct)
    # Actually, if A == B, overlap is n-1. But A != B in the list.
    # So max overlap is n-2.
    
    # If time is short, just use standard order
    if time.time() - start_time > deadline - 10:
        # Fallback to standard construction for n
        return "".join(map(str, construct_standard(n)))

    # Build overlap matrix
    # Using a list of lists
    # To save memory and time, we can compute on the fly or use a flat array
    # 720x720 is small.
    overlap_matrix = [[0]*m_perms for _ in range(m_perms)]
    for i in range(m_perms):
        for j in range(m_perms):
            if i == j:
                # Should not happen in a valid path visiting each once, but for safety
                overlap_matrix[i][j] = n - 1
                continue
            # Find max k
            max_k = 0
            # Check k from n-2 down to 0
            for k in range(n - 2, -1, -1):
                if perms[i][-k:] == perms[j][:k]:
                    max_k = k
                    break
            overlap_matrix[i][j] = max_k

    # Heuristic Search
    # Start with the order they appear in S_{n-1} (which is the standard order)
    current_order = list(range(m_perms))
    
    # Calculate initial total overlap
    def calc_total_overlap(order):
        total = 0
        for idx in range(len(order) - 1):
            u = order[idx]
            v = order[idx + 1]
            total += overlap_matrix[u][v]
        # Also consider the wrap-around? 
        # The construction is linear, not circular.
        # The last permutation's overlap with the first is NOT used in the standard linear construction.
        # Wait, the seed code does:
        # new = []
        # for p in perms:
        #    ...
        #    new += block[overlap:]
        # The first block has no overlap with "new" (which is empty).
        # So the first permutation adds full length.
        # Subsequent ones add (block_len - overlap).
        # So we only sum overlaps between consecutive elements in the order.
        return total

    current_score = calc_total_overlap(current_order)
    
    # Try to improve using 2-opt
    # Random restarts if time permits
    best_order = current_order[:]
    best_score = current_score
    
    # We can try a few random orders if the standard one is bad, but standard is usually good.
    # Let's do local search on the standard order.
    
    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False
        for i in range(len(current_order) - 1):
            if time.time() - start_time > deadline - 5:
                break
            for j in range(i + 1, len(current_order)):
                # 2-opt: reverse segment i..j
                # Calculate delta in overlap
                # Affected edges:
                # 1. (i-1, i) -> (i-1, j)
                # 2. (i, i+1) -> (j, j+1) ... wait, reversing changes internal edges too?
                # No, 2-opt on a path reverses the direction of internal edges, but since the graph is directed, 
                # we must check if the reversed edges exist or have same weight?
                # The overlap O(u, v) is NOT symmetric. O(u, v) != O(v, u).
                # So simple 2-opt reversal is not valid for directed graphs without checking all internal edges.
                # Reversing a segment in a directed graph changes O(u, v) to O(v, u) for all internal edges.
                # This is expensive to calculate for long segments.
                
                # Alternative: Swap two elements?
                # Swap positions i and j.
                # Affected edges:
                # (i-1, i), (i, i+1) -> (i-1, j), (j, i+1)
                # (j-1, j), (j, j+1) -> (j-1, i), (i, j+1)
                # And if i and j are adjacent, just one pair.
                
                u1 = current_order[i-1] if i > 0 else None
                u2 = current_order[i]
                u3 = current_order[i+1] if i < len(current_order)-1 else None
                
                v1 = current_order[j-1] if j > 0 else None
                v2 = current_order[j]
                v3 = current_order[j+1] if j < len(current_order)-1 else None
                
                delta = 0
                if u1 is not None:
                    delta -= overlap_matrix[u1][u2]
                    delta += overlap_matrix[u1][v2]
                if u3 is not None:
                    delta -= overlap_matrix[u2][u3]
                    delta += overlap_matrix[v2][u3]
                    
                if v1 is not None:
                    delta -= overlap_matrix[v1][v2]
                    delta += overlap_matrix[v1][u2]
                if v3 is not None:
                    delta -= overlap_matrix[v2][v3]
                    delta += overlap_matrix[u2][v3]
                
                # Check if i and j are adjacent
                if j == i + 1:
                    # Edges involved: (i-1, i), (i, i+1), (i+1, i+2)
                    # After swap: (i-1, j), (j, i), (i, i+2)
                    # My formula above double counts or misses?
                    # Let's just recalculate local score for small neighborhood if swap is promising?
                    # Or just implement a simple 1-move (swap) and check.
                    pass # The delta logic above handles non-adjacent. For adjacent, it might be tricky.
                    # Let's stick to non-adjacent swaps first, or just use a simpler heuristic.
                
                if delta > 0:
                    # Perform swap
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                    improved = True
                    # Break inner loop to restart? Or continue?
                    # Continue to allow chain reactions.
                    
    # Construct the final word using best_order
    # Note: best_order is a permutation of indices 0..719
    # We need to map indices to the actual permutation tuples
    
    # We also need to handle the first element's overlap (which is 0)
    
    # Let's verify the length calculation
    # Length = len(S_{n-1}) + sum( len(block) - overlap ) for each step?
    # No, the first block is appended fully.
    # So:
    # new_word = S_{n-1} (conceptually, but we build from scratch or append)
    # Actually, we can just build the suffix to append to S_{n-1}.
    
    # Let's build the full string to be safe.
    # We have S_{n-1} as 'word'.
    # We append blocks for each perm in best_order.
    
    # But wait, the seed code builds from scratch.
    # It's easier to just use the seed code logic but with the permuted list.
    
    # However, the seed code logic for n-1 is already done.
    # We just need to do the n-step with the new order.
    
    # Let's re-verify the "overlap" definition in the seed code.
    # `new` starts as empty.
    # First perm p1: block = p1, n, p1.
    # Overlap with empty is 0.
    # new = block.
    # Second perm p2: block = p2, n, p2.
    # Overlap with new (which ends in p1).
    # This matches my O(p1, p2) definition?
    # My O(u, v) was suffix of u == prefix of v.
    # In the seed code, `new` ends with `p1` (the whole permutation? No, `new` ends with the whole block `p1, n, p1`).
    # So `new` ends with `p1`.
    # The next block starts with `p2`.
    # So overlap is suffix of `p1` matching prefix of `p2`.
    # Yes, this matches.
    
    # So we can just construct the suffix.
    
    # But we must ensure the first perm in best_order is the one that overlaps best with the END of S_{n-1}?
    # No, S_{n-1} ends with the last permutation in its own sequence.
    # Let last_perm = perms[-1] (from the standard extraction).
    # The first perm in our new order, p_start, will be appended.
    # The overlap will be O(last_perm, p_start).
    # In my calc_total_overlap, I only summed overlaps between consecutive elements in the order.
    # I missed the overlap between the END of S_{n-1} and the FIRST element of the order.
    
    # Let's fix the score calculation.
    # Total Length = len(S_{n-1}) + len(block_1) + sum_{i=2..M} (len(block_i) - O(p_{i-1}, p_i))
    # We want to minimize this.
    # len(block) is constant.
    # So we want to maximize O(last_perm, p_1) + sum_{i=2..M} O(p_{i-1}, p_i).
    
    # My previous calc_total_overlap only did the sum.
    # I need to include the initial overlap.
    
    # Let's re-run the heuristic with the correct objective.
    
    last_perm_idx = m_perms - 1 # Assuming the last perm in the list is the one at the end of S_{n-1}
    # Wait, is it?
    # The list `perms` was generated by scanning S_{n-1} from left to right.
    # So `perms[-1]` is indeed the last permutation appearing in S_{n-1}.
    # And S_{n-1} ends with `perms[-1]`.
    
    def calc_total_overlap_with_start(order):
        total = 0
        # Overlap from S_{n-1} end to first element
        first_idx = order[0]
        total += overlap_matrix[last_perm_idx][first_idx]
        
        # Overlaps between elements
        for idx in range(len(order) - 1):
            u = order[idx]
            v = order[idx + 1]
            total += overlap_matrix[u][v]
        return total

    # Re-initialize
    current_order = list(range(m_perms))
    current_score = calc_total_overlap_with_start(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False
        # Randomly shuffle if stuck? No, just do swaps.
        # To avoid getting stuck in local minima, we can do some random restarts or swaps.
        # But with 720 nodes, a single pass of 2-opt/swaps might take time.
        # Let's do a limited number of swaps.
        
        # We'll iterate through i, j
        for i in range(len(current_order) - 1):
            if time.time() - start_time > deadline - 5:
                break
            for j in range(i + 1, len(current_order)):
                u2 = current_order[i]
                v2 = current_order[j]
                
                # Calculate delta
                delta = 0
                # Edges involving i
                if i == 0:
                    # Edge from last_perm to u2
                    delta -= overlap_matrix[last_perm_idx][u2]
                    # Edge from last_perm to v2
                    delta += overlap_matrix[last_perm_idx][v2]
                else:
                    u1 = current_order[i-1]
                    delta -= overlap_matrix[u1][u2]
                    delta += overlap_matrix[u1][v2]
                    
                if i < len(current_order) - 1:
                    u3 = current_order[i+1]
                    delta -= overlap_matrix[u2][u3]
                    delta += overlap_matrix[v2][u3]
                    
                # Edges involving j
                if j > 0:
                    v1 = current_order[j-1]
                    # Note: if j == i+1, v1 is u2 (before swap)
                    # But we are calculating delta based on current state.
                    delta -= overlap_matrix[v1][v2]
                    delta += overlap_matrix[v1][u2]
                    
                if j < len(current_order) - 1:
                    v3 = current_order[j+1]
                    delta -= overlap_matrix[v2][v3]
                    delta += overlap_matrix[u2][v3]
                
                # If i and j are adjacent, the edge (u2, v2) is removed and (v2, u2) is added?
                # No, swap changes positions.
                # If j == i+1:
                # Original: ... u1, u2, v2, v3 ...
                # New: ... u1, v2, u2, v3 ...
                # Removed edges: (u1, u2), (u2, v2), (v2, v3)
                # Added edges: (u1, v2), (v2, u2), (u2, v3)
                # My delta calculation:
                # i part:
                #   if i>0: -O(u1,u2) + O(u1,v2)
                #   if i<M-1: -O(u2,u3) + O(v2,u3)  <-- u3 is v2 here? No, u3 is current_order[i+1] which is v2.
                #   So -O(u2,v2) + O(v2,v2)? No, u3 is v2.
                #   Wait, if j=i+1, then u3 = v2.
                #   So delta += O(v2, v2)? That's wrong.
                #   The edge (u2, u3) becomes (v2, u3) where u3=v2. So O(v2, v2).
                #   But the edge (u2, v2) is removed.
                #   And the edge (v2, u3) is added.
                #   This logic is getting messy with adjacent swaps.
                
                # Let's just skip adjacent swaps or handle them separately.
                # For simplicity, if j == i+1, we can just calculate the new local score.
                # But for now, let's just use non-adjacent swaps or accept the approximation.
                # Actually, for j=i+1:
                # The term -O(u2, u3) + O(v2, u3) becomes -O(u2, v2) + O(v2, v2).
                # And the term -O(v1, v2) + O(v1, u2) becomes -O(u2, v2) + O(u2, u2).
                # This is incorrect.
                
                # Let's just implement a simple function to get the score of a segment and use it for small moves.
                # Or just do a full recalculation if delta > 0? No, too slow.
                
                # Let's just do the swap if delta > 0, but be careful with adjacent.
                # If j == i+1, we'll just skip for now to avoid bugs.
                if j == i + 1:
                    continue
                    
                if delta > 0:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                    improved = True

    # Build the result
    # We need to construct the string.
    # Start with S_{n-1} (which is 'word')
    # Append blocks for perms in best_order
    
    result = list(word)
    
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        # Calculate overlap with current end of result
        # result ends with the previous block's end, which is the previous perm.
        # Except for the first one, result ends with S_{n-1}'s end.
        
        # We can just use the overlap logic from the seed code
        overlap = next(o for o in range(min(len(result), len(block)), -1, -1) if result[len(result) - o:] == block[:o])
        result += block[overlap:]
        
    return "".join(map(str, result))

def construct_standard(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

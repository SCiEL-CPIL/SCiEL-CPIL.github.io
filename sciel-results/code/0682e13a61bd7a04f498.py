import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Recursively get the superpermutation for n-1
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Step 2: Extract the sequence of (n-1)-permutations (blocks) from prev_word
    # In the standard superpermutation, the (n-1)-permutations appear as a sequence of unique permutations.
    # We identify them by scanning windows of size n-1.
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    # Step 3: Prepare for Local Search
    # The length of the superpermutation for n, given an order of blocks $B_1, \dots, B_M$, is:
    # L = len(B_1) + 1 + sum_{i=2}^M (1 + len(B_i) - overlap(B_{i-1}, B_i))
    # Wait, let's re-derive the length formula carefully.
    # Standard construction: $S_n = B_1, n, B_1, B_2, n, B_2, \dots, B_M, n, B_M$? No.
    # The recursive step is: $S_n$ is formed by taking each permutation $P$ in $S_{n-1}$ and replacing it with $P, n, P$.
    # However, adjacent $P$'s in $S_{n-1}$ overlap.
    # Let the blocks in $S_{n-1}$ be $B_1, \dots, B_M$.
    # The word is constructed by concatenating segments $S_i = B_i, n, B_i$.
    # The overlap is between the end of $S_i$ (which is $B_i$) and the start of $S_{i+1}$ (which is $B_{i+1}$).
    # So the length is:
    # L = len(S_1) + sum_{i=2}^M (len(S_i) - overlap(B_{i-1}, B_i))
    # len(S_i) = 2*len(B_i) + 1 = 2(n-1) + 1 = 2n - 1.
    # Let $w = 2n - 1$.
    # L = w + sum_{i=2}^M (w - overlap(B_{i-1}, B_i))
    # L = M*w - sum_{i=1}^{M-1} overlap(B_i, B_{i+1})
    # To minimize L, we must maximize the sum of overlaps $O = \sum_{i=1}^{M-1} overlap(B_i, B_{i+1})$.
    
    M = len(blocks)
    # Precompute overlap matrix: overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # k can be from 0 to n-2.
    overlap_matrix = [[0] * M for _ in range(M)]
    for i in range(M):
        bi = blocks[i]
        for j in range(M):
            bj = blocks[j]
            if i == j:
                # Self overlap? In the sequence, we don't have Bi followed by Bi immediately usually,
                # but in the formula, we sum over adjacent pairs in the permutation of blocks.
                # If the order has ... Bi, Bi ..., the overlap is max k s.t. bi[-k:]==bi[:k].
                # For a permutation of distinct elements, bi[-1] != bi[0] generally?
                # Actually, bi is a permutation of 1..n-1.
                # Max overlap is n-2.
                pass
            max_k = n - 2
            best_k = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    best_k = k
                    break
            overlap_matrix[i][j] = best_k
            
    # Initial order: the order they appear in the standard construction (which is the order in `blocks`)
    current_order = list(range(M))
    
    def calc_score(order):
        # Returns total overlap sum. We want to maximize this.
        total_overlap = 0
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i+1]
            total_overlap += overlap_matrix[u][v]
        return total_overlap
        
    best_score = calc_score(current_order)
    best_order = current_order[:]
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for reconstruction and overhead
    
    # VNS / Local Search
    # Moves:
    # 1. Swap two adjacent blocks (2-opt on neighbors)
    # 2. Reverse a segment of blocks (2-opt)
    # 3. Move a block to a different position (Or-opt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 1. Adjacent Swap
        for i in range(M - 1):
            # Try swapping i and i+1
            # New order: ... A, B, C ... -> ... B, A, C ...
            # Change in overlap:
            # Old: ov(L, A) + ov(A, B) + ov(B, R)  (if i>0 and i<M-2)
            # New: ov(L, B) + ov(B, A) + ov(A, R)
            # We can just recalc score for speed simplicity if M is small, but M=720 for n=7.
            # Recalc is O(M). 720 * 720 * 720 is too slow.
            # We must use delta calculation.
            
            left = i - 1
            right = i + 2
            
            old_1 = overlap_matrix[left][i] if left >= 0 else 0
            old_2 = overlap_matrix[i][i+1]
            old_3 = overlap_matrix[i+1][right] if right < M else 0
            
            new_1 = overlap_matrix[left][i+1] if left >= 0 else 0
            new_2 = overlap_matrix[i+1][i]
            new_3 = overlap_matrix[i][right] if right < M else 0
            
            delta = (new_1 + new_2 + new_3) - (old_1 + old_2 + old_3)
            
            if delta > 0:
                # Apply swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                best_score += delta
                improved = True
                if time.time() - start_time > time_limit:
                    break
                    
        if time.time() - start_time > time_limit:
            break
            
        # 2. Segment Reversal (2-opt)
        # This is expensive to check all i,j. We sample or limit range.
        # For M=720, O(M^2) is ~500k checks. Each check is O(1) delta if we had prefix sums of overlaps?
        # Sum of overlaps is not additive in a simple prefix sum way for reversals because internal overlaps change.
        # Reversal of A..B changes internal overlaps to reverse order overlaps.
        # This is complex. Let's stick to Swaps and Or-opt (move one element).
        
        # 3. Or-opt: Move block i to position j
        for i in range(M):
            if time.time() - start_time > time_limit:
                break
            for j in range(M):
                if i == j or i == j + 1:
                    continue # Adjacent swaps are handled above, and i->i is no-op
                
                # Remove i and insert at j
                # This is O(M) to apply and O(1) to calculate delta if we are careful?
                # Actually, removing i breaks 2 links. Inserting at j breaks 2 links.
                # If i and j are far apart, 4 links change.
                # If they are close, fewer.
                
                # Let's just do a simple "Move" heuristic:
                # Try moving each block to a random position? No, systematic is better.
                # Given time constraints, let's try a limited 2-opt on reversals with small segments first?
                # Or just rely on the adjacent swaps which might be enough to improve from the standard order.
                
                # Let's add a "Reverse small segment" move.
                # Reverse segment [i, i+2] (length 3)
                # This is effectively a swap of neighbors with a shift?
                
                # Let's try reversing a segment of length 3 or 4.
                for L in [3, 4]:
                    if i + L > M:
                        continue
                    # Reverse i to i+L-1
                    seg = current_order[i:i+L]
                    rev_seg = seg[::-1]
                    
                    # Calculate delta
                    # Links affected:
                    # (i-1, i), (i, i+1), ..., (i+L-2, i+L-1), (i+L-1, i+L)
                    
                    left = i - 1
                    right = i + L
                    
                    # Old overlaps in segment
                    old_ov = 0
                    if left >= 0:
                        old_ov += overlap_matrix[left][current_order[i]]
                    for k in range(L - 1):
                        old_ov += overlap_matrix[current_order[i+k]][current_order[i+k+1]]
                    if right < M:
                        old_ov += overlap_matrix[current_order[i+L-1]][current_order[right]]
                        
                    # New overlaps in segment
                    new_ov = 0
                    if left >= 0:
                        new_ov += overlap_matrix[left][current_order[i+L-1]]
                    for k in range(L - 1):
                        new_ov += overlap_matrix[current_order[i+L-1-k]][current_order[i+L-2-k]]
                    if right < M:
                        new_ov += overlap_matrix[current_order[i]][current_order[right]]
                        
                    delta = new_ov - old_ov
                    
                    if delta > 0:
                        current_order[i:i+L] = rev_seg
                        best_score += delta
                        improved = True
                        if time.time() - start_time > time_limit:
                            break
                            
        if time.time() - start_time > time_limit:
            break

    # Reconstruct the word from best_order
    # The word is formed by concatenating segments S_i = B_{order[i]}, n, B_{order[i]}
    # With overlaps between S_i and S_{i+1}.
    # S_i ends with B_{order[i]}. S_{i+1} starts with B_{order[i+1]}.
    # Overlap k = overlap_matrix[order[i]][order[i+1]].
    
    word = []
    
    if M == 0:
        return ""
        
    # First block
    first_block = blocks[best_order[0]]
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    for idx in range(1, M):
        curr_block = blocks[best_order[idx-1]]
        next_block = blocks[best_order[idx]]
        
        # Overlap between curr_block (end of prev segment) and next_block (start of next segment)
        k = overlap_matrix[best_order[idx-1]][best_order[idx]]
        
        # The next segment is next_block, n, next_block
        # We have already appended the previous segment.
        # The previous segment ended with curr_block.
        # We need to append the next segment, overlapping the suffix of curr_block with prefix of next_block.
        
        # Wait, the segment S_i is B_i, n, B_i.
        # The end of S_i is B_i.
        # The start of S_{i+1} is B_{i+1}.
        # So we overlap suffix of B_i with prefix of B_{i+1}.
        
        # We have already written S_i.
        # We need to write S_{i+1} = B_{i+1}, n, B_{i+1}.
        # But we can overlap the last k chars of B_i with the first k chars of B_{i+1}.
        # So we append B_{i+1}[k:], then n, then B_{i+1}.
        
        word.extend(next_block[k:])
        word.append(n)
        word.extend(next_block)
        
    return "".join(map(str, word))

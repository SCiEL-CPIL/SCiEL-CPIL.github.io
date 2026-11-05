# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 6:
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
        return "".join(map(str, word))
    
    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
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
        
        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Precompute overlap matrix
        # ov[i][j] = max overlap between perms[i] and perms[j]
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                ov[i][j] = best
        
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word
        
        # Initialize with standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)
        
        # Efficient Local Search
        # We maximize total overlap.
        # Cost of path = sum(ov[path[i]][path[i+1]])
        
        # Helper to get overlap contribution of a segment or neighbors
        # For 2-opt (reverse segment i..j):
        # Old cost: ov[path[i-1]][path[i]] + ov[path[j]][path[j+1]] (with boundary checks)
        # New cost: ov[path[i-1]][path[j]] + ov[path[i]][path[j+1]]
        
        # For 1-opt (move item i to j):
        # Remove i, insert at j.
        
        # For 2-opt-relocate (move block i..i+1 to j):
        
        improved = True
        passes = 0
        max_passes = 500 # Limit to ensure time
        
        while improved and time.time() < deadline - 10 and passes < max_passes:
            improved = False
            passes += 1
            current_path = best_path[:]
            current_overlap = best_overlap
            
            # 1. 2-Opt (Reverse segment) - Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            for i in range(N - 1):
                left_idx = i - 1
                right_idx = i + 1
                # Cost of connections involving i and i+1
                # If i=0, no left. If i=N-2, no right for i+1.
                old_left = ov[left_idx][i] if left_idx >= 0 else 0
                old_right = ov[i+1][right_idx] if right_idx < N else 0
                old_mid = ov[i][i+1]
                old_cost = old_left + old_mid + old_right
                
                for j in range(i + 2, N):
                    # Segment i..j is reversed.
                    # New neighbors for i: left_idx and j
                    # New neighbors for j: i and right_idx (which is j+1 in original, but now j is at pos i+1... wait)
                    # Original: ... L, P_i, P_{i+1}, ..., P_j, R ...
                    # Reversed: ... L, P_j, ..., P_{i+1}, P_i, R ...
                    # New left neighbor of P_j is L (left_idx).
                    # New right neighbor of P_i is R (right_idx = j+1).
                    # New left neighbor of P_i is P_{i+1}.
                    # New right neighbor of P_j is P_{j-1}.
                    # Wait, the internal overlaps change too?
                    # No, 2-opt on a Hamiltonian path (or TSP path) usually assumes the cost is only on the edges.
                    # Here, the "cost" is the sum of overlaps between adjacent permutations.
                    # When we reverse a segment, the internal edges are reversed.
                    # ov[A][B] is NOT necessarily equal to ov[B][A].
                    # So we MUST recalculate the internal cost of the reversed segment.
                    # This makes 2-opt $O(N^3)$ or $O(N^4)$ if not careful.
                    # However, N=720. $N^3$ is 370M. Too slow for Python.
                    
                    # We can precompute prefix sums of overlaps?
                    # No, because ov is not symmetric.
                    # sum(ov[P[k]][P[k+1]]) for k in i..j-1.
                    # Reversed sum is sum(ov[P[k+1]][P[k]]) for k in i..j-1.
                    # We can precompute `fwd_overlap[i][j]` and `bwd_overlap[i][j]`?
                    # That's $O(N^2)$ space and $O(N^3)$ time to build. $720^3 \approx 370M$.
                    # In C, 370M ops is < 1s. In Python, it's ~30-60s.
                    # We have 115s budget.
                    # Maybe we can just do 1-Opt and 2-Opt-Relocate which are easier to evaluate?
                    # 1-Opt: Move single element. Internal cost unchanged. Only boundary costs change. $O(1)$ evaluation.
                    # 2-Opt-Relocate: Move 2 elements. Internal cost of the 2 elements (1 edge) changes direction?
                    # Block [A, B]. Move to new location.
                    # Original: ... X, A, B, Y ...
                    # New: ... U, A, B, V ...
                    # Cost change:
                    # Remove: ov[X][A] + ov[A][B] + ov[B][Y]
                    # Insert: ov[U][A] + ov[A][B] + ov[B][V]
                    # Note: ov[A][B] remains the same!
                    # So we only need to check boundary changes. $O(1)$ evaluation.
                    
                    # So 1-Opt and 2-Opt-Relocate are fast.
                    # 2-Opt (Reverse) is slow due to asymmetric overlaps.
                    # We will skip Reverse 2-Opt or implement it with precomputed prefix sums if we have time.
                    # Let's stick to 1-Opt and 2-Opt-Relocate first.
                    
                    break # Skip the inner loop for now, just to structure the code. We will implement 1-Opt below.
            
            # 2. 1-Opt (Relocate single element) - Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            for i in range(N):
                # Remove i
                if i > 0:
                    left = current_path[i-1]
                else:
                    left = -1
                if i < N - 1:
                    right = current_path[i+1]
                else:
                    right = -1
                
                # Cost of removing i
                # Original edges: (left, i) and (i, right)
                # New edge: (left, right)
                # Gain from removal = ov[left][right] - (ov[left][i] + ov[i][right])
                # (Handling boundaries: if left==-1, no left edge. etc.)
                
                gain_remove = 0
                if left != -1 and right != -1:
                    gain_remove = ov[left][right] - ov[left][i] - ov[i][right]
                elif left != -1:
                    gain_remove = 0 - ov[left][i]
                elif right != -1:
                    gain_remove = 0 - ov[i][right]
                else:
                    gain_remove = 0
                    
                item = current_path[i]
                
                # Try inserting at all positions j
                for j in range(N):
                    if j == i or j == i + 1:
                        continue # No change or same position
                    
                    # Determine neighbors in the "removed" path
                    # If j < i, indices shift left by 1.
                    # If j > i, indices shift right by 1 (relative to original, but we are inserting into the gap).
                    
                    # Let's simulate the path without i.
                    # If j is the insertion point in the NEW path (length N-1).
                    # It corresponds to index j in the original path if j < i, or j-1 if j > i?
                    # Actually, it's easier to think:
                    # We remove i. Path becomes P'.
                    # We insert item at position j in P'.
                    # If j == 0: Left=-1, Right=P'[0]
                    # If j == N-1: Left=P'[N-2], Right=-1
                    # Else: Left=P'[j-1], Right=P'[j]
                    
                    # Mapping j in P' to original indices:
                    # If j < i:
                    #   Left in P' is P'[j-1] which is original P[j-1] (if j>0)
                    #   Right in P' is P'[j] which is original P[j]
                    # If j >= i:
                    #   Left in P' is P'[j-1] which is original P[j] (since i removed, P[j] shifts to j-1? No.)
                    #   Original: 0, 1, ..., i-1, i, i+1, ..., N-1
                    #   Removed i: 0, 1, ..., i-1, i+1, ..., N-1
                    #   Index k in P' corresponds to:
                    #     if k < i: original k
                    #     if k >= i: original k+1
                    
                    # So if we insert at position j in P':
                    # The neighbors in P' are at j-1 and j.
                    # Neighbor Left (index j-1 in P'):
                    #   If j-1 < i: orig_idx = j-1
                    #   Else: orig_idx = j
                    # Neighbor Right (index j in P'):
                    #   If j < i: orig_idx = j
                    #   Else: orig_idx = j+1
                    
                    # Wait, if j == i, we are inserting back at the same spot?
                    # We skip j == i and j == i+1.
                    
                    # Let's just calculate neighbors directly.
                    if j < i:
                        left_orig = j - 1 if j > 0 else -1
                        right_orig = j
                    else:
                        # j > i (since j != i, i+1)
                        # If j == i+1, we skipped.
                        # So j >= i+2.
                        # Left in P' is at index j-1.
                        # If j-1 >= i, orig_idx = (j-1)+1 = j.
                        # If j-1 < i, orig_idx = j-1.
                        # But j >= i+2 implies j-1 >= i+1 > i. So left_orig = j.
                        # Right in P' is at index j.
                        # If j >= i, orig_idx = j+1.
                        left_orig = j
                        right_orig = j + 1 if j < N - 1 else -1 # Careful with bounds in P'
                        # P' has length N-1. Max index N-2.
                        # If j == N-1 (end of P'), right is -1.
                        # In original, j=N-1 means inserting at end.
                        # Right neighbor in P' is none.
                        # Left neighbor in P' is P'[N-2].
                        # If N-2 >= i (which is true if i < N-1), orig_idx = N-2+1 = N-1.
                        # If i == N-1, we removed the last element. P' ends at N-2.
                        # j=N-1 in P' is invalid? No, P' length is N-1. Indices 0..N-2.
                        # So j can be 0..N-1? No, insertion point j is 0..N-1 (before element j, or after last).
                        # Standard insertion: j is 0..N-1.
                        # If j == N-1, insert at end.
                        
                        # Let's refine the mapping.
                        # If j > i:
                        #   Left neighbor in P' is at index j-1.
                        #   Since j > i, j-1 >= i.
                        #   Orig index for P'[j-1] is (j-1) + 1 = j.
                        #   Right neighbor in P' is at index j.
                        #   If j < N-1: P'[j] exists. Orig index is j+1.
                        #   If j == N-1: No right neighbor.
                        
                        if j < N - 1:
                            right_orig = j + 1
                        else:
                            right_orig = -1

                    # Calculate gain from insertion
                    gain_insert = 0
                    if left_orig != -1 and right_orig != -1:
                        gain_insert = ov[left_orig][item] + ov[item][right_orig] - ov[left_orig][right_orig]
                    elif left_orig != -1:
                        gain_insert = ov[left_orig][item]
                    elif right_orig != -1:
                        gain_insert = ov[item][right_orig]
                    else:
                        gain_insert = 0
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i = best_i
                j = best_j
                item = current_path[i]
                del current_path[i]
                # Adjust j if i < j
                if i < j:
                    j -= 1
                current_path.insert(j, item)
                current_overlap += best_gain
                best_path = current_path
                best_overlap = current_overlap
                improved = True
                continue # Restart pass? Or continue? Best Improvement usually does one move per pass.
            
            # 3. 2-Opt-Relocate (Move block of 2)
            if not improved:
                best_gain = 0
                best_i = -1
                best_j = -1
                for i in range(N - 1):
                    block = current_path[i:i+2]
                    # Remove block i, i+1
                    if i > 0:
                        left = current_path[i-1]
                    else:
                        left = -1
                    if i + 2 < N:
                        right = current_path[i+2]
                    else:
                        right = -1
                    
                    gain_remove = 0
                    if left != -1 and right != -1:
                        gain_remove = ov[left][right] - ov[left][block[0]] - ov[block[0]][block[1]] - ov[block[1]][right]
                    elif left != -1:
                        gain_remove = 0 - ov[left][block[0]] - ov[block[0]][block[1]]
                    elif right != -1:
                        gain_remove = 0 - ov[block[0]][block[1]] - ov[block[1]][right]
                    else:
                        gain_remove = 0 - ov[block[0]][block[1]]
                    
                    # Try inserting block at j
                    for j in range(N - 1):
                        if j == i or j == i + 1:
                            continue
                        
                        # Mapping j in P' (length N-2) to original
                        # If j < i:
                        #   Left in P' at j-1 -> orig j-1
                        #   Right in P' at j -> orig j
                        # If j >= i:
                        #   Left in P' at j-1 -> orig j+1 (since 2 removed)
                        #   Right in P' at j -> orig j+2
                        
                        if j < i:
                            left_orig = j - 1 if j > 0 else -1
                            right_orig = j
                        else:
                            # j > i (skipped i, i+1)
                            # j >= i+2
                            # Left in P' at j-1.
                            # If j-1 >= i, orig = (j-1)+2 = j+1.
                            # Right in P' at j.
                            # If j < N-2, orig = j+2.
                            # If j == N-2, right = -1.
                            left_orig = j + 1
                            if j < N - 2:
                                right_orig = j + 2
                            else:
                                right_orig = -1
                            
                        gain_insert = 0
                        if left_orig != -1 and right_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]] + ov[block[1]][right_orig] - ov[left_orig][right_orig]
                        elif left_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]]
                        elif right_orig != -1:
                            gain_insert = ov[block[0]][block[1]] + ov[block[1]][right_orig]
                        else:
                            gain_insert = ov[block[0]][block[1]]
                        
                        total_gain = gain_remove + gain_insert
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_i = i
                            best_j = j
                
                if best_gain > 0:
                    i = best_i
                    j = best_j
                    block = current_path[i:i+2]
                    del current_path[i:i+2]
                    if i < j:
                        j -= 2
                    current_path.insert(j, block[0])
                    current_path.insert(j+1, block[1])
                    current_overlap += best_gain
                    best_path = current_path
                    best_overlap = current_overlap
                    improved = True
        
        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
    # Fallback for n > 7
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
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

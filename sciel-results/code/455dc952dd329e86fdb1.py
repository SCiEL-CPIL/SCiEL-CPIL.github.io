import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 5s margin for return/safety

    if n == 1:
        return "1"

    # Standard recursive construction for m up to n-1
    word = [1]
    for m in range(2, n):
        target_set = set(range(1, m))
        blocks = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(window) == m - 1 and set(window) == target_set:
                if window not in seen:
                    seen.add(window)
                    blocks.append(window)

        new_word = []
        for p in blocks:
            unit = list(p) + [m] + list(p)
            if not new_word:
                new_word.extend(unit)
            else:
                # Find max overlap
                max_k = min(len(new_word), len(unit))
                ov = 0
                for k in range(max_k, 0, -1):
                    if new_word[-k:] == unit[:k]:
                        ov = k
                        break
                new_word.extend(unit[ov:])
        word = new_word

    if n == 1:
        return "".join(map(str, word))

    # Now we have a superpermutation of 1..n-1.
    # We need to add n.
    # Extract all permutations of 1..n-1 in order of appearance.
    m = n
    target_set = set(range(1, m))
    perms = []
    seen = set()
    for i in range(len(word) - (m - 1) + 1):
        window = tuple(word[i:i + m - 1])
        if len(window) == m - 1 and set(window) == target_set:
            if window not in seen:
                seen.add(window)
                perms.append(window)

    N = len(perms)

    if N == 1:
        # Just in case, though for n>=2 N>1
        unit = list(perms[0]) + [m] + list(perms[0])
        return "".join(map(str, unit))

    # Precompute overlap matrix: overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Note: The unit is P + [m] + P.
    # When appending unit_j after unit_i, the overlap is between the suffix of unit_i and prefix of unit_j.
    # unit_i ends with P_i. unit_j starts with P_j.
    # So the overlap is determined by suffix of P_i and prefix of P_j.
    # Max possible overlap is m-2 (since P_i and P_j are permutations of 1..m-1, they can't match fully unless i==j, but i!=j in the path).
    # Actually, if P_i == P_j, overlap is m-1. But in a path of distinct perms, i!=j implies P_i != P_j.
    # So max overlap is m-2.
    
    overlap_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            if i == j:
                overlap_mat[i][j] = m - 1
            else:
                pj = perms[j]
                ov = 0
                # Check k from m-2 down to 0
                for k in range(m - 2, -1, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

    # Helper to calculate total overlap for a path
    def calc_total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap_mat[path[i]][path[i + 1]]
        return total

    # Helper to build word from path
    def build_word_from_path(path):
        w = []
        for idx in path:
            perm = perms[idx]
            unit = list(perm) + [m] + list(perm)
            if not w:
                w.extend(unit)
            else:
                # Find overlap
                # We know the overlap is determined by perms[path[-1]] and perms[idx]
                # But to be safe and simple, just do the list comparison
                # Optimization: use the precomputed overlap_mat
                prev_idx = path[path.index(idx) - 1] if idx != path[0] else -1
                # Wait, path is a list of indices.
                # Let's just use the logic from the loop.
                # Actually, we can just use the overlap_mat value directly.
                # overlap_mat[prev_idx][idx] gives the length of the overlap.
                # But we need to be careful with the first element.
                
                # Let's just recompute overlap for simplicity and correctness, it's O(m)
                max_k = min(len(w), len(unit))
                ov = 0
                for k in range(max_k, 0, -1):
                    if w[-k:] == unit[:k]:
                        ov = k
                        break
                w.extend(unit[ov:])
        return "".join(map(str, w))

    # 1. Generate Naive Word (standard order)
    naive_path = list(range(N))
    naive_overlap = calc_total_overlap(naive_path)
    best_path = naive_path[:]
    best_overlap = naive_overlap

    # 2. Heuristic Optimization
    # Nearest Neighbor
    random.seed(42)
    unvisited = list(range(N))
    random.shuffle(unvisited)
    path = [unvisited.pop()]

    while unvisited:
        last = path[-1]
        best_idx = -1
        best_ov = -1
        for idx in unvisited:
            ov = overlap_mat[last][idx]
            if ov > best_ov:
                best_ov = ov
                best_idx = idx
        path.append(best_idx)
        unvisited.remove(best_idx)

    current_overlap = calc_total_overlap(path)
    if current_overlap > best_overlap:
        best_overlap = current_overlap
        best_path = path[:]

    # 2-opt + Or-Opt improvement
    # We work on a copy of the best path so far? Or just the current one?
    # Usually, we take the best found so far and try to improve it.
    # Let's take the better of naive and NN as the starting point for local search.
    current_path = best_path[:]
    current_overlap_val = best_overlap

    improved = True
    # We will loop until no improvement or time runs out
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 2-Opt
        # For each i, j, check if reversing path[i+1..j] improves overlap.
        # Overlap change:
        # Old: ov(path[i], path[i+1]) + ov(path[j], path[j+1])
        # New: ov(path[i], path[j]) + ov(path[i+1], path[j+1])
        # Note: if i+1 == j, it's a swap.
        
        N_path = len(current_path)
        
        # To make it faster, we can iterate i and j.
        # N is 720. N^2 is ~500k.
        # Each check is O(1) with precomputed matrix.
        # 500k * 10 passes = 5M ops. Should be fast enough in Python?
        # 5M simple list lookups and comparisons in Python might take ~1-2 seconds.
        # Let's try.
        
        for i in range(N_path - 2):
            if time.time() > deadline - 5.0:
                break
            pi = current_path[i]
            pi1 = current_path[i + 1]
            old_ov1 = overlap_mat[pi][pi1]
            
            for j in range(i + 1, N_path - 1):
                pj = current_path[j]
                pj1 = current_path[j + 1]
                old_ov2 = overlap_mat[pj][pj1]
                
                new_ov1 = overlap_mat[pi][pj]
                new_ov2 = overlap_mat[pi1][pj1]
                
                if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                    # Reverse segment i+1 to j
                    current_path[i + 1:j + 1] = reversed(current_path[i + 1:j + 1])
                    current_overlap_val += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                    improved = True
                    # Break inner loop? Or continue?
                    # Usually breaking and restarting is safer for 2-opt, but continuing might find more improvements in one pass.
                    # Let's break to restart the outer loop for a "first improvement" strategy, or continue for "best improvement"?
                    # First improvement is faster per pass.
                    break
            
            # If we improved, we break the inner loop.
            # We might want to restart the outer loop to ensure consistency?
            # Or just continue. Continuing is fine.
        
        # Or-Opt (Move segment of length 1 or 2)
        if not improved and time.time() < deadline - 5.0:
            for L in [1, 2]:
                if time.time() > deadline - 5.0:
                    break
                for i in range(N_path - L):
                    if time.time() > deadline - 5.0:
                        break
                    # Segment to move: current_path[i : i+L]
                    # Left neighbor: i-1
                    # Right neighbor: i+L
                    # 
                    # We try to insert the segment at position k (0 <= k < N_path - L)
                    # This is O(N^3) which is 720^3 ~ 373M. Too slow.
                    # We only try to move to nearby positions or best position?
                    # Let's just try to move to the position that maximizes local gain.
                    # Or just try moving to i+1 (swap), i-1 (swap), etc.
                    # Actually, 2-opt already handles swaps (L=1, j=i+1 -> reverse length 1 segment? No, 2-opt reverses i+1..j. If j=i+1, it reverses a single element, which is a no-op for the element itself, but changes neighbors?
                    # Wait. 2-opt with j=i+1 reverses path[i+1:i+2] which is just path[i+1]. It doesn't swap path[i] and path[i+1].
                    # So 2-opt does NOT handle swaps!
                    # We need to handle swaps separately or use a different move.
                    # A swap is equivalent to moving a segment of length 1 to position i+1 or i-1.
                    
                    # Let's implement a specific swap check first, as it's O(N^2) and very effective.
                    if L == 1:
                        # Swap i and i+1
                        if i + 1 < N_path:
                            pi = current_path[i]
                            pi1 = current_path[i + 1]
                            left = current_path[i - 1] if i > 0 else None
                            right = current_path[i + 2] if i + 2 < N_path else None
                            
                            old_cost = 0
                            if left is not None:
                                old_cost += overlap_mat[left][pi]
                            old_cost += overlap_mat[pi][pi1]
                            if right is not None:
                                old_cost += overlap_mat[pi1][right]
                            
                            new_cost = 0
                            if left is not None:
                                new_cost += overlap_mat[left][pi1]
                            new_cost += overlap_mat[pi1][pi]
                            if right is not None:
                                new_cost += overlap_mat[pi][right]
                            
                            if new_cost > old_cost:
                                current_path[i], current_path[i + 1] = pi1, pi
                                current_overlap_val += new_cost - old_cost
                                improved = True
                                # Don't break, try other swaps
                    else:
                        # L=2, move segment of length 2.
                        # This is more complex. Let's skip for now to ensure speed, or implement a limited version.
                        # Given time, let's stick to swaps and 2-opt reversals.
                        pass
        
        # Update best
        if current_overlap_val > best_overlap:
            best_overlap = current_overlap_val
            best_path = current_path[:]

    # Construct final word
    final_word = build_word_from_path(best_path)
    
    return final_word

import time
import random
from itertools import permutations

_START_TIME = time.time()
DEADLINE = 145.0

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME)

def _standard_construct(n):
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _overlap_len(a, b):
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0

def _construct_from_order(order, n):
    if not order:
        return list(range(1, n + 1))
    word = list(order[0]) + [n] + list(order[0])
    for i in range(1, len(order)):
        prev = order[i - 1]
        curr = order[i]
        block = list(curr) + [n] + list(curr)
        ov = 0
        max_ov = min(len(prev), len(curr))
        for o in range(max_ov, 0, -1):
            if prev[-o:] == curr[:o]:
                ov = o
                break
        word += block[ov:]
    return word

def _verify_superpermutation(word, n):
    if len(word) < n:
        return False
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        found.add(window)
    return len(found) == _factorial(n)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _construct_n7_optimized():
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    overlap_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                overlap_matrix[i][j] = _overlap_len(list(perms_6[i]), list(perms_6[j]))
    
    word_6 = _standard_construct(6)
    order_6 = []
    seen = set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            order_6.append(window)
            
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    best_path = [perm_to_idx[p] for p in order_6]
    
    def path_score(p):
        total = 0
        for i in range(len(p) - 1):
            total += overlap_matrix[p[i]][p[i+1]]
        return total
    
    current_score = path_score(best_path)
    
    # Or-Opt Local Search
    # Move segment of length k from i to j
    # Internal cost is invariant.
    # Only boundary edges change.
    
    improved = True
    while improved:
        improved = False
        for k in [1, 2, 3]:
            for i in range(n_perms - k):
                if _remaining_time() < 10.0:
                    return _construct_from_order([perms_6[idx] for idx in best_path], 7)
                
                # Segment is best_path[i : i+k]
                # Edges to remove:
                # Left: (best_path[i-1], best_path[i]) if i > 0 else 0
                # Right: (best_path[i+k-1], best_path[i+k]) if i+k < n_perms else 0
                
                old_left = overlap_matrix[best_path[i-1]][best_path[i]] if i > 0 else 0
                old_right = overlap_matrix[best_path[i+k-1]][best_path[i+k]] if i+k < n_perms else 0
                
                for j in range(n_perms - k + 1):
                    # Skip if j is in [i, i+k)
                    if i <= j < i + k:
                        continue
                    
                    # Edges to add:
                    # The segment is inserted at j.
                    # New Left: (best_path[j-1], best_path[i]) if j > 0 else 0
                    # New Right: (best_path[i+k-1], best_path[j+k]) if j+k < n_perms else 0
                    # Note: indices shift if j > i+k.
                    
                    # Let's determine the neighbors in the NEW path.
                    # If j < i:
                    #   The segment is moved to the left.
                    #   The element before the segment in new path is best_path[j-1] (if j>0)
                    #   The element after the segment in new path is best_path[j+k] (which is best_path[i] in old path? No, best_path[j] in old path if j < i)
                    
                    # This index manipulation is error-prone.
                    # Let's just reconstruct the path and calculate score for promising moves?
                    # No, that's O(N).
                    
                    # Let's use a helper function to get the "effective" neighbors.
                    # Actually, simpler approach:
                    # If we move segment S from i to j:
                    # The gap at i is closed.
                    # The segment is placed at j.
                    
                    # Case 1: j < i
                    #   New Left Neighbor: best_path[j-1] (if j>0)
                    #   New Right Neighbor: best_path[j] (which becomes best_path[j+k] in new path? No, best_path[j] is the element that was at j, now shifted to j+k)
                    #   Wait, if we remove i..i+k-1, the elements i+k..end shift left by k.
                    #   If we insert at j, elements j..end shift right by k.
                    
                    # Let's just do a full score recalculation for the best candidate found in a small neighborhood?
                    # Or, let's stick to 2-opt but with FULL SCORE RECALCULATION if the segment is small?
                    # No, 2-opt changes internal edges.
                    
                    # Let's go back to 2-opt with full score recalculation.
                    # N=720. O(N^2) moves. Each move O(N) to score. O(N^3) = 3.7e8.
                    # In Python, 3.7e8 operations is ~30-60 seconds.
                    # We have 140s. So 1-2 passes of full 2-opt might fit.
                    
                    pass
    
    # Fallback: Just return the standard one if search is too complex.
    # But I can implement a simpler 2-opt that just checks a few random moves?
    # No, let's try to implement the Or-Opt delta correctly.
    
    # Correct Or-Opt Delta Logic:
    # Let P be the path.
    # Remove P[i:i+k].
    # Insert at j.
    # 
    # If j < i:
    #   Old Left: P[i-1] -> P[i]
    #   Old Right: P[i+k-1] -> P[i+k]
    #   New Left: P[j-1] -> P[i]  (if j>0)
    #   New Right: P[i+k-1] -> P[j] (if j+k < N? No, P[j] is the element that was at j, now at j+k. The element AFTER the inserted segment is the one that was at j. So P[i+k-1] -> P[j] is correct? 
    #   Wait. 
    #   Old: ... P[j-1] P[j] ... P[i-1] P[i] ... P[i+k-1] P[i+k] ...
    #   New: ... P[j-1] P[i] ... P[i+k-1] P[j] ... P[i-1] P[i+k] ...
    #   Edges Changed:
    #   Removed: (P[j-1], P[j]), (P[i-1], P[i]), (P[i+k-1], P[i+k])? No.
    #   The segment P[i..i+k-1] is removed.
    #   Edges removed: (P[i-1], P[i]) and (P[i+k-1], P[i+k]).
    #   Edges added: (P[j-1], P[i]) and (P[i+k-1], P[j]).
    #   (Assuming j>0 and i+k<N).
    #   Delta = Overlap(P[j-1], P[i]) + Overlap(P[i+k-1], P[j]) - Overlap(P[i-1], P[i]) - Overlap(P[i+k-1], P[i+k])
    
    # If j > i + k:
    #   Old: ... P[i-1] P[i] ... P[i+k-1] P[i+k] ... P[j-1] P[j] ...
    #   New: ... P[i-1] P[i+k] ... P[j-1] P[i] ... P[i+k-1] P[j] ...
    #   Edges removed: (P[i-1], P[i]), (P[i+k-1], P[i+k]), (P[j-1], P[j])? No.
    #   The segment is removed from i.
    #   Edges removed: (P[i-1], P[i]) and (P[i+k-1], P[i+k]).
    #   The segment is inserted at j.
    #   Edges added: (P[j-1], P[i]) and (P[i+k-1], P[j]).
    #   Wait, the element at j shifts to j+k.
    #   The element before the insertion point is P[j-1].
    #   The element after the insertion point is P[j] (which is now at j+k).
    #   So edges added: (P[j-1], P[i]) and (P[i+k-1], P[j]).
    #   This is the SAME delta formula!
    #   Delta = Overlap(P[j-1], P[i]) + Overlap(P[i+k-1], P[j]) - Overlap(P[i-1], P[i]) - Overlap(P[i+k-1], P[i+k])
    
    # Boundary conditions:
    # If i==0, Old Left is 0.
    # If i+k==N, Old Right is 0.
    # If j==0, New Left is 0.
    # If j+k==N, New Right is 0.
    
    # This allows O(1) delta calculation!
    
    improved = True
    while improved:
        improved = False
        for k in [1, 2, 3]:
            for i in range(n_perms - k):
                if _remaining_time() < 10.0:
                    return _construct_from_order([perms_6[idx] for idx in best_path], 7)
                
                old_left = overlap_matrix[best_path[i-1]][best_path[i]] if i > 0 else 0
                old_right = overlap_matrix[best_path[i+k-1]][best_path[i+k]] if i+k < n_perms else 0
                old_cost = old_left + old_right
                
                for j in range(n_perms - k + 1):
                    if i <= j < i + k:
                        continue
                    
                    new_left = overlap_matrix[best_path[j-1]][best_path[i]] if j > 0 else 0
                    new_right = overlap_matrix[best_path[i+k-1]][best_path[j]] if j + k < n_perms else 0
                    new_cost = new_left + new_right
                    
                    delta = new_cost - old_cost
                    if delta > 0:
                        # Apply move
                        segment = best_path[i:i+k]
                        del best_path[i:i+k]
                        # Adjust j if j > i
                        if j > i:
                            j -= k
                        best_path.insert(j, *segment)
                        current_score += delta
                        improved = True
                        # Restart i loop? Or continue?
                        # Continuing is fine, but indices change. 
                        # To be safe, break and restart the while loop.
                        break
                if improved:
                    break
            if improved:
                break
                
    word = _construct_from_order([perms_6[idx] for idx in best_path], 7)
    return word

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1312131"
    if n == 4:
        return "1413121413124131241"
    if n == 5:
        return _standard_construct(5)
    if n == 6:
        return _standard_construct(6)
    
    if n == 7:
        try:
            word = _construct_n7_optimized()
            if _verify_superpermutation(word, 7):
                return "".join(map(str, word))
        except Exception:
            pass
        return "".join(map(str, _standard_construct(7)))
    
    return "".join(map(str, _standard_construct(n)))

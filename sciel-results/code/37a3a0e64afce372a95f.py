import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431243121342132413214321"
    
    if n != 7:
        return _standard_superperm_str(n)

    # Optimization for n=7
    # 1. Generate the standard SP(6) to get the "standard" order of permutations
    word_6 = [1]
    for m in range(2, 7): 
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6)
    
    # 2. Precompute overlap matrix
    # M[i][j] = longest suffix of perms_6[i] matching prefix of perms_6[j]
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        # Convert to string for faster slicing? Or just tuple slicing.
        # Tuple slicing is fast.
        for j in range(num_perms):
            if i == j:
                M[i][j] = 6 # Not used in path since distinct, but good for safety
                continue
            p2 = perms_6[j]
            # Max overlap is 5 for distinct perms
            # Check 5, 4, 3, 2, 1
            # p1[-k:] == p2[:k]
            if p1[-1] == p2[0]:
                if p1[-2:] == p2[:2]:
                    if p1[-3:] == p2[:3]:
                        if p1[-4:] == p2[:4]:
                            if p1[-5:] == p2[:5]:
                                M[i][j] = 5
                            else:
                                M[i][j] = 4
                        else:
                            M[i][j] = 3
                    else:
                        M[i][j] = 2
                else:
                    M[i][j] = 1
            else:
                M[i][j] = 0

    # 3. Local Search
    # Initial order: the order in which they appear in SP(6)
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        
    best_order = order[:]
    best_score = calc_score(order)
    current_order = order[:]
    current_score = best_score

    # 2-Opt Local Search
    # We try to reverse segments to improve the score.
    # Delta for reversing [i+1 ... j]:
    # Old edges: (order[i], order[i+1]) and (order[j], order[j+1])
    # New edges: (order[i], order[j]) and (order[i+1], order[j+1])
    # Note: The edges inside the segment are reversed, so M[a][b] becomes M[b][a].
    # Wait, 2-opt in TSP usually assumes symmetric matrix or handles direction.
    # Here M is NOT symmetric. M[i][j] != M[j][i].
    # When we reverse the segment, the direction of traversal for internal edges flips.
    # So we need to sum M[order[k]][order[k-1]] for the internal part?
    # Actually, it's easier to just recalculate the score for the modified path if N is small?
    # N=720. O(N) recalculation per move is 720 ops.
    # Number of 2-opt moves is O(N^2) = 500,000.
    # 500,000 * 720 = 360,000,000 ops. Too slow for Python in 110s?
    # 360M simple ops might take 10-30s in Python. It's risky.
    
    # Let's try to calculate delta efficiently.
    # Path: ... A, B, C, ..., D, E ...
    # Reverse B..D.
    # New: ... A, D, ..., C, B, E ...
    # Old Score Part:
    # M[A][B] + M[B][C] + ... + M[C][D] + M[D][E]
    # New Score Part:
    # M[A][D] + M[D][C] + ... + M[C][B] + M[B][E]
    
    # We can precompute prefix sums of scores?
    # Score of path P is sum M[P[i]][P[i+1]].
    # If we reverse a segment, the internal edges are reversed.
    # Sum of M[P[k]][P[k+1]] for k in [i+1, j-1] becomes Sum of M[P[k+1]][P[k]] for k in [i+1, j-1].
    # Let S_forward = sum M[P[k]][P[k+1]] for k=i+1 to j-1.
    # Let S_backward = sum M[P[k+1]][P[k]] for k=i+1 to j-1.
    # We can maintain a running sum or just calculate it.
    # Calculating S_forward and S_backward for a segment of length L takes O(L).
    # Total complexity of one 2-opt pass: Sum over all (i,j) of (j-i).
    # This is O(N^3) in the worst case if we recalculate from scratch.
    # 720^3 is 373,000,000,000. Way too slow.
    
    # Alternative: Use Or-opt (move 1, 2, or 3 elements) or simple swaps.
    # Or, use the O(1) delta formula for swaps and 2-opt if we can compute the internal sum fast.
    # Actually, for 2-opt, if we just check the boundary changes and assume the internal sum is "similar" or use a heuristic?
    # No, we need exact delta.
    
    # Let's stick to Swaps (adjacent) and 2-Opt with O(1) delta if possible?
    # For adjacent swap (i, i+1):
    # Old: M[A][B] + M[B][C]
    # New: M[A][C] + M[C][B]
    # Delta is O(1).
    # This is what the current program does. It might be slow to converge.
    
    # Let's try a hybrid:
    # 1. Adjacent swaps (fast, O(1) delta).
    # 2. 2-Opt with O(L) internal sum calculation, but only for short segments?
    # Or just use the "Move" operation (remove i, insert at j) with O(N) recalc?
    # 720 * 720 * 720 is too slow for full recalc.
    
    # Let's optimize the Adjacent Swap search with multiple passes and random restarts.
    # Also, try "Or-opt" for segment length 2 and 3 with O(1) or O(2) delta?
    # Moving a segment of length 1 is just a move.
    # Moving a segment of length 2:
    # Old: ... A, B, C, D, E ...
    # Move B,C to after D: ... A, D, B, C, E ...
    # Old: M[A][B] + M[B][C] + M[C][D] + M[D][E]
    # New: M[A][D] + M[D][B] + M[B][C] + M[C][E]
    # Delta = (M[A][D] + M[D][B] + M[C][E]) - (M[A][B] + M[C][D] + M[D][E])
    # O(1) lookup!
    
    # So we can do Or-opt (move segment of length 1, 2, 3) with O(1) delta calculation.
    # This is much more powerful than adjacent swaps.
    
    def calc_delta_move_seg(order, i, j, seg_len):
        # Move segment [i, i+seg_len-1] to position j
        # Note: j is the insertion index in the ORIGINAL list before removal?
        # Let's define: Remove segment starting at i, length L.
        # Insert at index j (in the list AFTER removal).
        # This is tricky to delta-calc if j > i.
        
        # Simpler: 
        # Case 1: j < i. Move [i..i+L-1] to before j.
        # Old: ... P, S1, S2, S3, Q ... (S is segment)
        # New: ... P, Q, S1, S2, S3 ... 
        # Wait, if we insert before j, and j < i:
        # Original: ... [j-1], [j], ..., [i-1], [i]...[i+L-1], [i+L] ...
        # Remove [i]..[i+L-1].
        # List becomes: ... [j-1], [j], ..., [i-1], [i+L] ...
        # Insert at j.
        # New: ... [j-1], [i]..[i+L-1], [j], ..., [i-1], [i+L] ...
        
        # Edges affected:
        # Old:
        # (j-1, j)
        # (i-1, i)
        # (i+L-1, i+L)
        # (internal edges of segment: (i, i+1), ..., (i+L-2, i+L-1))
        
        # New:
        # (j-1, i)
        # (i+L-1, j)
        # (i-1, i+L)
        # (internal edges of segment: SAME, because order inside segment doesn't change)
        
        # Delta = (M[j-1][i] + M[i+L-1][j] + M[i-1][i+L]) - (M[j-1][j] + M[i-1][i] + M[i+L-1][i+L])
        # O(1)
        
        # Case 2: j > i.
        # Original: ... [i-1], [i]..[i+L-1], [i+L], ..., [j-1], [j] ...
        # Remove [i]..[i+L-1].
        # List: ... [i-1], [i+L], ..., [j-1], [j] ...
        # Insert at j (in new list).
        # Note: indices shift.
        # Let's use the property that the segment moves from i to j.
        # If j > i, the segment is moved to the right.
        # The "gap" left behind is closed.
        # Original edges:
        # (i-1, i)
        # (i+L-1, i+L)
        # (j-1, j)  <-- Note: j here is index in ORIGINAL list?
        # Let's say we want to insert AFTER element at index j-1 in the NEW list.
        # This is getting confusing.
        
        # Let's just use Adjacent Swaps and Full Recalc for Moves if time permits?
        # No, full recalc is too slow.
        
        # Let's implement Adjacent Swaps and "2-Opt" (reverse segment) with O(L) internal sum.
        # If we limit segment length to 50, O(L) is small.
        # But 2-opt usually needs long segments.
        
        # Let's stick to Adjacent Swaps but run MANY passes.
        # And add a "Random Restart" if stuck.
        
        pass

    # Strategy:
    # 1. Run Adjacent Swap Hill Climbing until no improvement.
    # 2. If stuck, do a random perturbation (swap 5 random pairs) and continue.
    # 3. Repeat until time limit.
    
    random.seed(42)
    
    def try_swaps(order, score, max_passes=100):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(num_perms - 1):
                if time.time() > deadline:
                    return order, score
                # Calculate delta for swap i, i+1
                A = order[i-1] if i > 0 else None
                B = order[i]
                C = order[i+1]
                D = order[i+2] if i+2 < num_perms else None
                
                old_part = 0
                new_part = 0
                
                if A is not None:
                    old_part += M[A][B]
                    new_part += M[A][C]
                    
                old_part += M[B][C]
                new_part += M[C][B]
                
                if D is not None:
                    old_part += M[C][D]
                    new_part += M[B][D]
                    
                delta = new_part - old_part
                if delta > 0:
                    order[i], order[i+1] = C, B
                    score += delta
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_order = order[:]
        return order, score

    current_order, current_score = try_swaps(current_order, current_score)
    
    # If we have time, try some random restarts
    while time.time() < deadline - 10:
        # Perturb best_order
        perturbed = best_order[:]
        for _ in range(10):
            i = random.randint(0, num_perms - 2)
            perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
        
        p_score = calc_score(perturbed)
        p_order, p_score = try_swaps(perturbed, p_score, max_passes=5)
        
        if p_score > best_score:
            best_score = p_score
            best_order = p_order

    # 4. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            # Suffix of word vs Prefix of block
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_superperm_str(n):
    """Generate the standard recursive superpermutation as a string."""
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

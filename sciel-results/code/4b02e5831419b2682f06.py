import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate the standard superpermutation for n-1 to get the blocks
    # Actually, we just need the permutations of 1..n-1
    # The blocks are of the form: perm + [n] + perm
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Create blocks
    # Block for permutation p is: p[0]...p[n-2] n p[0]...p[n-2]
    # As string for easy overlap? Or list of ints?
    # List of ints is better for memory and speed in Python if we use tuples for hashing.
    
    blocks = []
    for p in perms:
        block = list(p) + [n] + list(p)
        blocks.append(tuple(block))
        
    # Precompute overlaps
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Max possible overlap is 2*(n-1) = 12 for n=7.
    # Actually, the overlap cannot cross the 'n' in a way that creates invalid perms?
    # No, overlap is just string matching.
    
    # To speed up, we can store suffixes and prefixes.
    # But 720 is small.
    
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    
    # Symmetric? No, overlap(A,B) != overlap(B,A) generally.
    
    for i in range(num_perms):
        bi = blocks[i]
        for j in range(num_perms):
            bj = blocks[j]
            # Max overlap is len(bi) - 1
            # But bi ends with p_i, bj starts with p_j.
            # The overlap is between suffix of p_i and prefix of p_j.
            # Wait, bi is p_i + [n] + p_i.
            # Suffix of bi is suffix of p_i.
            # Prefix of bj is prefix of p_j.
            # So we only need to check suffix of p_i and prefix of p_j.
            # Length of p_i is n-1.
            # So max overlap is n-1.
            
            # Let's just check up to n-1
            max_ov = 0
            # We can optimize by checking from longest to shortest
            for k in range(n-1, 0, -1):
                if bi[-k:] == bj[:k]:
                    max_ov = k
                    break
            overlaps[i][j] = max_ov

    # Heuristic Search for Best Ordering
    # We want to maximize sum(overlaps[pi[k]][pi[k+1]])
    
    best_order = None
    best_score = -1
    
    # Initial order: standard order (index 0..N-1)
    # Score of standard order
    std_score = sum(overlaps[i][i+1] for i in range(num_perms-1))
    best_score = std_score
    best_order = list(range(num_perms))
    
    # Try to improve with Greedy + 2-opt
    
    def calc_score(order):
        s = 0
        for i in range(len(order)-1):
            s += overlaps[order[i]][order[i+1]]
        return s

    # Random restarts
    # Time budget for search: 10 seconds
    search_deadline = start_time + 10.0
    
    while time.time() < search_deadline:
        # Greedy construction
        current = [random.randint(0, num_perms-1)]
        remaining = set(range(num_perms)) - {current[0]}
        
        while remaining:
            last = current[-1]
            # Find neighbor in remaining with max overlap
            best_next = None
            best_ov = -1
            # Optimization: if time is tight, sample? No, 720 is small.
            for cand in remaining:
                ov = overlaps[last][cand]
                if ov > best_ov:
                    best_ov = ov
                    best_next = cand
            current.append(best_next)
            remaining.remove(best_next)
            
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(len(current)-1):
                for j in range(i+2, len(current)):
                    # Reverse current[i+1:j+1]
                    # Calculate delta
                    # Old: ... A B ... C D ...
                    # New: ... A C ... B D ...
                    # Where A=current[i], B=current[i+1], C=current[j], D=current[j+1]
                    # If j is last, D is 0 (no next)
                    
                    a_idx = current[i]
                    b_idx = current[i+1]
                    c_idx = current[j]
                    d_idx = current[j+1] if j+1 < len(current) else None
                    
                    old_ov = overlaps[a_idx][b_idx]
                    if d_idx is not None:
                        old_ov += overlaps[c_idx][d_idx]
                    
                    new_ov = overlaps[a_idx][c_idx]
                    if d_idx is not None:
                        new_ov += overlaps[b_idx][d_idx]
                    else:
                        # If j is last, the segment is at the end.
                        # Old: ... A B ... C
                        # New: ... A C ... B
                        # Overlap A-B is removed. Overlap C-D (none) is removed.
                        # New overlap A-C is added.
                        # Wait, if j is last, B...C is the tail.
                        # Old tail: B...C.
                        # New tail: C...B.
                        # Connection to A: A->B vs A->C.
                        # Internal overlaps in B...C are reversed?
                        # Overlap is not symmetric.
                        # So we must recalculate the whole segment score if we reverse.
                        # This makes 2-opt expensive if we don't precompute segment scores.
                        # Given 720 nodes, O(N^2) 2-opt steps is ~500k ops.
                        # Each op is O(1) if we just check the boundary, but internal overlaps change!
                        # So simple 2-opt with boundary check is INVALID for directed graphs.
                        # We must recalculate the score of the reversed segment.
                        # Length of segment can be up to 720.
                        # So one 2-opt pass is O(N^3) worst case?
                        # 720^3 is 370 million. Too slow in Python.
                        
                        # Alternative: Or-opt (move 1 or 2 elements).
                        # Or just accept that we might not do full 2-opt.
                        # Let's try a simpler local search: Swap adjacent elements.
                        pass
                    
                    # Since 2-opt is tricky with directed weights, let's stick to Greedy + Adjacent Swap (1-opt)
                    # Or just multiple Greedy restarts.
                    
        score = calc_score(current)
        if score > best_score:
            best_score = score
            best_order = current
            # If we found a very good one, maybe stop early?
            # Target overlap is 3455.
            # If best_score > 3450, we are close.

    # Construct the word using best_order
    if best_order is None:
        best_order = list(range(num_perms))
        
    word = []
    for i, idx in enumerate(best_order):
        b = blocks[idx]
        if i == 0:
            word.extend(b)
        else:
            # Find overlap with previous
            prev_b = blocks[best_order[i-1]]
            # We stored overlaps[prev_idx][idx]
            ov = overlaps[best_order[i-1]][idx]
            word.extend(b[ov:])
            
    # Convert to string
    return "".join(map(str, word))

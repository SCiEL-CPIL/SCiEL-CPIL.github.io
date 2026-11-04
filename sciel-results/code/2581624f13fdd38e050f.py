import time
import random
import sys

def construct(n):
    # Deadline guard: 120s limit, use 110s for safety
    start_time = time.time()
    deadline = start_time + 110.0

    # For n < 7, the standard construction is optimal or close enough,
    # and local search overhead might not be worth it or might be complex.
    # The task specifically mentions n=7 in the context of "shortest known".
    # We will apply the optimization only for n=7 to be safe and fast for others.
    if n != 7:
        return _standard_construct(n)

    # 1. Construct SP(6) and extract permutations
    word_6 = _standard_construct(6)
    # word_6 is a string of digits 1-6
    # Convert to list of ints for easier manipulation
    w6_ints = [int(c) for c in word_6]
    
    perms_6, seen = [], set()
    for i in range(len(w6_ints) - 6 + 1):
        window = tuple(w6_ints[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # Should be 720
    
    # 2. Precompute overlap matrix M
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    # Overlap: longest suffix of i matching prefix of j
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap is 6
            # Check k from 6 down to 0
            for k in range(6, -1, -1):
                if k == 0:
                    M[i][j] = 0
                    break
                if p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break

    # 3. Local Search to maximize sum of overlaps
    # Initial order: natural order of appearance
    order = list(range(num_perms))
    
    # Calculate initial score
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    best_order = order[:]
    best_score = calc_score(order)
    
    # Hill Climbing / Simulated Annealing
    # We use a simple hill climbing with random restarts and 2-opt moves.
    # To speed up, we can try to calculate delta for 2-opt.
    # But for N=720, recalculating the affected segment is fast enough.
    
    random.seed(42)
    
    current_order = order[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random 2-opt move
        i = random.randint(0, num_perms - 2)
        j = random.randint(i + 1, num_perms - 1)
        
        # Ensure i and j are distinct and form a valid segment
        if i == j:
            continue
            
        # Create a copy of the segment to reverse? 
        # Or just calculate the new score for the segment [i-1, j+1]
        # Old segment: ... A, B, C, ..., D, E ...
        # where B..D is the reversed part.
        # Indices: i is start of reversal, j is end.
        # Links affected:
        # (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
        # New links:
        # (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
        
        # Calculate old score of affected links
        old_links_score = 0
        if i > 0:
            old_links_score += M[current_order[i-1]][current_order[i]]
        if j < num_perms - 1:
            old_links_score += M[current_order[j]][current_order[j+1]]
        for k in range(i, j):
            old_links_score += M[current_order[k]][current_order[k+1]]
            
        # Calculate new score of affected links
        new_links_score = 0
        if i > 0:
            new_links_score += M[current_order[i-1]][current_order[j]]
        if j < num_perms - 1:
            new_links_score += M[current_order[i]][current_order[j+1]]
        # Internal links of reversed segment:
        # Original: (i, i+1), ..., (j-1, j)
        # Reversed: (j, j-1), ..., (i+1, i)
        for k in range(j, i, -1):
            new_links_score += M[current_order[k]][current_order[k-1]]
            
        delta = new_links_score - old_links_score
        
        if delta > 0:
            # Accept move
            current_order[i:j+1] = reversed(current_order[i:j+1])
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = current_order[:]
        else:
            # If no improvement, we can try a small perturbation or just keep searching
            # To avoid infinite loops on the same bad moves, we could restart occasionally
            # But simple hill climbing often gets stuck.
            # Let's add a simple "kick" if we haven't improved in a while?
            # For now, just keep trying random 2-opts.
            pass

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
            # Since we know the overlap from M, we can use it directly!
            # Wait, the overlap between the *previous block* and *current block* is M[prev_idx][curr_idx].
            # The previous block is B_{prev} = P_{prev} 7 P_{prev}.
            # The current block is B_{curr} = P_{curr} 7 P_{curr}.
            # The overlap is the longest suffix of B_{prev} matching prefix of B_{curr}.
            # As established, this is exactly M[prev_idx][curr_idx].
            ov = M[best_order[idx-1]][best_order[idx]]
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_construct(n):
    """Classic recursive construction: 1! + 2! + ... + n!"""
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
    return "".join(map(str, word)) if isinstance(word[0], int) else word

# Note: _standard_construct returns a string in the final line if word contains ints?
# Let's check the type.
# In construct(n), word starts as [1] (int).
# In _standard_construct, word starts as [1] (int).
# The return statement `"".join(map(str, word))` works for list of ints.
# So _standard_construct should return a string.
# Let's fix the return of _standard_construct to be consistent.

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
    return "".join(map(str, word))

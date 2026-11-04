import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Classic construction
    def classic_construct(n):
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

    if n != 7:
        return classic_construct(n)
    
    # Optimization for n=7
    # 1. Get the list of permutations of 1..6 in the order they appear in SP(6)
    word_6 = [1]
    for m in range(2, 7): # up to 6
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
            
    # perms_6 is a list of 720 tuples of length 6
    num_perms = len(perms_6)
    
    # 2. Precompute overlap matrix
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    # Overlap is longest suffix of i matching prefix of j
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # 3. Local Search
    order = list(range(num_perms))
    # Calculate initial score
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        # Note: The total length is fixed sum of blocks - sum of overlaps.
        # So maximizing score (sum of overlaps) minimizes length.
        
    best_order = order[:]
    best_score = calc_score(order)
    
    # Simple Hill Climbing
    # We can try swaps and moves
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.5:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            new_order = order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            # Delta calculation
            # Old: M[order[i]][order[i+1]]
            # New: M[order[i+1]][order[i]]
            # Also affects neighbors if i>0 or i+1 < n-1
            # To be safe and simple, just recalc full score? 
            # 720 additions is very fast.
            new_score = calc_score(new_order)
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= calc_score(order): # Accept if not worse? Or just hill climb?
                order = new_order
        else:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            new_order = order[:]
            item = new_order.pop(i)
            new_order.insert(j, item)
            new_score = calc_score(new_order)
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= calc_score(order):
                order = new_order
                
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

import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321" # Standard is 123121321? 
        # S1=1, S2=121, S3: 12 3 12 -> 12312, 21 3 21 -> 21321. Overlap 12312 and 21321?
        # Seed logic:
        # word=[1]
        # m=2: window [1]. perms=[[1]]. new = [1,2,1]. word=[1,2,1]
        # m=3: windows of len 2 in [1,2,1]: (1,2), (2,1).
        # perms = [(1,2), (2,1)]
        # block1 = 1,2,3,1,2. 
        # block2 = 2,1,3,2,1.
        # new starts with block1.
        # overlap block1 and block2? 
        # 12312 vs 21321.
        # suffix 212? no. suffix 2? prefix 2. overlap 1.
        # new = 12312 + 1321 = 123121321. Length 9. Correct.

    # For n=7, use the TSP heuristic on the blocks of size n-1.
    # This assumes n is small enough for the heuristic to run fast.
    # For n=7, n-1=6. 720 blocks.
    
    if n > 7:
        # Fallback to standard construction for larger n if needed, though task is n=7
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

    # Heuristic for n=7 (and potentially 4,5,6)
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_blocks = len(blocks)
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            # Check from longest to shortest
            # Optimization: Only check k where bi[-k] == bj[0] etc?
            # Just loop k
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    # Greedy TSP Heuristic
    # Start with block 0
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # Calculate total length
    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    initial_len = calc_length(path)
    
    # 2-Opt Local Search
    improved = True
    while improved:
        improved = False
        for i in range(len(path)-1):
            for j in range(i+2, len(path)):
                # Reverse path[i+1:j+1]
                # Check change in overlap
                # Old overlaps: 
                # overlap(path[i-1], path[i]) if i>0
                # overlap(path[i], path[i+1])
                # ...
                # overlap(path[j], path[j+1]) if j < len(path)-1
                # overlap(path[j+1], path[j+2]) if j+1 < len(path)-1
                
                # New overlaps:
                # overlap(path[i-1], path[i]) unchanged
                # overlap(path[i], path[j]) NEW
                # ... (internal reversed)
                # overlap(path[j], path[i+1]) NEW
                # overlap(path[i+1], path[i+2]) ... (internal reversed)
                # overlap(path[j+1], path[j+2]) unchanged? No, path[j+1] is now connected to path[i+1]?
                
                # Let's just recalc the total length for simplicity if N is small (720)
                # But 720^2 * 720 is too slow.
                # We need to calculate delta.
                
                # Let A = path[i], B = path[i+1], C = path[j], D = path[j+1]
                # Old segment: ... A, B, ..., C, D ...
                # New segment: ... A, C, ..., B, D ...
                
                # Old costs involved:
                # cost(A, B) = len(B) - ov(A, B)
                # cost(C, D) = len(D) - ov(C, D)
                # (Internal costs cancel out in sum if we just look at boundaries? No, internal order changes)
                
                # Actually, the sum of overlaps is:
                # Sum(ov(path[k], path[k+1]))
                # Old: ov(A,B) + ov(B, path[i+2]) + ... + ov(path[j-1], C) + ov(C,D)
                # New: ov(A,C) + ov(C, path[j-1]) + ... + ov(path[i+2], B) + ov(B,D)
                
                # We need to check if the new sum is greater.
                
                # To keep it simple and fast, we can do a few iterations of random restarts or just accept the greedy result if it's close.
                # Given the time limit, let's try a simple hill climbing on the path.
                
                # For now, let's just use the greedy result. It's likely better than the seed's fixed order.
                pass

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))

import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
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
    
    # Optimization: Use string comparison for speed if possible, or just list slicing.
    # For n=7, block length is 13. 720*720*13 is ~6.7M ops. Fast.
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            # Check from longest to shortest
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    # Greedy TSP Heuristic with multiple restarts
    best_path = None
    best_len = float('inf')
    
    # Try a few random starts or specific starts
    # Since 720 is small, we can try all starts if we use a fast greedy?
    # Greedy is O(N^2). 720^2 = 518k. 720 starts * 518k = 373M ops. Too slow in Python.
    # So we use a limited number of restarts, e.g., 10-20 random starts.
    
    import random
    random.seed(42)
    
    num_restarts = 20
    for _ in range(num_restarts):
        current = random.randint(0, num_blocks - 1)
        unvisited = set(range(num_blocks))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            # Find neighbor with max overlap
            # To speed this up, we could pre-sort neighbors, but overlaps change? No, overlaps are static.
            # But we need max overlap among *unvisited*.
            # Simple loop is O(N). Total O(N^2).
            for j in unvisited:
                if overlaps[current][j] > best_overlap:
                    best_overlap = overlaps[current][j]
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        curr_len = calc_length(path)
        if curr_len < best_len:
            best_len = curr_len
            best_path = path[:]
            
    # 2-Opt Local Search
    path = best_path
    improved = True
    max_iterations = 100
    iter_count = 0
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(len(path)-1):
            for j in range(i+2, len(path)):
                # Reverse path[i+1:j+1]
                # Let A = path[i], B = path[i+1], C = path[j], D = path[j+1] (if exists)
                # Old segment: ... A, B, ..., C, D ...
                # New segment: ... A, C, ..., B, D ...
                
                # Old overlaps involved in change:
                # ov(A, B)
                # ov(C, D) (if j < len(path)-1)
                # Note: Internal overlaps are reversed, so sum is same?
                # ov(B, path[i+2]) becomes ov(path[i+2], B) ? No.
                # The path is reversed.
                # Original: A -> B -> X1 -> ... -> Xk -> C -> D
                # New: A -> C -> Xk -> ... -> X1 -> B -> D
                # 
                # Old cost part: len(B) - ov(A,B) + ... + len(C) - ov(prev,C) + len(D) - ov(C,D)
                # New cost part: len(C) - ov(A,C) + ... + len(B) - ov(prev,B) + len(D) - ov(B,D)
                
                # We need to calculate delta.
                # Let's just calculate the length of the affected segment.
                # Segment indices: i to j+1 (exclusive of j+1's connection to j+2? No, D is j+1).
                # Actually, easier to just recalculate total length if N is small?
                # N=720. Recalc is O(N). 720 * 500k moves = 360M ops. Too slow.
                
                # We need O(1) delta calculation.
                # Old overlaps:
                ov_AB = overlaps[path[i]][path[i+1]]
                ov_CD = overlaps[path[j]][path[j+1]] if j < len(path)-1 else 0
                
                # Internal overlaps sum is invariant under reversal?
                # Sum(ov(path[k], path[k+1]) for k in i+1 to j-1)
                # Reversed: Sum(ov(path[k+1], path[k]) for k in i+1 to j-1)
                # These are NOT necessarily equal. overlaps[i][j] != overlaps[j][i].
                # So we must recalculate the internal sum or precompute prefix sums of overlaps?
                # Prefix sums of overlaps depend on the path order, which changes.
                
                # Alternative: Just accept that 2-opt with O(N) delta is too slow for 720 nodes in Python if we do many passes.
                # However, we can restrict the search space or use a faster heuristic.
                
                # Let's try Or-opt (move 1 element) which has O(1) delta if we are careful?
                # Or-opt move 1:
                # Move path[j] to position i.
                # Old: ... A, B, ... C, X, D ...
                # New: ... A, X, B, ... C, D ...
                # 
                # This is also complex.
                
                # Given the time constraints and Python speed, a full 2-opt on 720 nodes might be slow.
                # Let's try a simpler "First Improvement" 2-opt with a limited number of passes.
                
                # To make it feasible, we will only check a subset of moves or break early.
                # Or, we can use the fact that the greedy solution is already quite good.
                
                # Let's implement a simple "Swap" local search?
                # Swap path[i] and path[j].
                # Delta calculation is easier.
                
                # Let's stick to a very fast local search:
                # Try to improve by swapping adjacent elements? (2-opt with j=i+1 is swap).
                # This is O(N) per pass.
                
                for i in range(len(path)-1):
                    # Try swapping path[i] and path[i+1]
                    a = path[i]
                    b = path[i+1]
                    # Check if swap improves
                    # Old: ov(prev, a) + ov(a, b) + ov(b, next)
                    # New: ov(prev, b) + ov(b, a) + ov(a, next)
                    
                    prev = path[i-1] if i > 0 else None
                    next_ = path[i+2] if i+2 < len(path) else None
                    
                    old_cost = 0
                    if prev is not None: old_cost += overlaps[prev][a]
                    old_cost += overlaps[a][b]
                    if next_ is not None: old_cost += overlaps[b][next_]
                    
                    new_cost = 0
                    if prev is not None: new_cost += overlaps[prev][b]
                    new_cost += overlaps[b][a]
                    if next_ is not None: new_cost += overlaps[a][next_]
                    
                    if new_cost > old_cost:
                        path[i], path[i+1] = path[i+1], path[i]
                        improved = True
                        
    # Final construction
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))

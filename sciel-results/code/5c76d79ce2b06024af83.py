import time
import random
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _solve_csp_backtracking()
    else:
        return _construct_generic(q, n, radius)

def _solve_csp_backtracking():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Precompute all words as integers 0..728
    # Word w is represented as an integer in base 3
    # We need to cover all 729 words.
    
    # Precompute neighbors (ball of radius 1) for each word
    # neighbors[w] is a set of words within distance 1 of w
    neighbors = [None] * 729
    for w in range(729):
        word = []
        temp = w
        for _ in range(6):
            word.append(temp % 3)
            temp //= 3
        word.reverse()
        
        nbs = {w}
        for i in range(6):
            for val in range(3):
                if val == word[i]:
                    continue
                new_word = word.copy()
                new_word[i] = val
                new_w = 0
                for d in new_word:
                    new_w = new_w * 3 + d
                nbs.add(new_w)
        neighbors[w] = nbs
    
    # We want to find a set of codewords C such that union of neighbors[c] for c in C is all 729 words.
    # Best known size is 73. We aim for <= 73.
    
    # Initial heuristic: Greedy set cover to get a good starting point and upper bound
    uncovered = set(range(729))
    initial_code = []
    while uncovered:
        best_c = None
        max_cover = -1
        # To speed up, we don't check all 729 every time if we have a good heuristic, 
        # but 729 is small enough.
        for c in range(729):
            if c in initial_code: continue
            # Count how many uncovered words this covers
            # Optimization: Precompute intersection sizes? 
            # For now, simple loop. 729 * 729 is ~500k ops, fine.
            # But we can be smarter: only consider c that cover the "hardest" uncovered word.
            
            # For initial greedy, just pick max coverage
            cover_count = 0
            for u in uncovered:
                if u in neighbors[c]:
                    cover_count += 1
            if cover_count > max_cover:
                max_cover = cover_count
                best_c = c
        
        if best_c is None:
            break
            
        initial_code.append(best_c)
        # Remove covered words
        to_remove = set()
        for u in uncovered:
            if u in neighbors[best_c]:
                to_remove.add(u)
        uncovered -= to_remove
        
        if time.time() > deadline:
            break
            
    current_best_size = len(initial_code)
    current_best_code = initial_code.copy()
    
    if current_best_size <= 73:
        # If we already found a solution <= 73, we can try to improve or just return
        # But let's run the CSP search to see if we can find something smaller or confirm 73.
        # Actually, if we found 73, we can return it immediately if time is tight, 
        # but let's try to optimize further with CSP.
        pass

    # CSP Backtracking Search
    # State: 
    # - current_code: list of codewords selected so far
    # - uncovered: set of words not yet covered
    # - target_size: current best size found
    
    # Lower Bound Estimation:
    # The minimum number of additional codewords needed to cover 'uncovered' set.
    # Simple bound: Let M be the maximum number of uncovered words a single codeword can cover.
    # Then we need at least ceil(len(uncovered) / M) more codewords.
    # A tighter bound: Use a greedy packing lower bound.
    # But for speed, the simple bound is often enough with good variable ordering.
    
    # Variable Ordering:
    # Pick the codeword that covers the most uncovered words.
    # Or, pick the codeword that covers the "most constrained" uncovered word.
    # Let's stick to: Pick codeword c that maximizes |neighbors[c] ∩ uncovered|.
    
    global_best_size = current_best_size
    global_best_code = current_best_code
    
    def lower_bound(uncovered_set):
        if not uncovered_set:
            return 0
        # Max coverage of any single codeword for the remaining uncovered set
        # We can precompute this or compute on the fly.
        # On the fly:
        max_cov = 0
        # Optimization: We don't need to check all 729 codewords if we have a good heuristic.
        # But 729 is small.
        for c in range(729):
            # Count intersection
            # We can speed this up by checking if c's neighbors intersect uncovered
            # If we precompute neighbor lists as sets, intersection is fast.
            # neighbors[c] is a set.
            inter = neighbors[c] & uncovered_set
            if len(inter) > max_cov:
                max_cov = len(inter)
        if max_cov == 0:
            return float('inf') # Impossible to cover
        return (len(uncovered_set) + max_cov - 1) // max_cov
        
    def backtrack(current_code, uncovered_set, depth, max_depth):
        nonlocal global_best_size, global_best_code
        
        if time.time() > deadline:
            return False # Timeout, abort search
        
        if not uncovered_set:
            # Found a solution
            if len(current_code) < global_best_size:
                global_best_size = len(current_code)
                global_best_code = current_code.copy()
            return True # We found a valid solution, but we might want to keep searching for smaller?
                       # For minimization, if we found one, we can try to find smaller.
                       # But backtracking usually stops at first solution or explores all.
                       # Let's explore all branches that could lead to better solution.
                       # Actually, if we just want the minimum, we can prune branches 
                       # where len(current_code) + lower_bound(uncovered) >= global_best_size.
        
        # Pruning
        lb = lower_bound(uncovered_set)
        if len(current_code) + lb >= global_best_size:
            return False # Cannot improve on best found
        
        # Variable Ordering: Select next codeword
        # We want to pick a codeword that reduces the search space most effectively.
        # Heuristic: Pick the codeword that covers the most uncovered words.
        # Ties: Pick random to diversify? Or fixed.
        
        best_c = None
        max_cov = -1
        # To avoid iterating all 729 every time, we could maintain a list of candidate codewords.
        # But 729 is small.
        # Optimization: Only consider codewords that cover at least one uncovered word.
        # Even better: Pick the "hardest" uncovered word u (one with fewest covering codewords among remaining candidates).
        # Then pick a codeword that covers u and maximizes total coverage.
        
        # Let's try the "hardest word" strategy.
        # Find an uncovered word u that has the minimum number of potential covering codewords.
        # A codeword c can cover u if u is in neighbors[c].
        # Since we are selecting from all 729, every u can be covered by 1 + 6*2 = 13 codewords.
        # So the "hardness" is uniform in the full space.
        # However, as we select codewords, we don't remove candidates from the pool of 729, 
        # we just add to the code. The "remaining" choices are still all 729.
        # Wait, the CSP formulation:
        # We are choosing a subset S of {0..728}.
        # The order of selection doesn't matter for the final set, but it matters for the search path.
        # To avoid symmetry (permutations of the same set), we can enforce an ordering:
        # Only consider codewords c > last_selected_codeword.
        # This drastically reduces the search space!
        
        # Let's re-implement with ordering constraint.
        # State: last_selected, current_code, uncovered_set
        # Next choice: c > last_selected
        
        # If we use ordering, we must ensure we don't miss any solution.
        # Any set S can be represented by its sorted elements.
        # So we search for sorted sequences.
        
        # Let's restart the logic with ordering.
        
        return True # Placeholder, need to restructure

    # Restructured Backtracking with Ordering
    
    def backtrack_ordered(last_selected, current_code, uncovered_set):
        nonlocal global_best_size, global_best_code
        
        if time.time() > deadline:
            return
        
        if not uncovered_set:
            if len(current_code) < global_best_size:
                global_best_size = len(current_code)
                global_best_code = current_code.copy()
            return
        
        # Pruning
        lb = lower_bound(uncovered_set)
        if len(current_code) + lb >= global_best_size:
            return
        
        # Variable Ordering:
        # We must pick the next codeword c such that c > last_selected.
        # To guide the search, we should pick c that covers the most uncovered words.
        
        # Find the best c > last_selected
        best_c = -1
        max_cov = -1
        
        # Optimization: We can limit the search for c.
        # If we pick a c that doesn't cover any new uncovered words, it's useless (unless we are forced to, but we aren't).
        # So we only consider c where neighbors[c] & uncovered_set is not empty.
        
        # Iterate c from last_selected + 1 to 728
        for c in range(last_selected + 1, 729):
            # Check if c covers any uncovered words
            # If not, skip
            # To speed up, we can check if there's any u in uncovered such that u in neighbors[c]
            # This is still O(|uncovered|) per c.
            
            # Heuristic: If we have a "hard" uncovered word, prioritize c that covers it.
            # Let's just use max coverage for now.
            
            cov_count = 0
            # Fast check: if neighbors[c] is disjoint from uncovered_set, skip
            # Using set intersection
            inter = neighbors[c] & uncovered_set
            if not inter:
                continue
            cov_count = len(inter)
            
            if cov_count > max_cov:
                max_cov = cov_count
                best_c = c
                
        if best_c == -1:
            # No valid codeword found to cover remaining uncovered words
            return
        
        # Recursive step
        new_uncovered = uncovered_set - neighbors[best_c]
        current_code.append(best_c)
        backtrack_ordered(best_c, current_code, new_uncovered)
        current_code.pop()
        
        # Continue searching for other possibilities?
        # In standard backtracking for optimization, we would continue the loop to find other c's.
        # But we are iterating c in the loop above. 
        # Wait, the code above picks ONLY the best_c and recurses.
        # This is a Greedy Backtracking (Branch and Bound with greedy variable selection).
        # It might miss better solutions if the greedy choice is suboptimal.
        # To be more robust, we should iterate through ALL valid c's, not just the best one.
        # But that explodes the search space.
        # Compromise: Iterate through top K candidates.
        
    # Let's implement a version that iterates through top candidates.
    
    def backtrack_top_k(last_selected, current_code, uncovered_set, top_k=5):
        nonlocal global_best_size, global_best_code
        
        if time.time() > deadline:
            return
        
        if not uncovered_set:
            if len(current_code) < global_best_size:
                global_best_size = len(current_code)
                global_best_code = current_code.copy()
            return
        
        lb = lower_bound(uncovered_set)
        if len(current_code) + lb >= global_best_size:
            return
        
        # Find top K candidates c > last_selected
        candidates = []
        for c in range(last_selected + 1, 729):
            inter = neighbors[c] & uncovered_set
            if inter:
                candidates.append((len(inter), c))
        
        if not candidates:
            return
            
        # Sort by coverage descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take top K
        for _, c in candidates[:top_k]:
            new_uncovered = uncovered_set - neighbors[c]
            current_code.append(c)
            backtrack_top_k(c, current_code, new_uncovered, top_k)
            current_code.pop()
            
            if time.time() > deadline:
                return

    # Start backtracking
    # Initial uncovered is all words
    initial_uncovered = set(range(729))
    # We can start with last_selected = -1
    # To speed up, we can use the initial greedy code as a warm start for the bound,
    # but the search starts from scratch to ensure optimality (or near-optimality).
    # However, starting from scratch with top_k=5 might be slow.
    # Let's try top_k=10 initially.
    
    backtrack_top_k(-1, [], initial_uncovered, top_k=10)
    
    # If we didn't find anything better than initial, return initial
    if global_best_size <= len(initial_code):
        return [list(map(int, [divmod(global_best_code[i], 3**j)[1] for j in range(5, -1, -1)])) for i in range(len(global_best_code))]
    else:
        return [list(map(int, [divmod(initial_code[i], 3**j)[1] for j in range(5, -1, -1)])) for i in range(len(initial_code))]

def _construct_generic(q, n, radius):
    # Fallback generic greedy
    total = q ** n
    words = [list(map(int, itertools.product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    remaining = set(range(total))
    
    while remaining:
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code

# Helper to convert int to list of digits
def int_to_word(val, n, q):
    word = []
    for _ in range(n):
        word.append(val % q)
        val //= q
    return word[::-1]

import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Handle the specific task case: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1()
    
    # General fallback: greedy set cover
    total = q ** n
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Build coverage masks
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    coverage = []
    for i, w in enumerate(words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                idx = word_to_idx[tuple(nw)]
                mask |= (1 << idx)
        coverage.append(mask)
    
    start = time.time()
    deadline = start + 100.0
    
    # Greedy set cover
    uncovered = (1 << total) - 1
    code = []
    used = [False] * total
    while uncovered:
        if time.time() > deadline - 10:
            break
        best = -1
        best_cnt = 0
        for c in range(total):
            if used[c]:
                continue
            cnt = (coverage[c] & uncovered).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        if best == -1:
            break
        code.append(best)
        used[best] = True
        uncovered &= ~coverage[best]
        
    return [words[i] for i in code]


def _solve_ternary_7_1():
    """Solve K_3(7,1) using Branch and Bound with bit-parallelism."""
    N = 3 ** 7  # 2187
    start_time = time.time()
    # Allow 140s out of 150s limit
    deadline = start_time + 140.0
    
    # Precompute all words
    words = [list(w) for w in itertools.product(range(3), repeat=7)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute ball masks (Hamming balls of radius 1)
    # Each mask is an integer with N bits
    ball_masks = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(7):
            orig = w[pos]
            for nv in range(3):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                j = word_to_idx[tuple(nw)]
                mask |= (1 << j)
        ball_masks[i] = mask

    # 1. Get a good initial upper bound using a fast greedy heuristic
    # We run a few random greedy passes to get a tight UB
    best_ub = float('inf')
    
    def greedy_cover(uncovered_mask, randomize=False):
        if randomize:
            # Perturb the choice slightly
            pass
        uncovered = uncovered_mask
        count = 0
        while uncovered:
            best_c = -1
            best_cnt = 0
            # To speed up, we can just scan all N
            # Optimization: If we had a list of candidates, we'd use it. Here we scan all.
            for c in range(N):
                cnt = (ball_masks[c] & uncovered).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1:
                break
            uncovered &= ~ball_masks[best_c]
            count += 1
        return count

    # Run greedy a few times to get a solid UB
    # Since greedy is deterministic if we always pick the first max, 
    # we need randomization or different tie-breaking for different UBs.
    # However, for B&B, a single good UB is often enough.
    # Let's try a simple greedy.
    ub = greedy_cover((1 << N) - 1)
    best_ub = ub
    
    # Try to improve UB with a simple local search on the greedy result?
    # No, let's just run the B&B. The B&B will improve the UB as it finds solutions.
    
    # 2. Branch and Bound
    # State: uncovered_mask
    # We need to track the current path to reconstruct the solution, 
    # but for the score, we just need the size. 
    # However, the function must return the list of codewords.
    # So we need to store the best solution found.
    
    best_solution = []
    best_size = best_ub
    
    # Global variables for B&B
    # We use recursion. Python's recursion limit might be an issue if depth > 1000,
    # but max depth is ~186, so it's fine.
    
    # To make the B&B faster, we need a good lower bound.
    # LB: Greedy set cover size on the remaining uncovered set.
    # This is expensive to compute at every node.
    # Alternative LB: Max number of disjoint balls in the uncovered set?
    # Or simply: ceil(popcount(uncovered) / max_ball_size)?
    # max_ball_size is 22. 2187/22 = 99. This is weak.
    
    # A better LB: 
    # Pick an uncovered word u. Any solution must pick at least one codeword from ball(u).
    # This doesn't give a count LB directly.
    
    # Let's use the "Greedy LB" but computed efficiently.
    # Actually, a common technique is to use the "Maximal Independent Set" on the dual?
    # No, let's stick to a simple but effective LB:
    # LB = max(ceil(popcount(uncovered) / 22), greedy_lb(uncovered))
    # But greedy_lb is O(N^2) roughly.
    
    # Let's try a different approach:
    # Branch on the first uncovered word.
    # Order candidates by how many uncovered words they cover.
    
    # To avoid deep recursion overhead, we can implement it iteratively or just use recursion.
    # Recursion depth ~186 is safe.
    
    # Optimization: Precompute list of set bits for each mask? No, masks change.
    
    # Let's implement the B&B.
    
    # We need to be careful with time.
    # If we spend too much time in one branch, we might miss better ones.
    
    # Strategy:
    # 1. Find a word u in uncovered.
    # 2. Candidates are all c such that u is in ball(c). i.e., c is in ball(u).
    # 3. Sort candidates by coverage of uncovered (descending).
    # 4. Branch.
    
    # To speed up "sort candidates by coverage", we can just iterate and keep the best?
    # No, we need to branch on ALL of them.
    # But we can order them to find solutions faster.
    
    # We will use a global timer check.
    
    def bnb(uncovered_mask, depth, current_code):
        nonlocal best_size, best_solution, deadline
        
        if time.time() > deadline:
            return
        
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_solution = list(current_code)
            return
        
        # Lower Bound
        # Simple LB: ceil(popcount / 22)
        pop = uncovered_mask.bit_count()
        lb_simple = (pop + 21) // 22
        
        # Stronger LB: Greedy on uncovered
        # This is expensive. Let's only do it if depth + lb_simple < best_size
        # And maybe only at shallow depths?
        # Actually, let's try a "packing" LB.
        # Find a set of words in uncovered that are pairwise distance > 2.
        # Each such word requires a distinct codeword.
        # This is a Maximum Independent Set on the graph of uncovered words where edges are dist<=2.
        # Too expensive.
        
        # Let's use a heuristic LB:
        # Pick a word u in uncovered.
        # The next codeword must cover u.
        # Let C_u be the set of codewords covering u.
        # We must pick one from C_u.
        # This doesn't give a number > 1.
        
        # Let's just use the simple LB for now, and rely on the ordering to prune.
        # If depth + lb_simple >= best_size:
        #     return
        
        # A better LB is the greedy count, but computed fast.
        # Let's approximate greedy LB by picking the best cover, subtracting, and repeating?
        # That's the whole algorithm.
        
        # Let's try:
        # LB = 1 if uncovered else 0
        # Plus...
        
        # Actually, the "Greedy Lower Bound" is very effective for Set Cover.
        # Let's compute it but limit the iterations or use a faster method.
        # Given N=2187, and we are at depth ~100, uncovered is small.
        # Greedy on small uncovered is fast.
        
        # Let's compute a quick greedy LB
        lb_greedy = 0
        temp_unc = uncovered_mask
        while temp_unc:
            best_c = -1
            best_cnt = 0
            # Scan all N? 
            # If uncovered is small, we can scan only candidates?
            # No, we don't have a list of candidates for arbitrary masks.
            # Scanning 2187 items is fast in Python if optimized?
            # 2187 * bit_count is ~microseconds.
            # If we do this 100 times per node, it might be slow.
            # Let's only do it if pop > 100.
            if pop > 100:
                for c in range(N):
                    cnt = (ball_masks[c] & temp_unc).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                if best_c == -1: break
                temp_unc &= ~ball_masks[best_c]
                lb_greedy += 1
            else:
                # For small sets, just use pop/22
                lb_greedy += (temp_unc.bit_count() + 21) // 22
                break
                
        if depth + lb_greedy >= best_size:
            return
            
        # Branching
        # Pick the first set bit in uncovered_mask
        # lsb = uncovered_mask & -uncovered_mask
        # u = lsb.bit_length() - 1
        
        # Find candidates: all c such that u is in ball(c)
        # This is equivalent to c being in ball(u)
        # We can get this by iterating through bits of ball_masks[u]
        
        # Get list of candidates
        candidates = []
        m = ball_masks[u]
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            candidates.append(c)
            m ^= lsb
            
        # Sort candidates by how many uncovered words they cover (descending)
        # This helps find a good solution early
        candidates.sort(key=lambda c: (ball_masks[c] & uncovered_mask).bit_count(), reverse=True)
        
        for c in candidates:
            new_uncovered = uncovered_mask & ~ball_masks[c]
            current_code.append(c)
            bnb(new_uncovered, depth + 1, current_code)
            current_code.pop()
            
            # If we found a solution with size equal to the theoretical minimum (156?), we could stop.
            # But we don't know the minimum.
            # If best_size is updated, the pruning in the parent call will handle it.
            
            # Optional: If we found a solution, we might want to restart with the new UB?
            # The nonlocal best_size is updated, so the next sibling branch will prune better.

    # Start B&B
    # Initial uncovered is all 1s
    initial_uncovered = (1 << N) - 1
    
    # To improve performance, we can fix the first codeword.
    # Due to symmetry, we can assume the first codeword is 0000000.
    # Let's try that.
    # If we fix c=0, we remove its ball from uncovered.
    # This reduces the branching factor at the root.
    
    # Let's do a standard B&B first.
    # If it's too slow, we add symmetry breaking.
    
    # Symmetry breaking:
    # The group of symmetries acts on the coordinates and symbols.
    # We can force the first codeword to be lexicographically smallest in its orbit.
    # Or simply, since 0000000 is in every orbit? No.
    # But we can WLOG assume 0000000 is in the code?
    # Yes, because for any code C, we can apply a permutation of symbols and coordinates to map any codeword to 0000000?
    # No, we can map *one* codeword to 0000000.
    # So we can branch on the assumption that 0 is in the code.
    # If the optimal solution doesn't contain 0, there is an isomorphic one that does.
    # So we can force 0 to be in the code.
    
    # Let's force 0.
    code_with_0 = [0]
    uncovered_with_0 = initial_uncovered & ~ball_masks[0]
    
    # Run B&B starting with 0 included
    # Depth 1
    bnb(uncovered_with_0, 1, code_with_0)
    
    # If time is left, try other starting points?
    # No, the symmetry argument holds. Any optimal code can be transformed to one containing 0.
    # So the global optimum is found in this subtree.
    
    if not best_solution:
        # Fallback if B&B failed or didn't find anything (shouldn't happen)
        return [words[i] for i in range(best_ub)] # Just return something valid? No, must be valid.
        # Let's return the greedy result if B&B fails to produce a valid one (e.g. timeout before first leaf)
        # But B&B updates best_solution only when a leaf is reached.
        # If timeout happens before any leaf, best_solution is empty.
        # We should have a valid fallback.
        
        # Let's run a fast greedy to ensure we return a valid code.
        uncovered = initial_uncovered
        code = []
        while uncovered:
            best_c = -1
            best_cnt = 0
            for c in range(N):
                cnt = (ball_masks[c] & uncovered).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1: break
            code.append(best_c)
            uncovered &= ~ball_masks[best_c]
        return [words[i] for i in code]
        
    return [words[i] for i in best_solution]


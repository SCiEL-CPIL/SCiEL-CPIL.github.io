import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    # Precompute coverage counts for all balls to help with lower bounds and sorting
    ball_counts = [b.bit_count() for b in balls]
    
    # Initial Greedy to get a good upper bound
    def greedy_cover(uncovered_mask):
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    initial_selected = greedy_cover(all_ones_mask)
    best_size = len(initial_selected)
    best_code = [all_words[i] for i in initial_selected]
    
    # Branch and Bound for Set Cover
    # State: uncovered_mask, current_depth
    # Global: best_size, best_selection
    
    # Optimization: Remove redundant balls?
    # If ball[i] is a subset of ball[j], ball[i] is never better than ball[j] 
    # unless we are forced to pick i? No, in Set Cover, if A subset B, 
    # picking B covers everything A covers and more. So A is dominated.
    # We can filter out dominated balls to reduce branching factor.
    
    # However, checking all pairs is O(N^2) = ~4M operations, which is fine.
    # Let's create a list of non-dominated ball indices.
    
    # Actually, for radius 1 in 3-ary space, balls are likely similar in size (1 + 2n).
    # Dominance is rare but possible.
    
    # Let's stick to a simpler B&B with strong pruning.
    
    start_time = time.time()
    time_limit = 115.0
    
    # We use a recursive solver. Python recursion depth might be an issue if solution size is large (~186).
    # We can increase recursion limit or use iterative stack.
    import sys
    sys.setrecursionlimit(10000)
    
    # To speed up, we pre-compute for each word index, which balls cover it.
    # This helps in "Most Constrained Variable" selection.
    # covered_by[word_idx] = list of ball_indices that cover word_idx
    # This is expensive to build (N * N / word_size), but N=2187.
    # Let's just iterate through balls when needed, or use bitsets.
    
    # Lower Bound: 
    # 1. Uncovered count / Max ball size (1+2n = 15 for n=7, q=3)
    # 2. Packing bound: How many disjoint uncovered regions? Hard to compute.
    # Let's use: ceil(uncovered_count / max_possible_cover)
    # Max possible cover for a ball is 15.
    max_ball_size = 1 + 2 * n * (q - 1)
    
    def solve_bnb(uncovered_mask, current_depth, current_selection):
        nonlocal best_size, best_code
        
        if time.time() - start_time > time_limit:
            return
        
        if uncovered_mask == 0:
            if current_depth < best_size:
                best_size = current_depth
                best_code = current_selection[:]
            return
            
        # Lower bound
        # Simple bound: ceil(uncovered / max_ball_size)
        # Better bound: Sum of (1 / size of covering sets) for each uncovered element? (LP relaxation)
        # Let's stick to simple bound first, add LP if needed.
        unc_count = uncovered_mask.bit_count()
        # If even the largest possible ball can't cover enough, we need at least this many
        lb = (unc_count + max_ball_size - 1) // max_ball_size
        
        if current_depth + lb >= best_size:
            return
            
        # Find an uncovered element with the fewest candidate covering balls (Most Constrained)
        # Iterate through set bits in uncovered_mask
        # To avoid iterating all 2187 bits, we can just pick the first one? 
        # Picking the first one is fast but might not be optimal for branching.
        # Let's find the element with minimum coverage options among uncovered.
        
        # Optimization: If we just pick the first uncovered element, it's O(1) to find.
        # Let's try that first. If it's too slow, we'll optimize.
        
        # Get first uncovered bit
        # lsb = uncovered_mask & -uncovered_mask
        # elem_idx = lsb.bit_length() - 1
        
        # However, branching on *all* balls that cover this element is huge (up to 15 balls cover a point? No, 1 + 2n = 15 balls cover a specific point in radius 1? 
        # Wait. A point is covered by itself and its neighbors. 
        # Number of balls covering a specific word W:
        # The balls are centered at words. 
        # Word W is in Ball(X) if dist(W,X) <= 1.
        # So X can be W or a neighbor of W.
        # There are 1 + 2n = 15 such X's.
        # So branching factor is at most 15! This is very small.
        
        # So:
        # 1. Pick an uncovered element `e`.
        # 2. Find all centers `c` such that `balls[c]` has bit `e` set.
        # 3. Branch on each `c`.
        
        # To find centers covering `e`:
        # We can precompute `coverers[e]` = list of indices `i` such that `balls[i]` covers `e`.
        
        # Let's precompute coverers once.
        # coverers = [[] for _ in range(total_words)]
        # for i in range(total_words):
        #     mask = balls[i]
        #     while mask:
        #         lsb = mask & -mask
        #         idx = lsb.bit_length() - 1
        #         coverers[idx].append(i)
        #         mask ^= lsb
        
        # This is efficient.
        
        # Let's do the precomputation here to keep the function clean, or outside?
        # It's better outside or in a closure.
        
        # We'll define coverers outside the recursive function.
        
        # Pick the uncovered element with the *fewest* coverers to minimize branching.
        # We iterate through uncovered bits.
        # Since branching factor is small (<=15), even picking the first one is okay.
        # But picking the most constrained one is better for B&B.
        
        # Let's find the min coverers count among uncovered.
        # This requires iterating all uncovered bits.
        
        min_cover_count = float('inf')
        best_elem = -1
        temp_mask = uncovered_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            cnt = len(coverers[idx])
            if cnt < min_cover_count:
                min_cover_count = cnt
                best_elem = idx
            temp_mask ^= lsb
            
        if best_elem == -1:
            return # Should not happen if uncovered_mask != 0
            
        candidates = coverers[best_elem]
        
        # Sort candidates by how much they cover of the remaining uncovered (Greedy order)
        # This helps find a good solution early, tightening the bound.
        candidates.sort(key=lambda c: (balls[c] & uncovered_mask).bit_count(), reverse=True)
        
        for c in candidates:
            new_uncovered = uncovered_mask & ~balls[c]
            current_selection.append(c)
            solve_bnb(new_uncovered, current_depth + 1, current_selection)
            current_selection.pop()
            
            # Pruning: if we found a solution of size current_depth+1, 
            # and best_size is updated, we might prune further?
            # The check at the start of the function handles this.

    # Precompute coverers
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = balls[i]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            coverers[idx].append(i)
            mask ^= lsb

    # Start B&B
    # Use the greedy solution as the initial best
    current_selection = []
    solve_bnb(all_ones_mask, 0, current_selection)
    
    return best_code

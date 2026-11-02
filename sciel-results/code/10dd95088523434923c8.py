import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: all words (worst case) or greedy
        # Let's do a quick greedy for generality
        total = q ** n
        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17
    
    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)
    
    # Precompute inverse: for each y, list of x such that y in balls[x]
    # This is the same as balls[y] due to symmetry
    # So inv[y] = balls[y]
    
    # Current best solution
    best_size = total
    best_solution = None
    
    # 1. Greedy to get initial upper bound
    covered = np.zeros(total, dtype=bool)
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    greedy_selected = []
    
    while True:
        if time.time() > deadline - 30:
            break
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
        greedy_selected.append(max_idx)
        for y in balls[max_idx]:
            if not covered[y]:
                covered[y] = True
                for i in range(total):
                    if y in balls[i]:
                        counts[i] -= 1
        if np.all(covered):
            break
    
    if len(greedy_selected) < best_size:
        best_size = len(greedy_selected)
        best_solution = greedy_selected[:]
    
    # 2. Local Search on greedy solution to improve further
    current_set = set(greedy_selected)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
    
    def is_redundant(c):
        for y in balls[c]:
            if cover_count[y] == 1:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_set:
                continue
            if is_redundant(c):
                for y in balls[c]:
                    cover_count[y] -= 1
                current_set.remove(c)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # 3. Try a few random restarts with different greedy heuristics
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        # Weighted greedy: randomly perturb the counts
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        # Add random noise to break ties
        noise = np.random.uniform(0, 1, size=total)
        counts_noisy = counts.astype(float) + noise * 0.1
        
        covered = np.zeros(total, dtype=bool)
        selected = []
        
        while True:
            if time.time() > deadline - 10:
                break
            max_idx = np.argmax(counts_noisy)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
                            counts_noisy[i] -= 1  # Keep noise consistent? No, just recompute
            if np.all(covered):
                break
        
        # Local search on this new solution
        current_set = set(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                if is_redundant(c):
                    for y in balls[c]:
                        cover_count[y] -= 1
                    current_set.remove(c)
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_solution = list(current_set)
    
    # 4. If we have time, try a more aggressive local search with swaps
    if best_solution is not None and time.time() < deadline - 5:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        # Try to remove any redundant
        for c in list(current_set):
            if is_redundant(c):
                for y in balls[c]:
                    cover_count[y] -= 1
                current_set.remove(c)
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # 5. Partition-based recursive construction with cross-subcube pruning
    # Decompose 8D space into 3^4=81 subcubes by fixing first 4 coordinates
    # For each subcube, solve a 4D covering problem, then prune globally
    
    if time.time() < deadline - 30:
        # Precompute the 4D ternary space (3^4 = 81 points)
        dim4 = 4
        q4 = 3
        total4 = q4 ** dim4  # 81
        
        # For each of the 81 subcubes (indexed by first 4 digits),
        # find a minimal covering of the 4D subcube with radius-1 balls
        # restricted to the subcube. But we need to consider that balls
        # can extend outside the subcube, so we use a global approach.
        
        # Alternative: use the known optimal 4D ternary covering code structure
        # K_3(4,1) = 9 (the tetracode)
        
        # The tetracode: 9 codewords in {0,1,2}^4 that cover all 81 words
        # Let's construct it using a greedy approach on the 4D space
        
        # Precompute 4D balls
        balls4 = []
        for x in range(total4):
            ball = [x]
            digits = []
            tmp = x
            for i in range(dim4):
                d = tmp % q4
                digits.append(d)
                tmp //= q4
            for i in range(dim4):
                old_d = digits[i]
                mult = q4 ** i
                for new_d in range(q4):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls4.append(ball)
        
        # Greedy for 4D
        covered4 = np.zeros(total4, dtype=bool)
        counts4 = np.array([len(b) for b in balls4], dtype=np.int32)
        selected4 = []
        
        while True:
            if time.time() > deadline - 20:
                break
            max_idx = np.argmax(counts4)
            if counts4[max_idx] == 0:
                break
            selected4.append(max_idx)
            for y in balls4[max_idx]:
                if not covered4[y]:
                    covered4[y] = True
                    for i in range(total4):
                        if y in balls4[i]:
                            counts4[i] -= 1
            if np.all(covered4):
                break
        
        # Now, for each of the 81 subcubes, place the 4D code shifted into the subcube
        # Subcube s (0..80) corresponds to first 4 digits being the digits of s
        # The remaining 4 digits vary over all 81 possibilities
        
        # For a subcube with prefix s (4 digits), the codewords in that subcube
        # are: prefix_s + suffix_t where t is in selected4
        # But we need to map this to the 8D space
        
        # Actually, let's think differently. We'll create a candidate set by
        # taking, for each of the 81 subcubes, the 9 codewords from the 4D code,
        # but we'll use different 4D codes for different subcubes to get diversity.
        
        # First, let's generate multiple 4D codes via local search
        codes4 = []
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            covered4 = np.zeros(total4, dtype=bool)
            counts4 = np.array([len(b) for b in balls4], dtype=np.int32)
            noise = np.random.uniform(0, 1, size=total4)
            counts4_noisy = counts4.astype(float) + noise * 0.1
            selected4 = []
            
            while True:
                if time.time() > deadline - 20:
                    break
                max_idx = np.argmax(counts4_noisy)
                if counts4[max_idx] == 0:
                    break
                selected4.append(max_idx)
                for y in balls4[max_idx]:
                    if not covered4[y]:
                        covered4[y] = True
                        for i in range(total4):
                            if y in balls4[i]:
                                counts4[i] -= 1
                                counts4_noisy[i] -= 1
                if np.all(covered4):
                    break
            
            # Local search on 4D
            current4 = set(selected4)
            cover_count4 = np.zeros(total4, dtype=np.int32)
            for c in current4:
                for y in balls4[c]:
                    cover_count4[y] += 1
            
            improved4 = True
            while improved4 and time.time() < deadline - 20:
                improved4 = False
                candidates4 = list(current4)
                random.shuffle(candidates4)
                for c in candidates4:
                    if c not in current4:
                        continue
                    if all(cover_count4[y] > 1 for y in balls4[c]):
                        for y in balls4[c]:
                            cover_count4[y] -= 1
                        current4.remove(c)
                        improved4 = True
            
            codes4.append(list(current4))
        
        # Now build 8D codes by combining 4D codes
        # For each combination of 4D codes assigned to subcubes, build an 8D code
        # But 81 subcubes x 9 codewords = 729 codewords, which is too many.
        
        # Better approach: use the 4D structure as a guide for the 8D greedy
        # Weight the greedy based on which subcube a word belongs to
        
        if time.time() < deadline - 20:
            # Build a weighted greedy where the weight depends on the subcube
            # and the 4D position within the subcube
            
            # For each word x in 8D, compute:
            # - its subcube index (first 4 digits)
            # - its position within the subcube (last 4 digits)
            
            subcube_idx = np.zeros(total, dtype=np.int32)
            pos_in_subcube = np.zeros(total, dtype=np.int32)
            
            for x in range(total):
                digits = []
                tmp = x
                for i in range(8):
                    d = tmp % 3
                    digits.append(d)
                    tmp //= 3
                # First 4 digits (least significant in our representation)
                # Wait, our representation has digit 0 as least significant
                # So "first 4 coordinates" in the problem might be ambiguous.
                # Let's define subcube by the first 4 digits in our integer representation
                # i.e., digits 0-3 (least significant)
                sub = digits[0] + digits[1]*3 + digits[2]*9 + digits[3]*27
                pos = digits[4] + digits[5]*3 + digits[6]*9 + digits[7]*27
                subcube_idx[x] = sub
                pos_in_subcube[x] = pos
            
            # For each subcube, compute the 4D coverage demand
            # We want to prioritize codewords that cover uncovered positions
            # in their subcube according to the 4D code structure
            
            # Use the best 4D code as a template
            best_code4 = min(codes4, key=len) if codes4 else selected4
            
            # Build a priority score for each word based on how well it matches
            # the 4D code structure for its subcube
            
            # For each subcube s, the ideal codewords are those whose position
            # in the subcube is in best_code4
            
            priority = np.zeros(total, dtype=np.float64)
            for x in range(total):
                s = subcube_idx[x]
                p = pos_in_subcube[x]
                if p in best_code4:
                    priority[x] = 1.0
                else:
                    # Compute distance to nearest point in best_code4
                    min_dist = 4
                    for c in best_code4:
                        dist = 0
                        tmp1 = p
                        tmp2 = c
                        for i in range(4):
                            if (tmp1 % 3) != (tmp2 % 3):
                                dist += 1
                            tmp1 //= 3
                            tmp2 //= 3
                        min_dist = min(min_dist, dist)
                    priority[x] = 1.0 / (1.0 + min_dist)
            
            # Now do a greedy with this priority
            covered = np.zeros(total, dtype=bool)
            counts = np.array([len(b) for b in balls], dtype=np.int32)
            selected = []
            
            while True:
                if time.time() > deadline - 10:
                    break
                # Score = counts * priority
                scores = counts.astype(float) * priority
                max_idx = np.argmax(scores)
                if counts[max_idx] == 0:
                    break
                selected.append(max_idx)
                for y in balls[max_idx]:
                    if not covered[y]:
                        covered[y] = True
                        for i in range(total):
                            if y in balls[i]:
                                counts[i] -= 1
                if np.all(covered):
                    break
            
            # Local search on this solution
            current_set = set(selected)
            cover_count = np.zeros(total, dtype=np.int32)
            for c in current_set:
                for y in balls[c]:
                    cover_count[y] += 1
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                candidates = list(current_set)
                random.shuffle(candidates)
                for c in candidates:
                    if c not in current_set:
                        continue
                    if is_redundant(c):
                        for y in balls[c]:
                            cover_count[y] -= 1
                        current_set.remove(c)
                        improved = True
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_solution = list(current_set)
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        # Fallback: return all words (invalid for score but valid structure)
        # Actually, we must return a valid cover.
        # If best_solution is None, use the greedy one from step 1
        best_solution = greedy_selected
    
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result

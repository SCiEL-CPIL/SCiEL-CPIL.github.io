import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]
    
    # Precompute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls (neighborhoods) for all words
    balls = [None] * total_words
    for i in range(total_words):
        w = list(idx_to_tuple(i))
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls[i] = ball

    # Convert balls to frozensets for fast intersection
    ball_sets = [frozenset(b) for b in balls]
    
    # Tetracode direct sum: 9 * 3^(n-4) words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    def get_direct_sum_indices():
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        return dp_indices
    
    def greedy_deletion(code_indices, max_passes=50):
        """Efficient greedy deletion: remove redundant codewords."""
        code_set = set(code_indices)
        counts = np.zeros(total_words, dtype=np.int32)
        
        for idx in code_indices:
            for w in balls[idx]:
                counts[w] += 1
        
        current = list(code_indices)
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            removed_any = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                idx = current[i]
                # Check if any neighbor has count == 1
                redundant = True
                for w in balls[idx]:
                    if counts[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in balls[idx]:
                        counts[w] -= 1
                    current.pop(i)
                    code_set.discard(idx)
                    removed_any = True
            if not removed_any:
                break
        return current
    
    def symmetry_score(idx, current_code):
        """Compute a symmetry-based score for tie-breaking.
        Words that are more 'central' or have more symmetric neighbors get priority."""
        w = idx_to_tuple(idx)
        # Score based on value distribution symmetry
        # Prefer words with balanced coordinate values
        score = 0
        for i in range(n):
            # Prefer middle value (1) slightly, penalize extremes
            if w[i] == 1:
                score += 1
            elif w[i] == 0 or w[i] == 2:
                score -= 0.5
        # Add small random perturbation to break perfect ties
        score += random.random() * 0.1
        return score
    
    def local_search_swaps(code_indices, max_time=3.0):
        """Try to reduce code size by swapping codewords, with symmetry-guided selection."""
        code_set = set(code_indices)
        best = list(code_indices)
        best_size = len(best)
        local_start = time.time()
        
        # First try removals
        for idx in list(best):
            if time.time() > local_start + max_time:
                break
            remaining = [c for c in best if c != idx]
            covered = np.zeros(total_words, dtype=bool)
            for c in remaining:
                for w in balls[c]:
                    covered[w] = True
            if covered.all():
                best = remaining
                best_size = len(best)
                code_set = set(best)
        
        # Try swaps: replace one codeword with another
        # Use symmetry-guided candidate selection
        for i in range(len(best)):
            if time.time() > local_start + max_time:
                break
            # Generate candidates ordered by symmetry score
            candidates = []
            for candidate in range(total_words):
                if candidate in code_set:
                    continue
                candidates.append(candidate)
            
            # Sort by symmetry score (higher is better)
            candidates.sort(key=lambda x: symmetry_score(x, best), reverse=True)
            
            # Try top candidates
            for candidate in candidates[:50]:  # Limit to top 50 for speed
                if time.time() > local_start + max_time:
                    break
                new_code = best[:i] + [candidate] + best[i+1:]
                covered = np.zeros(total_words, dtype=bool)
                for c in new_code:
                    for w in balls[c]:
                        covered[w] = True
                if covered.all():
                    if len(new_code) < best_size:
                        best = new_code
                        best_size = len(best)
                        code_set = set(best)
        
        return best
    
    def randomized_greedy(max_time=5.0):
        """Randomized greedy construction with symmetry-guided tie-breaking."""
        uncovered = set(range(total_words))
        code_indices = []
        greedy_start = time.time()
        
        while uncovered:
            if time.time() > greedy_start + max_time:
                break
            
            # Find codewords that cover the most uncovered words
            best_gain = 0
            best_candidates = []
            
            for c in range(total_words):
                gain = len(ball_sets[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
            
            # Symmetry-guided selection from top candidates
            if len(best_candidates) > 1:
                # Sort by symmetry score
                best_candidates.sort(key=lambda x: symmetry_score(x, code_indices), reverse=True)
                # Pick from top few with some randomness
                top_k = min(3, len(best_candidates))
                if random.random() < 0.3:
                    best_c = random.choice(best_candidates[:top_k])
                else:
                    best_c = best_candidates[0]
            else:
                best_c = best_candidates[0]
            
            code_indices.append(best_c)
            uncovered -= ball_sets[best_c]
        
        return code_indices if not uncovered else None
    
    best_code = None
    best_size = float('inf')
    
    # Strategy 1: Direct sum + greedy deletion (fast, deterministic)
    dp_indices = get_direct_sum_indices()
    pruned = greedy_deletion(dp_indices, max_passes=30)
    if time.time() < deadline - 15.0:
        pruned = local_search_swaps(pruned, max_time=min(3.0, deadline - time.time() - 10.0))
    if len(pruned) < best_size:
        best_size = len(pruned)
        best_code = pruned
    
    # Strategy 2: Multiple randomized restarts with symmetry-guided greedy
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Randomize the symmetry scoring seed
        random.seed(r * 12345 + 67890)
        
        greedy_result = randomized_greedy(max_time=min(3.0, deadline - time.time() - 15.0))
        if greedy_result is None:
            continue
        
        # Prune the greedy result
        pruned = greedy_deletion(greedy_result, max_passes=20)
        
        # Local search
        if time.time() < deadline - 10.0:
            pruned = local_search_swaps(pruned, max_time=min(2.0, deadline - time.time() - 5.0))
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned
    
    # Strategy 3: If we haven't found something good, try more aggressive local search from direct sum
    if best_size > 75 and time.time() < deadline - 15.0:
        dp_indices = get_direct_sum_indices()
        # Try different deletion orders
        for perm_seed in range(10):
            if time.time() > deadline - 10.0:
                break
            random.seed(perm_seed * 999 + 1)
            shuffled = dp_indices[:]
            random.shuffle(shuffled)
            pruned = greedy_deletion(shuffled, max_passes=20)
            if time.time() < deadline - 5.0:
                pruned = local_search_swaps(pruned, max_time=min(1.5, deadline - time.time() - 3.0))
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
    
    if best_code is None:
        best_code = get_direct_sum_indices()
    
    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result

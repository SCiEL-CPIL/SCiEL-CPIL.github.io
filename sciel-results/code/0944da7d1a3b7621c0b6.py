import random
import time

def construct(q, n, radius):
    # This implementation assumes the specific case q=3, n=6, radius=1
    # which is the target of the task.
    # For generality, we fall back to a simple method if parameters differ.
    
    if q != 3 or n != 6 or radius != 1:
        return _construct_fallback(q, n, radius)

    total_words = q ** n # 729
    indices = range(total_words)
    
    def tuple_to_index(t):
        idx = 0
        for digit in t:
            idx = idx * q + digit
        return idx

    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute ball masks for all 729 words
    ball_masks = [0] * total_words
    for i in indices:
        w = index_to_tuple(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = tuple_to_index(tuple(new_w))
                    mask |= (1 << j)
        ball_masks[i] = mask

    all_bits = (1 << total_words) - 1
    
    def greedy_cover():
        uncovered = all_bits
        codewords = []
        while uncovered:
            max_cov = -1
            best_idx = -1
            # Vectorized-like search for max coverage
            # We iterate through all candidates
            for i in indices:
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
                elif cov == max_cov and best_idx != -1:
                    # Tie-breaking: prefer lower index for determinism/consistency
                    if i < best_idx:
                        best_idx = i
            if best_idx == -1:
                break
            codewords.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return codewords

    def local_search(codewords):
        improved = True
        while improved:
            improved = False
            # 1. Remove redundant
            for i in range(len(codewords)):
                # Check if codewords[i] is redundant
                union_others = 0
                for j in range(len(codewords)):
                    if j != i:
                        union_others |= ball_masks[codewords[j]]
                if (union_others & all_bits) == all_bits:
                    codewords.pop(i)
                    improved = True
                    break
            
            if len(codewords) == 0:
                break
                
            # 2. Try swaps to enable future removals or just maintain validity
            # We try to swap a codeword with another to see if it makes a different codeword redundant
            # This is O(N^2 * K) which is fine for K~80, N=729.
            # However, a simpler effective move is:
            # For each i, try to replace codewords[i] with a random non-codeword.
            # If the new set is valid, accept it.
            # To make it more directed, we can try to replace the codeword that is "most redundant"
            # i.e., its ball is almost covered by others.
            
            # Let's try a systematic swap search for the first few iterations of local search
            # If no removal was found, try to find a swap that results in a set where
            # at least one codeword becomes redundant.
            
            if not improved:
                # Try random swaps with a limit
                for _ in range(100):
                    i = random.randint(0, len(codewords) - 1)
                    # Pick a random candidate
                    cand = random.randint(0, total_words - 1)
                    if cand in codewords:
                        continue
                    
                    # Check if swapping codewords[i] with cand keeps the set valid
                    # Union of (codewords \ {codewords[i]}) U {cand}
                    union_new = ball_masks[cand]
                    for j in range(len(codewords)):
                        if j != i:
                            union_new |= ball_masks[codewords[j]]
                    
                    if (union_new & all_bits) == all_bits:
                        # Valid swap. Does it allow a removal?
                        # We don't know yet. Just accept to move in the state space.
                        codewords[i] = cand
                        improved = True
                        break
        return codewords

    best_codewords = greedy_cover()
    best_size = len(best_codewords)
    
    start_time = time.time()
    time_limit = 115
    
    # Iterated Local Search
    current = best_codewords[:]
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        # Local search
        improved_ls = local_search(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_codewords = current[:]
            
        # Perturbation: Remove a random codeword and re-greedy fill
        # This helps escape local minima
        if len(current) > 0:
            # Create a perturbed version
            perturbed = current[:]
            # Remove a random codeword
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx_to_remove)
            
            # Re-greedy fill to make it valid again
            # Calculate current coverage
            covered = 0
            for c in perturbed:
                covered |= ball_masks[c]
            uncovered = all_bits & ~covered
            
            # Greedy fill
            while uncovered:
                max_cov = -1
                best_idx = -1
                for i in indices:
                    # Only consider words not in perturbed? 
                    # It's safer to just pick the best covering word.
                    # If it's already in the set, it doesn't help.
                    if i in perturbed:
                        continue
                    cov = (uncovered & ball_masks[i]).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = i
                if best_idx == -1:
                    break
                perturbed.append(best_idx)
                uncovered &= ~ball_masks[best_idx]
            
            # If perturbed is valid (it is by construction), run local search on it?
            # To save time, we can just evaluate its size.
            # If it's smaller, update best.
            # But it's likely larger or equal.
            # The point is to use this as the *new* current for the next ILS iteration?
            # Yes.
            current = perturbed
        
        # Check if we found the theoretical lower bound or best known
        if best_size == 73:
            break

    result = [list(index_to_tuple(i)) for i in best_codewords]
    return result

def _construct_fallback(q, n, radius):
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1:
            break
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
    return [list(c) for c in codewords]

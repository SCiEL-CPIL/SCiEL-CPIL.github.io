import random
import time

def construct(q, n, radius):
    # This implementation uses a Local Search approach:
    # 1. Initial Greedy Set Cover
    # 2. Iterative removal of redundant codewords
    # 3. Kicks (remove random, greedy repair) to escape local minima
    
    if q != 3 or n != 6 or radius != 1:
        return _construct_fallback(q, n, radius)

    total_words = q ** n # 729
    
    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute balls as lists of indices
    # ball[i] contains all indices j such that dist(i, j) <= radius
    balls = []
    for i in range(total_words):
        w = index_to_tuple(i)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = 0
                    for digit in new_w:
                        j = j * q + digit
                    ball.append(j)
        balls.append(ball)

    # Initial Greedy Set Cover
    uncovered = [1] * total_words # 1 means uncovered
    cover_counts = [0] * total_words # Number of codewords covering each word
    codewords = []
    
    # We use a list for codewords to allow removal
    # To speed up greedy, we can just pick max coverage each time
    # Since N is small, O(N^2) greedy is fine for initialization
    
    remaining = set(range(total_words))
    while remaining:
        best_idx = -1
        max_cov = -1
        # Optimization: Only check candidates that cover at least one remaining word?
        # For 729 words, checking all is fast enough.
        for i in range(total_words):
            cov = sum(1 for j in balls[i] if j in remaining)
            if cov > max_cov:
                max_cov = cov
                best_idx = i
                if cov == len(remaining): # Perfect cover found
                    break
        if best_idx == -1:
            break
        codewords.append(best_idx)
        # Update remaining
        for j in balls[best_idx]:
            if j in remaining:
                remaining.remove(j)
                cover_counts[j] += 1 # Wait, this logic is flawed for later steps. 
                # Let's restart cover_counts calculation properly after init.
        
    # Recalculate cover_counts from scratch for the initial greedy solution
    cover_counts = [0] * total_words
    for c in codewords:
        for j in balls[c]:
            cover_counts[j] += 1

    best_size = len(codewords)
    best_codewords = codewords[:]
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for overhead
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try to remove redundant codewords
        # Iterate through codewords. If removing one doesn't leave any word with count 0, remove it.
        i = 0
        while i < len(codewords):
            c = codewords[i]
            # Check if removal is safe
            safe = True
            for j in balls[c]:
                if cover_counts[j] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove codeword c
                for j in balls[c]:
                    cover_counts[j] -= 1
                codewords.pop(i)
                improved = True
                # Do not increment i, check the new codeword at position i
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
            else:
                i += 1
        
        if not improved:
            # Phase 2: Kick to escape local minimum
            # Randomly remove a codeword and repair with greedy
            if len(codewords) > 0:
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_removed = codewords[idx_to_remove]
                
                # Tentatively remove
                for j in balls[c_removed]:
                    cover_counts[j] -= 1
                codewords.pop(idx_to_remove)
                
                # Find uncovered words
                uncovered_indices = [j for j in range(total_words) if cover_counts[j] == 0]
                
                if uncovered_indices:
                    # Greedily add codewords to cover uncovered_indices
                    # We want to minimize the number of added codewords
                    # Note: We might add the same codeword back, or a different one
                    # To ensure we don't just add back the same one if it's not optimal, 
                    # we can prioritize candidates that cover many uncovered words.
                    
                    # Simple greedy on the uncovered set
                    un_covered_set = set(uncovered_indices)
                    while un_covered_set:
                        best_new = -1
                        max_cov = -1
                        for k in range(total_words):
                            # Count how many words in un_covered_set are covered by k
                            cov = 0
                            for j in balls[k]:
                                if j in un_covered_set:
                                    cov += 1
                            if cov > max_cov:
                                max_cov = cov
                                best_new = k
                        if best_new == -1 or max_cov == 0:
                            break
                        # Add best_new
                        codewords.append(best_new)
                        for j in balls[best_new]:
                            cover_counts[j] += 1
                            if j in un_covered_set:
                                un_covered_set.remove(j)
                        
                        # If we added more codewords than we removed (1), size increased.
                        # But maybe this new configuration allows for more removals later.
                        
                # After kick, immediately try to remove redundancies again in the next loop iteration
                # We set improved = False so the while loop continues, 
                # but we should ensure we don't infinite loop without progress.
                # The removal phase will run again. If it removes nothing, we kick again.
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]

    # Convert best indices back to lists
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

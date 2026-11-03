import itertools

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as lists of indices
    balls = []
    for w in all_words:
        ball_indices = []
        # Self
        ball_indices.append(word_to_idx[w])
        
        if radius >= 1:
            w_list = list(w)
            for i in range(n):
                orig_val = w_list[i]
                for new_val in range(q):
                    if new_val != orig_val:
                        w_list[i] = new_val
                        neighbor = tuple(w_list)
                        ball_indices.append(word_to_idx[neighbor])
                w_list[i] = orig_val
        
        balls.append(ball_indices)
        
    # Greedy Set Cover
    # uncovered_count[i] = 1 if word i is not yet covered, 0 otherwise
    uncovered = [1] * N
    uncovered_set = set(range(N)) # For efficient iteration of uncovered words? 
    # Actually, for greedy, we just need to know if a word is uncovered.
    # We can maintain a list of uncovered indices to iterate? No, we iterate all words to find best.
    
    code_indices = []
    
    while any(uncovered):
        best_idx = -1
        max_new = -1
        for i in range(N):
            # Count how many words in balls[i] are currently uncovered
            new_covered = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    new_covered += 1
            if new_covered > max_new:
                max_new = new_covered
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        # Mark words covered by best_idx
        for idx in balls[best_idx]:
            uncovered[idx] = 0
            
    # Pruning with coverage counts
    # coverage_count[i] = number of codewords covering word i
    coverage_count = [0] * N
    for idx in code_indices:
        for w_idx in balls[idx]:
            coverage_count[w_idx] += 1
            
    improved = True
    while improved:
        improved = False
        for i in range(len(code_indices)):
            # Try removing code_indices[i]
            idx_to_remove = code_indices[i]
            
            # Check if any word becomes uncovered (count drops to 0)
            can_remove = True
            for w_idx in balls[idx_to_remove]:
                if coverage_count[w_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Actually remove it
                for w_idx in balls[idx_to_remove]:
                    coverage_count[w_idx] -= 1
                code_indices.pop(i)
                improved = True
                break # Restart loop
                
    return [list(all_words[idx]) for idx in code_indices]

import itertools
import time
import random

def construct(q, n, radius):
    # Optimized path for q=3, n=6, R=1
    if q == 3 and n == 6 and radius == 1:
        start_time = time.time()
        deadline = start_time + 110  # Safety margin
        
        total_words = 3 ** n
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute balls as lists of indices
        balls = []
        for i, w in enumerate(words):
            ball = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    ball.append(word_to_idx[new_w])
            balls.append(ball)
            
        best_solution = None
        best_size = total_words
        
        # Run multiple greedy attempts with random tie-breaking
        num_iterations = 100 # Increased iterations due to speed
        
        for _ in range(num_iterations):
            if time.time() > deadline:
                break
                
            uncovered_mask = [True] * total_words
            selected = []
            
            while any(uncovered_mask):
                best_gain = 0
                best_candidates = []
                
                for i in range(total_words):
                    gain = 0
                    for j in balls[i]:
                        if uncovered_mask[j]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
                        
                if not best_candidates:
                    break
                    
                chosen = random.choice(best_candidates)
                selected.append(chosen)
                
                for j in balls[chosen]:
                    uncovered_mask[j] = False
                        
                if time.time() > deadline:
                    break
            
            if not any(uncovered_mask):
                # Pruning phase
                changed = True
                while changed:
                    changed = False
                    if time.time() > deadline:
                        break
                    
                    # Compute coverage counts
                    cov_counts = [0] * total_words
                    for c in selected:
                        for j in balls[c]:
                            cov_counts[j] += 1
                            
                    # Try to remove each word
                    for idx, c in enumerate(selected):
                        critical = False
                        for j in balls[c]:
                            if cov_counts[j] == 1:
                                critical = True
                                break
                        if not critical:
                            selected.pop(idx)
                            changed = True
                            break # Restart pruning loop
                            
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected[:]
                    
        if best_solution is not None:
            return [list(words[i]) for i in best_solution]
        else:
            # Fallback to seed
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Generic fallback
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

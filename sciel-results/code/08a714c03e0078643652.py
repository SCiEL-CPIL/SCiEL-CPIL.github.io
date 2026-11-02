import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # Fallback for other parameters (simple greedy)
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple fallback, return empty or minimal
        return []
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _solve_ternary_8_1():
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Precompute balls efficiently using base-3 arithmetic
    # We don't need all_words array explicitly if we use indices, 
    # but we need to return vectors. 
    # Let's keep all_words for final conversion.
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute covers: list of codeword indices that cover each word
    # covers[w] is a list of c such that w is in balls[c]
    # Note: covers[w] is exactly balls[w] due to symmetry of Hamming distance
    # But let's verify or just use balls. 
    # Since distance is symmetric, w in balls[c] <=> c in balls[w].
    # So covers[w] == balls[w]. This is a huge simplification.
    
    def run_greedy_and_refine(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Greedy Initialization
        is_covered = np.zeros(total, dtype=bool)
        codewords = []
        
        # Optimization: Maintain a count of how many times each word is covered? 
        # No, just track uncovered.
        # To speed up greedy, we can use the fact that we only care about uncovered words.
        
        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break
            
            # Find word covering most uncovered
            # newly_covered = (~is_covered)[balls]
            # scores = newly_covered.sum(axis=1)
            # This is O(N * BallSize). N=6561, BallSize=17. ~110k ops per step.
            # Greedy takes ~400 steps. 400 * 110k = 44M ops. Fast enough in numpy.
            
            scores = (~is_covered)[balls].sum(axis=1)
            max_score = np.max(scores)
            
            if max_score == 0:
                break
            
            # Pick randomly among best to add diversity
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        if len(codewords) == 0:
            return [], 0
            
        # Local Search
        # cover_counts[w] = number of codewords covering w
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
            
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant codewords
            i = 0
            while i < len(codewords):
                c = codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
                    
            if not improved:
                # Phase 2: Swaps to enable removal (limited to avoid timeout)
                swap_found = False
                cw_set = set(codewords)
                for i in range(len(codewords)):
                    if swap_found or time.time() > deadline - 10:
                        break
                    c = codewords[i]
                    
                    unique_mask = (cover_counts == 1)
                    unique_in_ball = unique_mask[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    
                    if len(unique_words) == 0:
                        continue
                    
                    # Limit candidates to keep it fast
                    candidates = set(balls[unique_words[0]].tolist())
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates &= set(balls[w].tolist())
                        
                    if not candidates:
                        continue
                        
                    candidates -= cw_set
                    
                    if not candidates:
                        continue
                        
                    candidate_list = list(candidates)
                    random.shuffle(candidate_list)
                    
                    for b in candidate_list[:20]:  # Limit candidates tried
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        
                        if not np.any(cover_counts == 0):
                            removal_found = False
                            for j in range(len(codewords)):
                                if j == i:
                                    continue
                                c_j = codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                                    
                            if removal_found:
                                codewords[i] = b
                                cw_set.discard(c)
                                cw_set.add(b)
                                improved = True
                                swap_found = True
                                break
                            else:
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
                            
        return codewords, len(codewords)

    def simulated_annealing_polish(codewords, cover_counts, start_time, deadline):
        """SA-based polish: accept moves that don't violate coverage, prefer removals."""
        n_cw = len(codewords)
        current_size = n_cw
        best_size = n_cw
        best_codewords = codewords[:]
        best_cover = cover_counts.copy()
        cw_set = set(codewords)
        
        # Temperature schedule
        T0 = 2.0
        T_min = 0.005
        step = 0
        max_steps = 300000
        
        while step < max_steps and time.time() < deadline - 5:
            T = T0 * (T_min / T0) ** (step / max_steps)
            
            i = np.random.randint(n_cw)
            c = codewords[i]
            
            # Fast move type selection
            r = np.random.random()
            if r < 0.50:
                move_type = 0
            elif r < 0.85:
                move_type = 1
            else:
                move_type = 2
            
            if move_type == 0:
                cover_counts[balls[c]] -= 1
                uncovered = (cover_counts == 0).sum()
                
                if uncovered == 0:
                    codewords.pop(i)
                    cw_set.discard(c)
                    n_cw -= 1
                    current_size = n_cw
                    if current_size < best_size:
                        best_size = current_size
                        best_codewords = codewords[:]
                        best_cover = cover_counts.copy()
                else:
                    accept_prob = np.exp(-uncovered / T)
                    if np.random.random() < accept_prob:
                        codewords.pop(i)
                        cw_set.discard(c)
                        n_cw -= 1
                        current_size = n_cw
                    else:
                        cover_counts[balls[c]] += 1
            
            elif move_type == 1:
                b = np.random.randint(total)
                if b in cw_set:
                    step += 1
                    continue
                
                cover_counts[balls[c]] -= 1
                cover_counts[balls[b]] += 1
                uncovered = (cover_counts == 0).sum()
                
                if uncovered == 0:
                    codewords[i] = b
                    cw_set.discard(c)
                    cw_set.add(b)
                else:
                    accept_prob = np.exp(-uncovered / T)
                    if np.random.random() < accept_prob:
                        codewords[i] = b
                        cw_set.discard(c)
                        cw_set.add(b)
                    else:
                        cover_counts[balls[b]] -= 1
                        cover_counts[balls[c]] += 1
            
            else:
                b = np.random.randint(total)
                if b in cw_set:
                    step += 1
                    continue
                
                codewords.append(b)
                cw_set.add(b)
                n_cw += 1
                current_size = n_cw
                cover_counts[balls[b]] += 1
            
            step += 1
            
            # Periodic cleanup every 500 steps (faster)
            if step % 500 == 0:
                j = 0
                while j < n_cw:
                    c_j = codewords[j]
                    cover_counts[balls[c_j]] -= 1
                    if (cover_counts == 0).sum() == 0:
                        codewords.pop(j)
                        cw_set.discard(c_j)
                        n_cw -= 1
                        current_size = n_cw
                        if current_size < best_size:
                            best_size = current_size
                            best_codewords = codewords[:]
                            best_cover = cover_counts.copy()
                    else:
                        cover_counts[balls[c_j]] += 1
                        j += 1
        
        # Final aggressive removal pass
        j = 0
        while j < n_cw:
            c_j = codewords[j]
            cover_counts[balls[c_j]] -= 1
            if (cover_counts == 0).sum() == 0:
                codewords.pop(j)
                cw_set.discard(c_j)
                n_cw -= 1
                current_size = n_cw
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = codewords[:]
                    best_cover = cover_counts.copy()
            else:
                cover_counts[balls[c_j]] += 1
                j += 1
        
        return best_codewords, best_size, best_cover

    best_codewords = None
    best_size = total
    best_cover = None
    
    # Run multiple restarts with different seeds
    seed = 42
    while time.time() < deadline - 20:
        codewords, size = run_greedy_and_refine(seed)
        
        if size < best_size or best_codewords is None:
            # Rebuild cover_counts for SA
            cover_counts = np.zeros(total, dtype=np.int32)
            for c in codewords:
                cover_counts[balls[c]] += 1
            
            # Apply SA polish with time budget
            sa_deadline = min(deadline - 5, time.time() + 60)
            polished, polished_size, polished_cover = simulated_annealing_polish(
                codewords, cover_counts, time.time(), sa_deadline
            )
            
            if polished_size < best_size:
                best_size = polished_size
                best_codewords = polished[:]
                best_cover = polished_cover
            else:
                if best_codewords is None or size < best_size:
                    best_size = size
                    best_codewords = codewords[:]
                    best_cover = cover_counts
        
        seed += 1
        
        if best_size <= 486:
            break
            
    # Final polish on best solution if time remains
    if best_codewords is not None and time.time() < deadline - 10:
        final_cover = np.zeros(total, dtype=np.int32)
        for c in best_codewords:
            final_cover[balls[c]] += 1
        polished, polished_size, polished_cover = simulated_annealing_polish(
            best_codewords[:], final_cover, time.time(), deadline - 2
        )
        if polished_size < best_size:
            best_size = polished_size
            best_codewords = polished[:]
            best_cover = polished_cover
    
    if best_codewords is None:
        return []
        
    return [list(all_words[c]) for c in best_codewords]

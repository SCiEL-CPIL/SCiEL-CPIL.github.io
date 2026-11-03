<<<<<<< SEARCH
import time
import numpy as np
import random
=======
import time
import numpy as np
import random
from collections import Counter
>>>>>>> REPLACE
<<<<<<< SEARCH
    balls = np.array(balls_list, dtype=np.int32)

    # Precompute: for each center, which other centers share at least one neighbor
    # This helps with fast swap validation
    # ball_sets[i] = frozenset of balls[i] for O(1) subset checks
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]
=======
    balls = np.array(balls_list, dtype=np.int32)

    # Precompute: for each center, which other centers share at least one neighbor
    # This helps with fast swap validation
    # ball_sets[i] = frozenset of balls[i] for O(1) subset checks
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]
    
    # Precompute neighbor lists for faster candidate filtering
    # For each word, list of centers whose ball contains it
    word_to_centers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for w in balls[i]:
            word_to_centers[w].append(i)
    
    # Convert to numpy arrays for faster processing
    word_to_centers_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in word_to_centers]
>>>>>>> REPLACE
<<<<<<< SEARCH
    def targeted_local_search(code, cover_count, max_iter=200000):
        """
        Aggressive targeted local search:
        For each codeword, find candidates that cover its uniquely-covered words.
        Use precomputed ball_sets for fast subset checks.
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Shuffle code order
            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                # Words only covered by i
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Targeted search: find candidates that cover all of only_by_i
                # First, pre-filter: candidate j must have balls[j] ⊇ only_by_i
                # Use ball_sets for fast check
                found = False
                n_candidates = len(candidates)
                if n_candidates == 0:
                    break

                # Sample candidates, prioritizing those with high overlap
                # Compute a quick score: number of only_by_i words in balls[j]
                # For efficiency, sample a subset
                sample_size = min(200, n_candidates)
                sample_idx = np.random.choice(n_candidates, size=sample_size, replace=False)

                for si in sample_idx:
                    j = candidates[si]
                    if j in code_set:
                        continue
                    # Check if balls[j] covers all of only_by_i
                    # only_by_i is small (typically < 17), so set check is fast
                    j_set = ball_sets[j]
                    # Check subset: all(w in j_set for w in only_by_i)
                    valid = True
                    for w in only_by_i:
                        if w not in j_set:
                            valid = False
                            break
                    if valid:
                        # Valid swap!
                        code[ci] = j
                        code_set.discard(i)
                        code_set.add(j)
                        cover_count[i_ball] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        found = True
                        break

                if found:
                    break

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count
=======
    def targeted_local_search(code, cover_count, max_iter=200000):
        """
        Aggressive targeted local search:
        For each codeword, find candidates that cover its uniquely-covered words.
        Use precomputed ball_sets for fast subset checks.
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Shuffle code order
            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                # Words only covered by i
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Targeted search: find candidates that cover all of only_by_i
                # First, pre-filter: candidate j must have balls[j] ⊇ only_by_i
                # Use ball_sets for fast check
                found = False
                n_candidates = len(candidates)
                if n_candidates == 0:
                    break

                # Use word_to_centers to find candidates that cover all only_by_i words
                # Intersect the candidate lists for each word in only_by_i
                if len(only_by_i) > 0:
                    # Start with candidates for the first word
                    candidate_set = set(word_to_centers_np[only_by_i[0]].tolist())
                    # Intersect with candidates for subsequent words
                    for w in only_by_i[1:]:
                        candidate_set &= set(word_to_centers_np[w].tolist())
                        if not candidate_set:
                            break
                    
                    # Filter out existing codewords
                    candidate_list = [j for j in candidate_set if j not in code_set]
                else:
                    candidate_list = []

                if not candidate_list:
                    # Fallback to random sampling if no candidates found
                    sample_size = min(200, n_candidates)
                    sample_idx = np.random.choice(n_candidates, size=sample_size, replace=False)
                    for si in sample_idx:
                        j = candidates[si]
                        if j in code_set:
                            continue
                        j_set = ball_sets[j]
                        valid = True
                        for w in only_by_i:
                            if w not in j_set:
                                valid = False
                                break
                        if valid:
                            candidate_list.append(j)
                            break

                # Try candidates in random order
                random.shuffle(candidate_list)
                for j in candidate_list:
                    if j in code_set:
                        continue
                    # Check if balls[j] covers all of only_by_i
                    j_set = ball_sets[j]
                    valid = True
                    for w in only_by_i:
                        if w not in j_set:
                            valid = False
                            break
                    if valid:
                        # Valid swap!
                        code[ci] = j
                        code_set.discard(i)
                        code_set.add(j)
                        cover_count[i_ball] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        found = True
                        break

                if found:
                    break

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count
>>>>>>> REPLACE
<<<<<<< SEARCH
    def simulated_annealing(code, cover_count, start_time, deadline, max_iter=500000):
        """
        Simulated annealing with pair-swap and single-swap moves to escape local minima.
        Objective: minimize number of codewords (penalty for uncovered words).
        """
        rng = np.random.RandomState(42)
        
        # Current solution
        current_code = list(code)
        current_cc = cover_count.copy()
        current_size = len(current_code)
        
        # Track uncovered count
        uncovered_mask = current_cc == 0
        uncovered_count = int(uncovered_mask.sum())
        
        best_code = current_code[:]
        best_size = current_size
        best_uncovered = uncovered_count
        best_cc = current_cc.copy()
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        T = T_start
        
        code_set = set(current_code)
        all_centers = list(range(total_words))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Cool down
            progress = iteration / max_iter
            T = T_start * (T_end / T_start) ** progress
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Single swap: replace one codeword with another
                if len(current_code) == 0:
                    break
                i = rng.randint(len(current_code))
                old_c = current_code[i]
                
                # Find candidate: prefer centers that cover uniquely-covered words
                old_ball = balls[old_c]
                only_by_old = old_ball[current_cc[old_ball] == 1]
                
                if len(only_by_old) == 0:
                    # Can remove
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    # Update uncovered
                    newly_uncovered = old_ball[current_cc[old_ball] == 0]
                    uncovered_count += len(newly_uncovered)
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue
                else:
                    # Try to find a replacement
                    # Sample candidates
                    sample_size = min(100, total_words - len(code_set))
                    if sample_size <= 0:
                        break
                    candidates = [c for c in rng.choice(total_words, size=sample_size, replace=False)
                                  if c not in code_set]
                    
                    best_new_c = None
                    best_delta = float('inf')
                    for j in candidates:
                        j_ball = balls[j]
                        # Check if j covers all of only_by_old
                        valid = True
                        for w in only_by_old:
                            if w not in j_ball:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Compute delta in uncovered count
                        # Words that become uncovered: in old_ball but not in j_ball
                        # Words that become covered: in j_ball but not in old_ball
                        old_set = set(old_ball)
                        j_set = set(j_ball)
                        newly_uncovered = len(old_set - j_set)
                        newly_covered = len(j_set - old_set)
                        delta = newly_uncovered - newly_covered
                        if delta < best_delta:
                            best_delta = delta
                            best_new_c = j
                    
                    if best_new_c is not None:
                        # Accept with Metropolis criterion
                        delta_uncovered = best_delta
                        if delta_uncovered <= 0 or rng.random() < np.exp(-delta_uncovered / max(T, 1e-10)):
                            current_code[i] = best_new_c
                            code_set.discard(old_c)
                            code_set.add(best_new_c)
                            current_cc[old_ball] -= 1
                            current_cc[balls[best_new_c]] += 1
                            uncovered_count += best_delta
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()
                
            elif move_type < 0.7:
                # Pair swap: replace two codewords with two others
                if len(current_code) < 2:
                    continue
                i1, i2 = rng.choice(len(current_code), size=2, replace=False)
                old_c1 = current_code[i1]
                old_c2 = current_code[i2]
                
                # Find replacements
                combined_only = np.zeros(total_words, dtype=bool)
                combined_only[balls[old_c1]] = True
                combined_only[balls[old_c2]] = True
                # Words only covered by old_c1 or old_c2
                mask1 = current_cc[balls[old_c1]] == 1
                mask2 = current_cc[balls[old_c2]] == 1
                only_by_pair = np.zeros(total_words, dtype=bool)
                only_by_pair[balls[old_c1][mask1]] = True
                only_by_pair[balls[old_c2][mask2]] = True
                only_words = np.where(only_by_pair)[0]
                
                if len(only_words) == 0:
                    # Can remove both
                    current_code.pop(i2)
                    current_code.pop(i1)
                    code_set.discard(old_c1)
                    code_set.discard(old_c2)
                    current_cc[balls[old_c1]] -= 1
                    current_cc[balls[old_c2]] -= 1
                    current_size -= 2
                    uncovered_count += len(only_words)
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue
                else:
                    # Sample candidate pairs
                    sample_size = min(50, total_words - len(code_set))
                    if sample_size <= 0:
                        break
                    candidates = [c for c in rng.choice(total_words, size=sample_size, replace=False)
                                  if c not in code_set]
                    
                    best_pair = None
                    best_delta = float('inf')
                    for j1 in candidates:
                        for j2 in candidates:
                            if j1 == j2:
                                continue
                            # Check if {j1, j2} covers all only_words
                            j1_ball = balls[j1]
                            j2_ball = balls[j2]
                            valid = True
                            for w in only_words:
                                if w not in j1_ball and w not in j2_ball:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            # Compute delta
                            old_set = set(balls[old_c1]) | set(balls[old_c2])
                            new_set = set(j1_ball) | set(j2_ball)
                            newly_uncovered = len(old_set - new_set)
                            newly_covered = len(new_set - old_set)
                            delta = newly_uncovered - newly_covered
                            if delta < best_delta:
                                best_delta = delta
                                best_pair = (j1, j2)
                    
                    if best_pair is not None:
                        delta_uncovered = best_delta
                        if delta_uncovered <= 0 or rng.random() < np.exp(-delta_uncovered / max(T, 1e-10)):
                            current_code[i1] = best_pair[0]
                            current_code[i2] = best_pair[1]
                            code_set.discard(old_c1)
                            code_set.discard(old_c2)
                            code_set.add(best_pair[0])
                            code_set.add(best_pair[1])
                            current_cc[balls[old_c1]] -= 1
                            current_cc[balls[old_c2]] -= 1
                            current_cc[balls[best_pair[0]]] += 1
                            current_cc[balls[best_pair[1]]] += 1
                            uncovered_count += best_delta
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()
            
            else:
                # Try to add a codeword if there are uncovered words
                if uncovered_count > 0:
                    uncovered_words = np.where(current_cc == 0)[0]
                    if len(uncovered_words) > 0:
                        # Find center that covers most uncovered words
                        gains = current_cc[balls] == 0
                        gains = gains.sum(axis=1)
                        max_gain = gains.max()
                        if max_gain > 0:
                            best_centers = np.where(gains == max_gain)[0]
                            new_c = int(best_centers[rng.randint(len(best_centers))])
                            if new_c not in code_set:
                                current_code.append(new_c)
                                code_set.add(new_c)
                                current_cc[balls[new_c]] += 1
                                current_size += 1
                                # Update uncovered
                                newly_covered = balls[new_c][current_cc[balls[new_c]] == 1]
                                uncovered_count -= len(newly_covered)
                                if current_size + uncovered_count < best_size + best_uncovered:
                                    best_code = current_code[:]
                                    best_size = current_size
                                    best_uncovered = uncovered_count
                                    best_cc = current_cc.copy()
        
        # Return best solution found
        return best_code, best_cc
=======
    def simulated_annealing(code, cover_count, start_time, deadline, max_iter=500000):
        """
        Simulated annealing with pair-swap and single-swap moves to escape local minima.
        Objective: minimize number of codewords (penalty for uncovered words).
        """
        rng = np.random.RandomState(42)
        
        # Current solution
        current_code = list(code)
        current_cc = cover_count.copy()
        current_size = len(current_code)
        
        # Track uncovered count
        uncovered_mask = current_cc == 0
        uncovered_count = int(uncovered_mask.sum())
        
        best_code = current_code[:]
        best_size = current_size
        best_uncovered = uncovered_count
        best_cc = current_cc.copy()
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        T = T_start
        
        code_set = set(current_code)
        all_centers = list(range(total_words))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Cool down
            progress = iteration / max_iter
            T = T_start * (T_end / T_start) ** progress
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Single swap: replace one codeword with another
                if len(current_code) == 0:
                    break
                i = rng.randint(len(current_code))
                old_c = current_code[i]
                
                # Find candidate: prefer centers that cover uniquely-covered words
                old_ball = balls[old_c]
                only_by_old = old_ball[current_cc[old_ball] == 1]
                
                if len(only_by_old) == 0:
                    # Can remove
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    # Update uncovered
                    newly_uncovered = old_ball[current_cc[old_ball] == 0]
                    uncovered_count += len(newly_uncovered)
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue
                else:
                    # Use word_to_centers to find candidates that cover all only_by_old words
                    if len(only_by_old) > 0:
                        # Start with candidates for the first word
                        candidate_set = set(word_to_centers_np[only_by_old[0]].tolist())
                        # Intersect with candidates for subsequent words
                        for w in only_by_old[1:]:
                            candidate_set &= set(word_to_centers_np[w].tolist())
                            if not candidate_set:
                                break
                        
                        # Filter out existing codewords
                        candidates = [j for j in candidate_set if j not in code_set]
                    else:
                        candidates = []
                    
                    if not candidates:
                        # Fallback to random sampling
                        sample_size = min(100, total_words - len(code_set))
                        if sample_size <= 0:
                            break
                        candidates = [c for c in rng.choice(total_words, size=sample_size, replace=False)
                                      if c not in code_set]
                    
                    best_new_c = None
                    best_delta = float('inf')
                    for j in candidates:
                        j_ball = balls[j]
                        # Check if j covers all of only_by_old
                        valid = True
                        for w in only_by_old:
                            if w not in j_ball:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Compute delta in uncovered count
                        # Words that become uncovered: in old_ball but not in j_ball
                        # Words that become covered: in j_ball but not in old_ball
                        old_set = set(old_ball)
                        j_set = set(j_ball)
                        newly_uncovered = len(old_set - j_set)
                        newly_covered = len(j_set - old_set)
                        delta = newly_uncovered - newly_covered
                        if delta < best_delta:
                            best_delta = delta
                            best_new_c = j
                    
                    if best_new_c is not None:
                        # Accept with Metropolis criterion
                        delta_uncovered = best_delta
                        if delta_uncovered <= 0 or rng.random() < np.exp(-delta_uncovered / max(T, 1e-10)):
                            current_code[i] = best_new_c
                            code_set.discard(old_c)
                            code_set.add(best_new_c)
                            current_cc[old_ball] -= 1
                            current_cc[balls[best_new_c]] += 1
                            uncovered_count += best_delta
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()
                
            elif move_type < 0.7:
                # Pair swap: replace two codewords with two others
                if len(current_code) < 2:
                    continue
                i1, i2 = rng.choice(len(current_code), size=2, replace=False)
                old_c1 = current_code[i1]
                old_c2 = current_code[i2]
                
                # Find replacements
                combined_only = np.zeros(total_words, dtype=bool)
                combined_only[balls[old_c1]] = True
                combined_only[balls[old_c2]] = True
                # Words only covered by old_c1 or old_c2
                mask1 = current_cc[balls[old_c1]] == 1
                mask2 = current_cc[balls[old_c2]] == 1
                only_by_pair = np.zeros(total_words, dtype=bool)
                only_by_pair[balls[old_c1][mask1]] = True
                only_by_pair[balls[old_c2][mask2]] = True
                only_words = np.where(only_by_pair)[0]
                
                if len(only_words) == 0:
                    # Can remove both
                    current_code.pop(i2)
                    current_code.pop(i1)
                    code_set.discard(old_c1)
                    code_set.discard(old_c2)
                    current_cc[balls[old_c1]] -= 1
                    current_cc[balls[old_c2]] -= 1
                    current_size -= 2
                    uncovered_count += len(only_words)
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue
                else:
                    # Use word_to_centers to find candidate pairs
                    # For each word in only_words, find candidates that cover it
                    # A valid pair (j1, j2) must cover all only_words
                    # This is more complex, so use random sampling as fallback
                    sample_size = min(50, total_words - len(code_set))
                    if sample_size <= 0:
                        break
                    candidates = [c for c in rng.choice(total_words, size=sample_size, replace=False)
                                  if c not in code_set]
                    
                    best_pair = None
                    best_delta = float('inf')
                    for j1 in candidates:
                        for j2 in candidates:
                            if j1 == j2:
                                continue
                            # Check if {j1, j2} covers all only_words
                            j1_ball = balls[j1]
                            j2_ball = balls[j2]
                            valid = True
                            for w in only_words:
                                if w not in j1_ball and w not in j2_ball:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            # Compute delta
                            old_set = set(balls[old_c1]) | set(balls[old_c2])
                            new_set = set(j1_ball) | set(j2_ball)
                            newly_uncovered = len(old_set - new_set)
                            newly_covered = len(new_set - old_set)
                            delta = newly_uncovered - newly_covered
                            if delta < best_delta:
                                best_delta = delta
                                best_pair = (j1, j2)
                    
                    if best_pair is not None:
                        delta_uncovered = best_delta
                        if delta_uncovered <= 0 or rng.random() < np.exp(-delta_uncovered / max(T, 1e-10)):
                            current_code[i1] = best_pair[0]
                            current_code[i2] = best_pair[1]
                            code_set.discard(old_c1)
                            code_set.discard(old_c2)
                            code_set.add(best_pair[0])
                            code_set.add(best_pair[1])
                            current_cc[balls[old_c1]] -= 1
                            current_cc[balls[old_c2]] -= 1
                            current_cc[balls[best_pair[0]]] += 1
                            current_cc[balls[best_pair[1]]] += 1
                            uncovered_count += best_delta
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()
            
            else:
                # Try to add a codeword if there are uncovered words
                if uncovered_count > 0:
                    uncovered_words = np.where(current_cc == 0)[0]
                    if len(uncovered_words) > 0:
                        # Find center that covers most uncovered words
                        gains = current_cc[balls] == 0
                        gains = gains.sum(axis=1)
                        max_gain = gains.max()
                        if max_gain > 0:
                            best_centers = np.where(gains == max_gain)[0]
                            new_c = int(best_centers[rng.randint(len(best_centers))])
                            if new_c not in code_set:
                                current_code.append(new_c)
                                code_set.add(new_c)
                                current_cc[balls[new_c]] += 1
                                current_size += 1
                                # Update uncovered
                                newly_covered = balls[new_c][current_cc[balls[new_c]] == 1]
                                uncovered_count -= len(newly_covered)
                                if current_size + uncovered_count < best_size + best_uncovered:
                                    best_code = current_code[:]
                                    best_size = current_size
                                    best_uncovered = uncovered_count
                                    best_cc = current_cc.copy()
        
        # Return best solution found
        return best_code, best_cc
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 3
    for start_i in range(n_starts):
        if time.time() > deadline - 60:  # Leave 60s for SA + final local search
            break

        # Greedy initialization with different randomization
        code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        # Trim
        code, cover_count = trim_code(code, cover_count)

        # Local search
        code, cover_count = targeted_local_search(code, cover_count, max_iter=100000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Simulated annealing on best solution
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        
        # Run SA with time budget
        sa_start = time.time()
        sa_deadline = min(deadline - 20, sa_start + 60)
        best_code, cover_count = simulated_annealing(best_code, cover_count, sa_start, sa_deadline, max_iter=300000)
        
        # Final aggressive local search
        if time.time() < deadline - 10:
            best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=200000)

    # Convert to list of lists
    if best_code is None:
        # Fallback: should not happen
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
=======
    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 4
    for start_i in range(n_starts):
        if time.time() > deadline - 80:  # Leave 80s for SA + final local search
            break

        # Greedy initialization with different randomization
        code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        # Trim
        code, cover_count = trim_code(code, cover_count)

        # Local search
        code, cover_count = targeted_local_search(code, cover_count, max_iter=150000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Simulated annealing on best solution
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        
        # Run SA with time budget
        sa_start = time.time()
        sa_deadline = min(deadline - 30, sa_start + 80)
        best_code, cover_count = simulated_annealing(best_code, cover_count, sa_start, sa_deadline, max_iter=400000)
        
        # Final aggressive local search
        if time.time() < deadline - 15:
            best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=300000)

    # Convert to list of lists
    if best_code is None:
        # Fallback: should not happen
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
>>>>>>> REPLACE

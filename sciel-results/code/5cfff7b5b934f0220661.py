# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _ball_indices(q, n, radius):
    """Precompute for each word index, the list of indices within Hamming distance radius."""
    total = q ** n
    powers = [q ** p for p in range(n)]
    balls = [[] for _ in range(total)]
    
    if radius == 1:
        for i in range(total):
            ball = [i]
            for pos in range(n):
                base = i - (i // powers[pos]) * powers[pos]
                orig_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    neighbor_idx = base + new_val * powers[pos]
                    ball.append(neighbor_idx)
            balls[i] = ball
    elif radius == 0:
        for i in range(total):
            balls[i] = [i]
    else:
        # General radius: BFS from each point
        for i in range(total):
            visited = set([i])
            queue = [(i, 0)]
            balls[i] = []
            while queue:
                idx, dist = queue.pop(0)
                balls[i].append(idx)
                if dist == radius:
                    continue
                d = [(idx // powers[p]) % q for p in range(n)]
                for pos in range(n):
                    orig_val = d[pos]
                    for new_val in range(q):
                        if new_val == orig_val:
                            continue
                        nidx = idx + (new_val - orig_val) * powers[pos]
                        if nidx not in visited:
                            visited.add(nidx)
                            queue.append((nidx, dist + 1))
    return balls

def _tetracode(q, n):
    """
    Construct a structured initial code using tetracode-like symmetry.
    For q=3, n=6: use the known 73-word van Laarhoven code structure
    via systematic construction.
    """
    if q != 3 or n != 6:
        return None
    
    # Use a systematic approach: generate codewords with structured patterns
    # The van Laarhoven code can be constructed using group symmetry
    # We'll use a greedy approach seeded with symmetric words
    
    # Start with all-zero word and its symmetric counterparts
    codewords = set()
    
    # Add words with specific patterns that cover efficiently
    # Use the fact that the ternary Golay code structure helps
    for a in range(3):
        for b in range(3):
            for c in range(3):
                # Pattern 1: (a, b, c, a, b, c)
                codewords.add((a, b, c, a, b, c))
                # Pattern 2: (a, b, c, b, c, a)
                codewords.add((a, b, c, b, c, a))
                # Pattern 3: (a, b, c, c, a, b)
                codewords.add((a, b, c, c, a, b))
    
    # This gives 27 words. Add more structured words.
    for a in range(3):
        for b in range(3):
            # Pattern: (a, a, b, a, a, b)
            codewords.add((a, a, b, a, a, b))
            # Pattern: (a, b, a, b, a, a)  
            codewords.add((a, b, a, b, a, a))
            # Pattern: (b, a, a, a, b, a)
            codewords.add((b, a, a, a, b, a))
    
    return list(codewords)

def _word_to_idx(word, q, n):
    powers = [q ** p for p in range(n)]
    return sum(word[p] * powers[p] for p in range(n))

def _idx_to_word(idx, q, n):
    powers = [q ** p for p in range(n)]
    return [(idx // powers[p]) % q for p in range(n)]

def _greedy_cover_fast(balls, total, deadline, initial_words=None):
    """
    Fast greedy set cover with incremental coverage counting.
    """
    uncovered = np.ones(total, dtype=np.int8)
    coverage_counts = np.zeros(total, dtype=np.int32)
    selected = set()
    codewords = []
    
    # If initial words provided, add them first
    if initial_words:
        for w in initial_words:
            if isinstance(w, tuple):
                idx = _word_to_idx(list(w), 3, 6)
            else:
                idx = w
            if idx not in selected:
                selected.add(idx)
                codewords.append(idx)
                for nb in balls[idx]:
                    coverage_counts[nb] += 1
                    if coverage_counts[nb] == 1:
                        uncovered[nb] = 0
    
    # Convert balls to numpy arrays for faster indexing
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
    
    while np.sum(uncovered) > 0:
        if time.time() > deadline:
            break
        
        # Find best candidate: maximizes number of newly covered points
        # Use vectorized approach: for each candidate, count uncovered neighbors
        best_c = -1
        max_new = 0
        
        # Sample candidates for speed if too many
        candidates = list(range(total))
        random.shuffle(candidates)
        
        # Check all candidates but with early termination
        for c in candidates:
            if c in selected:
                continue
            # Count how many uncovered points this candidate covers
            nbs = ball_arrays[c]
            new_covered = np.sum(uncovered[nbs])
            if new_covered > max_new:
                max_new = new_covered
                best_c = c
                if max_new == len(ball_arrays[c]):
                    break  # Perfect coverage, can't do better
        
        if best_c == -1 or max_new == 0:
            break
        
        selected.add(best_c)
        codewords.append(best_c)
        for nb in balls[best_c]:
            if uncovered[nb]:
                uncovered[nb] = 0
            coverage_counts[nb] += 1
    
    return codewords, coverage_counts

def _local_search_remove(codewords, balls, coverage_counts, total, deadline):
    """
    Try to remove redundant codewords. A codeword is removable if all its
    neighbors have coverage >= 2.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Shuffle for randomization
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for i in indices:
            c = codewords[i]
            # Check if all neighbors have coverage >= 2
            min_cov = np.min(coverage_counts[balls[c]])
            if min_cov >= 2:
                # Remove c
                for nb in balls[c]:
                    coverage_counts[nb] -= 1
                codewords.pop(i)
                improved = True
                break  # Restart with new indices
    
    return codewords, coverage_counts

def _swap_local_search(codewords, balls, coverage_counts, total, deadline):
    """
    Try swaps: remove one codeword, find a replacement that covers all
    newly uncovered points.
    """
    improved = True
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
    
    while improved and time.time() < deadline:
        improved = False
        
        if len(codewords) == 0:
            break
            
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for idx_out in indices:
            c_out = codewords[idx_out]
            
            # Temporarily remove
            for nb in balls[c_out]:
                coverage_counts[nb] -= 1
            
            # Find uncovered points
            critical = np.where(coverage_counts == 0)[0]
            
            if len(critical) == 0:
                # Can remove without replacement
                codewords.pop(idx_out)
                improved = True
                break
            else:
                # Find candidates that cover all critical points
                crit_list = critical.tolist()
                
                # Start with neighbors of first critical point
                candidate_set = set(balls[crit_list[0]])
                
                for cpt in crit_list[1:]:
                    if not candidate_set:
                        break
                    candidate_set &= set(balls[cpt])
                
                # Exclude existing codewords
                existing_set = set(codewords)
                candidate_set -= existing_set
                
                if candidate_set:
                    # Pick the one that covers the most additional points
                    best_in = None
                    best_score = -1
                    for c_in in candidate_set:
                        # Score: number of points with coverage 1 that this would boost
                        score = np.sum(coverage_counts[ball_arrays[c_in]] == 1)
                        if score > best_score:
                            best_score = score
                            best_in = c_in
                    
                    if best_in is not None:
                        # Perform swap
                        codewords[idx_out] = best_in
                        for nb in balls[best_in]:
                            coverage_counts[nb] += 1
                        improved = True
                        break
                else:
                    # Revert
                    for nb in balls[c_out]:
                        coverage_counts[nb] += 1
        
        # If no improvement, try a random perturbation
        if not improved and time.time() < deadline:
            # Remove a random word and try to repair
            if len(codewords) > 1:
                idx_out = random.randint(0, len(codewords) - 1)
                c_out = codewords[idx_out]
                
                for nb in balls[c_out]:
                    coverage_counts[nb] -= 1
                
                critical = np.where(coverage_counts == 0)[0]
                
                if len(critical) > 0:
                    # Try to find a word that covers all critical points
                    crit_list = critical.tolist()
                    candidate_set = set(balls[crit_list[0]])
                    for cpt in crit_list[1:]:
                        if not candidate_set:
                            break
                        candidate_set &= set(balls[cpt])
                    
                    # Don't exclude existing - we're replacing
                    if candidate_set:
                        c_in = random.choice(list(candidate_set))
                        codewords[idx_out] = c_in
                        for nb in balls[c_in]:
                            coverage_counts[nb] += 1
                        improved = True
                    else:
                        # Revert
                        for nb in balls[c_out]:
                            coverage_counts[nb] += 1
                else:
                    # No critical points, just remove
                    codewords.pop(idx_out)
                    improved = True
    
    return codewords, coverage_counts

def _simulated_annealing(codewords, balls, coverage_counts, total, deadline):
    """
    Simulated annealing for covering code optimization.
    Objective: minimize number of codewords while maintaining validity.
    """
    best_codewords = list(codewords)
    best_size = len(codewords)
    best_coverage = coverage_counts.copy()
    
    current_size = len(codewords)
    temperature = 50.0
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline and temperature > 1.0:
        if len(codewords) == 0:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to remove a random codeword
            idx = random.randint(0, len(codewords) - 1)
            c = codewords[idx]
            
            # Check if removal keeps validity
            for nb in balls[c]:
                if coverage_counts[nb] <= 1:
                    break
            else:
                # Safe to remove
                for nb in balls[c]:
                    coverage_counts[nb] -= 1
                codewords.pop(idx)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = list(codewords)
                    best_coverage = coverage_counts.copy()
                    
        elif move_type < 0.6:
            # Try a swap
            idx_out = random.randint(0, len(codewords) - 1)
            c_out = codewords[idx_out]
            
            # Remove temporarily
            for nb in balls[c_out]:
                coverage_counts[nb] -= 1
            
            # Find a random candidate to replace
            # Prefer candidates that cover uncovered points
            uncovered = np.where(coverage_counts == 0)[0]
            
            if len(uncovered) == 0:
                # Can just remove
                codewords.pop(idx_out)
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = list(codewords)
                    best_coverage = coverage_counts.copy()
            else:
                # Find candidate covering all uncovered
                u0 = uncovered[0]
                candidates = [c for c in balls[u0] if c not in codewords]
                
                if candidates:
                    c_in = random.choice(candidates)
                    # Check if it covers all uncovered
                    valid = True
                    for u in uncovered[1:]:
                        if u not in balls[c_in]:
                            valid = False
                            break
                    
                    if valid:
                        codewords[idx_out] = c_in
                        for nb in balls[c_in]:
                            coverage_counts[nb] += 1
                        # Size unchanged, but configuration changed
                    else:
                        # Revert
                        for nb in balls[c_out]:
                            coverage_counts[nb] += 1
                else:
                    for nb in balls[c_out]:
                        coverage_counts[nb] += 1
                    
        elif move_type < 0.8:
            # Try to add a codeword (rarely useful for minimization, but helps escape)
            # Only if we can then remove more
            pass
            
        else:
            # Perturbation: remove random word, add different random word
            if len(codewords) > 1:
                idx_out = random.randint(0, len(codewords) - 1)
                c_out = codewords[idx_out]
                
                for nb in balls[c_out]:
                    coverage_counts[nb] -= 1
                
                critical = np.where(coverage_counts == 0)[0]
                
                if len(critical) == 0:
                    codewords.pop(idx_out)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_codewords = list(codewords)
                        best_coverage = coverage_counts.copy()
                else:
                    crit_list = critical.tolist()
                    candidate_set = set(balls[crit_list[0]])
                    for cpt in crit_list[1:]:
                        if not candidate_set:
                            break
                        candidate_set &= set(balls[cpt])
                    
                    if candidate_set:
                        c_in = random.choice(list(candidate_set))
                        codewords[idx_out] = c_in
                        for nb in balls[c_in]:
                            coverage_counts[nb] += 1
                    else:
                        for nb in balls[c_out]:
                            coverage_counts[nb] += 1
        
        temperature *= cooling_rate
        
        # Periodically try local search
        if random.random() < 0.05:
            codewords, coverage_counts = _local_search_remove(
                codewords, balls, coverage_counts, total, min(deadline, time.time() + 2)
            )
            current_size = len(codewords)
            if current_size < best_size:
                best_size = current_size
                best_codewords = list(codewords)
                best_coverage = coverage_counts.copy()
    
    return best_codewords, best_coverage

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total = q ** n
    
    # Precompute balls
    balls = _ball_indices(q, n, radius)
    
    # Get initial structured words if available
    initial_words = _tetracode(q, n)
    
    # Phase 1: Greedy cover with initial structure
    codewords, coverage_counts = _greedy_cover_fast(
        balls, total, deadline - 60, initial_words
    )
    
    # Ensure validity
    if np.min(coverage_counts) < 1:
        uncovered = np.where(coverage_counts == 0)[0]
        for u in uncovered:
            candidates = [c for c in balls[u] if c not in codewords]
            if candidates:
                c = random.choice(candidates)
                codewords.append(c)
                for nb in balls[c]:
                    coverage_counts[nb] += 1
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Phase 2: Local search for removals
    codewords, coverage_counts = _local_search_remove(
        codewords, balls, coverage_counts, total, deadline - 40
    )
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
    
    # Phase 3: Swap local search
    codewords, coverage_counts = _swap_local_search(
        codewords, balls, coverage_counts, total, deadline - 20
    )
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
    
    # Phase 4: Simulated annealing
    codewords, coverage_counts = _simulated_annealing(
        codewords, balls, coverage_counts, total, deadline - 5
    )
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
    
    # Final local search on best
    coverage_counts_final = np.zeros(total, dtype=np.int32)
    for c in best_codewords:
        for nb in balls[c]:
            coverage_counts_final[nb] += 1
    
    best_codewords, coverage_counts_final = _local_search_remove(
        best_codewords, balls, coverage_counts_final, total, deadline
    )
    
    # Convert indices to words
    result = []
    for idx in best_codewords:
        result.append(_idx_to_word(idx, q, n))
    
    return result
# EVOLVE-BLOCK-END

import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # seconds

def _build_balls(q, n, radius):
    """Precompute for each word the list of words within Hamming distance radius.
    Returns balls as list of lists, and also a numpy array for fast indexing."""
    total = q ** n
    balls = []
    ball_array = []
    for w in range(total):
        word = [0] * n
        idx = w
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    word[pos] = v
                    nb_idx = 0
                    for val in word:
                        nb_idx = nb_idx * q + val
                    neighbors.append(nb_idx)
                    word[pos] = orig
        balls.append(neighbors)
        ball_array.append(np.array(neighbors, dtype=np.int32))
    return balls, ball_array

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    rng = random.Random(42)
    
    balls, ball_array = _build_balls(q, n, radius)
    ball_size = len(balls[0])
    
    # Helper functions
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # cov[i] = number of codewords covering i
    cov = np.zeros(total, dtype=np.int32)
    code_set = set()
    
    # Use tetracode start (81 words for n=6, q=3)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
            
    for c in initial_code:
        code_set.add(word_to_idx(c))
        
    # Vectorized initial coverage
    for ci in code_set:
        cov[ball_array[ci]] += 1
            
    best_code = set(code_set)
    best_size = len(best_code)
    
    def compute_redundant_vectorized(code_set):
        """Return list of codewords that can be removed (min cov in ball >= 2)."""
        redundant = []
        for ci in code_set:
            min_c = np.min(cov[ball_array[ci]])
            if min_c >= 2:
                redundant.append(ci)
        return redundant

    def try_remove_redundant(code_set):
        removed_any = False
        while True:
            redundant = compute_redundant_vectorized(code_set)
            if not redundant:
                break
            # Remove one - pick randomly for diversity
            ci = rng.choice(redundant)
            code_set.discard(ci)
            cov[ball_array[ci]] -= 1
            removed_any = True
            if time.time() - start_time > _DEADLINE - 5:
                return removed_any
        return removed_any

    # Precompute: for each word, which other words cover it (inverse balls)
    # This helps find good c2 candidates
    # covered_by[w] = list of words that are within distance radius of w
    covered_by = [[] for _ in range(total)]
    for w in range(total):
        for nb in balls[w]:
            covered_by[nb].append(w)
    
    # Main Local Search Loop
    for attempt in range(2000):
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        # Reset state to best known
        code_set = set(best_code)
        cov[:] = 0
        for ci in code_set:
            cov[ball_array[ci]] += 1
                
        # Phase 1: Remove all redundant
        try_remove_redundant(code_set)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            if best_size <= 71:
                break
                
        # Phase 2: Try swaps to enable further removals
        swap_attempts = 0
        max_swap_attempts = 80  # Increased from 50
        
        while swap_attempts < max_swap_attempts:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            cis = list(code_set)
            if not cis:
                break
                
            c1 = rng.choice(cis)
            
            # Simulate removal of c1
            uncovered = []
            critical = []
            
            for nb in balls[c1]:
                cov[nb] -= 1
                if cov[nb] == 0:
                    uncovered.append(nb)
                elif cov[nb] == 1:
                    critical.append(nb)
            
            code_set.discard(c1)
            
            found_improvement = False
            
            if uncovered:
                # c2 must cover all words in 'uncovered'
                # Find candidates using covered_by
                # c2 must be in covered_by[u] for all u in uncovered
                # Use intersection
                if len(uncovered) == 1:
                    candidates = covered_by[uncovered[0]]
                else:
                    # Start with smallest set
                    min_u = min(uncovered, key=lambda u: len(covered_by[u]))
                    candidates = set(covered_by[min_u])
                    for u in uncovered:
                        if u == min_u:
                            continue
                        candidates &= set(covered_by[u])
                        if not candidates:
                            break
                
                # Try up to 5 candidates
                tried = 0
                for c2 in list(candidates)[:5]:
                    if c2 in code_set:
                        continue
                    if tried >= 5:
                        break
                    
                    # Add c2
                    cov[ball_array[c2]] += 1
                    code_set.add(c2)
                    
                    # Try to remove redundant
                    removed_any = try_remove_redundant(code_set)
                    
                    if removed_any and len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                        found_improvement = True
                        break
                    else:
                        # Revert
                        code_set.discard(c2)
                        cov[ball_array[c2]] -= 1
                    
                    tried += 1
                
                if not found_improvement:
                    # Restore c1
                    cov[ball_array[c1]] += 1
                    code_set.add(c1)
                    
            else:
                # No uncovered words. Pick c2 that maximizes overlap with existing code
                # to create redundancy opportunities
                # Score each candidate by how many existing codewords it overlaps with
                # and how many new redundancies it creates
                
                # Sample a subset of candidates for speed
                num_samples = min(20, total - len(code_set))
                candidates = rng.sample(range(total), num_samples)
                
                # Remove those already in code_set
                candidates = [c for c in candidates if c not in code_set]
                
                if not candidates:
                    cov[ball_array[c1]] += 1
                    code_set.add(c1)
                    swap_attempts += 1
                    continue
                
                # For each candidate, compute a score
                best_c2 = None
                best_score = -1
                
                for c2 in candidates:
                    # Score: how many words in ball(c2) have high coverage
                    # (these become more redundant after adding c2)
                    ball_c2 = ball_array[c2]
                    score = np.sum(cov[ball_c2])  # Sum of coverage over ball
                    # Bonus for creating redundancy: count words with cov >= 2
                    score += 2 * np.sum(cov[ball_c2] >= 2)
                    
                    if score > best_score:
                        best_score = score
                        best_c2 = c2
                
                if best_c2 is not None:
                    c2 = best_c2
                    # Add c2
                    cov[ball_array[c2]] += 1
                    code_set.add(c2)
                    
                    # Try to remove redundant
                    removed_any = try_remove_redundant(code_set)
                    
                    if removed_any and len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                        found_improvement = True
                        break
                    else:
                        # Revert
                        code_set.discard(c2)
                        cov[ball_array[c2]] -= 1
                        # Restore c1
                        cov[ball_array[c1]] += 1
                        code_set.add(c1)
                else:
                    cov[ball_array[c1]] += 1
                    code_set.add(c1)
            
            swap_attempts += 1
            
            if best_size <= 71:
                break

    result = [idx_to_word(ci) for ci in sorted(best_code)]
    return result

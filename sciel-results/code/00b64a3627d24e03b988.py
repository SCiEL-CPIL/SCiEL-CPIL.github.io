"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

def _word_to_idx(w, q, n):
    """Convert a word (tuple/list of ints) to an index in [0, q^n)."""
    idx = 0
    for i in range(n):
        idx = idx * q + w[i]
    return idx

def _idx_to_word(idx, q, n):
    """Convert an index to a word (list of ints)."""
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w

def _generate_balls(q, n, radius):
    """
    Precompute the ball (set of covered word indices) for each possible codeword.
    For q=3, n=7, radius=1, this is 2187 sets of size 15.
    Returns a list of numpy arrays for fast boolean masking.
    """
    total_words = q ** n
    balls = []
    
    # We can compute neighbors efficiently
    # For radius 1: self + n*(q-1) neighbors
    # For general radius, we'd need to generate combinations, but for this task r=1 is standard.
    # Let's implement a generic ball generator for safety, but optimize for r=1 if possible.
    
    for idx in range(total_words):
        w = _idx_to_word(idx, q, n)
        covered = {idx}
        
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w[pos] = val
                    covered.add(_word_to_idx(w, q, n))
                w[pos] = orig
        
        if radius >= 2:
            # Only implement up to radius 2 for efficiency, as r>2 is rare in these small n tasks
            # and the space is small enough that r=1 is the main case.
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = w[pos1]
                    orig2 = w[pos2]
                    for val1 in range(q):
                        if val1 == orig1:
                            continue
                        w[pos1] = val1
                        for val2 in range(q):
                            if val2 == orig2:
                                continue
                            w[pos2] = val2
                            covered.add(_word_to_idx(w, q, n))
                        w[pos2] = orig2
                    w[pos1] = orig1
        
        balls.append(np.array(sorted(covered), dtype=np.int64))
        
    return balls

def _is_covered(code, q, n, radius, balls):
    """Check if a code covers the entire space."""
    if not code:
        return False
    covered_mask = np.zeros(q ** n, dtype=bool)
    for cw in code:
        idx = _word_to_idx(cw, q, n)
        covered_mask[balls[idx]] = True
    return covered_mask.all()

def _greedy_init(q, n, radius, balls, time_limit=10):
    """
    Fast greedy initialization.
    """
    start_time = time.time()
    total_words = q ** n
    covered_mask = np.zeros(total_words, dtype=bool)
    selected = []
    
    while not covered_mask.all():
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_count = -1
        # Optimization: We can skip indices that are already covered? No, a covered index can still cover many *other* uncovered indices.
        # However, we can maintain a list of uncovered words and for each candidate, count how many of those it covers.
        
        # To speed this up, we iterate through all candidates. 
        # 2187 candidates * 186 steps is ~400k operations. Each op is a numpy sum over 15 elements.
        # This is fast enough in Python/NumPy.
        
        for idx in range(total_words):
            # Count how many currently uncovered words are covered by this candidate
            b = balls[idx]
            # We only care about the ones that are NOT covered yet
            count = np.sum(~covered_mask[b])
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
            
        selected.append(best_idx)
        covered_mask[balls[best_idx]] = True
        
    return [_idx_to_word(idx, q, n) for idx in selected]

def _repair(code, q, n, radius, balls, time_limit=5):
    """
    Repair a code that might have holes by adding codewords that cover the most uncovered words.
    This is similar to greedy_init but starts from an existing code.
    """
    start_time = time.time()
    total_words = q ** n
    covered_mask = np.zeros(total_words, dtype=bool)
    
    # Mark currently covered
    for cw in code:
        idx = _word_to_idx(cw, q, n)
        covered_mask[balls[idx]] = True
        
    if covered_mask.all():
        return code
        
    new_code = list(code)
    
    while not covered_mask.all():
        if time.time() - start_time > time_limit:
            break
            
        best_idx = -1
        best_count = -1
        
        # Find the candidate that covers the most uncovered words
        for idx in range(total_words):
            b = balls[idx]
            count = np.sum(~covered_mask[b])
            if count > best_count:
                best_count = count
                best_idx = idx
                
        if best_idx == -1 or best_count == 0:
            break
            
        new_code.append(best_idx)
        covered_mask[balls[best_idx]] = True
        
    return [_idx_to_word(idx, q, n) for idx in new_code]

def _reduce(code, q, n, radius, balls, time_limit=5):
    """
    Remove redundant codewords.
    """
    start_time = time.time()
    code = list(code)
    changed = True
    
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(code)):
            if time.time() - start_time > time_limit:
                break
            # Try removing code[i]
            test_code = code[:i] + code[i+1:]
            if _is_covered(test_code, q, n, radius, balls):
                code = test_code
                changed = True
                break
                
    return code

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0,...,q-1} with covering radius r.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    total_time_limit = 110 # Safety margin for 120s limit
    
    # 1. Precompute balls
    balls = _generate_balls(q, n, radius)
    
    # 2. Initial Greedy
    code = _greedy_init(q, n, radius, balls, time_limit=20)
    
    # Ensure it's a valid cover (greedy might stop early due to time, but usually finishes)
    if not _is_covered(code, q, n, radius, balls):
        code = _repair(code, q, n, radius, balls, time_limit=10)
        
    best_code = code
    best_len = len(code)
    
    # 3. Iterated Local Search
    rng = random.Random(42)
    
    while time.time() - start_time < total_time_limit:
        # Perturbation: Remove a few random codewords
        if len(code) <= 1:
            break
            
        # Remove 1-3 codewords randomly
        num_remove = rng.randint(1, min(3, len(code)))
        indices_to_remove = rng.sample(range(len(code)), num_remove)
        
        perturbed_code = []
        remove_set = set(indices_to_remove)
        for i, cw in enumerate(code):
            if i not in remove_set:
                perturbed_code.append(cw)
                
        # Repair: Add back necessary codewords
        repaired_code = _repair(perturbed_code, q, n, radius, balls, time_limit=5)
        
        # Reduce: Remove redundancies
        reduced_code = _reduce(repaired_code, q, n, radius, balls, time_limit=5)
        
        # Check validity
        if _is_covered(reduced_code, q, n, radius, balls):
            if len(reduced_code) < best_len:
                best_len = len(reduced_code)
                best_code = reduced_code
                code = reduced_code # Update current state to the new best
            else:
                # Even if not better, accept with some probability (simulated annealing style) 
                # or just keep the best. Let's stick to hill climbing for now.
                # If we want to explore more, we could accept worse solutions sometimes.
                # But for covering codes, "worse" means more codewords, which is harder to reduce.
                # Let's just keep the best.
                pass
        else:
            # If repair failed to cover, revert to previous best or current code
            pass
            
    return best_code
# EVOLVE-BLOCK-END

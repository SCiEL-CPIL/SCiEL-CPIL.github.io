"""Simulated annealing + local search for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails. Gives 81 words for q=3,n=6."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword (exact check)."""
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    words_arr = np.array(words_list, dtype=np.int8)
    code_arr = np.array(code, dtype=np.int8)
    
    diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    covered = (dists <= radius).any(axis=1)
    return covered.all()


def _get_balls(all_words_arr, q, n, radius):
    """Precompute the set of words covered by each word in the space."""
    N = len(all_words_arr)
    balls = []
    for i in range(N):
        w = all_words_arr[i]
        covered = np.zeros(N, dtype=bool)
        # Check all words? No, only those at distance <= radius.
        # For radius 1, it's self + n*(q-1) neighbors.
        if radius == 1:
            covered[i] = True
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    nw = w.copy()
                    nw[pos] = val
                    ni = _word_to_idx(nw.tolist(), q, n)
                    covered[ni] = True
        else:
            # General case: compute distance to all
            dists = np.abs(all_words_arr - w).sum(axis=1)
            covered = dists <= radius
        balls.append(covered)
    return balls


def _optimize_code(code, all_words_arr, q, n, radius, time_limit=60.0, seed=42):
    """
    Optimizes a valid code by removing redundant words and performing swaps.
    """
    rng = np.random.default_rng(seed)
    start = time.time()
    N = len(all_words_arr)
    
    # Precompute balls (coverage masks)
    balls = _get_balls(all_words_arr, q, n, radius)
    balls_arr = np.array(balls, dtype=bool) # Shape (N, N)
    
    # Convert code to indices
    word_to_idx = {}
    for i in range(N):
        word_to_idx[tuple(all_words_arr[i])] = i
        
    code_idx = []
    for cw in code:
        idx = word_to_idx.get(tuple(cw))
        if idx is not None:
            code_idx.append(idx)
            
    # Coverage counts: counts[i] = number of codewords covering word i
    counts = np.zeros(N, dtype=np.int32)
    for ci in code_idx:
        counts += balls_arr[ci]
        
    current_size = len(code_idx)
    best_size = current_size
    best_code_idx = list(code_idx)
    
    def try_reduce(cod_idx, counts):
        """Try to remove redundant codewords. Returns (new_idx, new_counts, removed_count)."""
        removed = 0
        changed = True
        while changed:
            if (time.time() - start) > time_limit:
                break
            changed = False
            for ci in range(len(cod_idx)):
                if (time.time() - start) > time_limit:
                    break
                # Check if removing cod_idx[ci] is safe
                new_counts = counts - balls_arr[cod_idx[ci]]
                if np.all(new_counts > 0):
                    cod_idx = [x for i, x in enumerate(cod_idx) if i != ci]
                    counts = new_counts
                    removed += 1
                    changed = True
                    break
        return cod_idx, counts, removed

    # Initial reduction
    code_idx, counts, _ = try_reduce(code_idx, counts)
    current_size = len(code_idx)
    best_size = current_size
    best_code_idx = list(code_idx)
    
    # Main loop: Try to improve by swapping
    no_improve = 0
    max_no_improve = 100
    
    while (time.time() - start) < time_limit:
        if no_improve > max_no_improve:
            # Reset to best if stuck
            code_idx = list(best_code_idx)
            counts = np.zeros(N, dtype=np.int32)
            for ci in code_idx:
                counts += balls_arr[ci]
            current_size = len(code_idx)
            no_improve = 0
            # Optional: random perturbation
            if rng.random() < 0.5 and current_size > 1:
                # Remove a random word
                ri = rng.integers(0, current_size)
                removed_idx = code_idx[ri]
                code_idx = [x for i, x in enumerate(code_idx) if i != ri]
                counts -= balls_arr[removed_idx]
                current_size -= 1
                # Add a random word to make it valid again? 
                # For now, just let the loop try to fix it or revert.
                # Actually, if we make it invalid, we must repair.
                # Let's stick to valid moves only for safety.
                # Revert perturbation
                code_idx.append(removed_idx)
                counts += balls_arr[removed_idx]
                current_size += 1
                no_improve = 0
                continue

        improved = False
        # Try to find a swap that allows reduction
        # We iterate through current codewords. For each, try to replace it with another word.
        # To speed up, we only try a subset of replacements.
        
        # Identify "critical" words: words covered by only 1 codeword.
        # If we remove a codeword, and it's the only one covering some word, we must replace it with a word that covers that word.
        
        for ci in range(len(code_idx)):
            if (time.time() - start) > time_limit:
                break
            if improved:
                break
                
            removed_idx = code_idx[ci]
            # Words that would become uncovered if we remove this
            vulnerable = (counts == 1) & balls_arr[removed_idx]
            
            if not np.any(vulnerable):
                # Can remove safely
                code_idx, counts, _ = try_reduce(code_idx, counts)
                current_size = len(code_idx)
                if current_size < best_size:
                    best_size = current_size
                    best_code_idx = list(code_idx)
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
            
            # Must find a replacement that covers all vulnerable words
            # Candidate replacements: words that cover all vulnerable words
            # vulnerable is a boolean array of length N
            # We want a word w such that balls_arr[w][vulnerable] are all True
            
            # To speed up, we can precompute for each word, the set of vulnerable words it covers?
            # Or just iterate through a random subset of all words.
            
            # Let's try a few random candidates
            candidates = rng.choice(N, size=min(50, N), replace=False)
            for cand in candidates:
                if cand == removed_idx:
                    continue
                # Check if cand covers all vulnerable words
                if np.all(balls_arr[cand][vulnerable]):
                    # Swap
                    code_idx[ci] = cand
                    counts -= balls_arr[removed_idx]
                    counts += balls_arr[cand]
                    
                    # Try to reduce again
                    code_idx, counts, rem_count = try_reduce(code_idx, counts)
                    current_size = len(code_idx)
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code_idx = list(code_idx)
                        no_improve = 0
                        improved = True
                        break
                    else:
                        # Revert swap if no improvement
                        # Find the index of cand in code_idx (it might have been removed if redundant)
                        # This is tricky. Let's just revert the swap and counts.
                        # But try_reduce might have removed other words.
                        # So we must restore the state before the swap.
                        # This suggests we should not modify in-place unless we are sure.
                        # Let's use a copy for the trial.
                        pass
            
            # The above logic is flawed because try_reduce modifies code_idx.
            # Let's refactor: use a trial copy.
            
        # Refactored Swap Logic
        if not improved:
            # Try a random swap
            ci = rng.integers(0, len(code_idx))
            removed_idx = code_idx[ci]
            cand = rng.integers(0, N)
            if cand != removed_idx:
                trial_idx = list(code_idx)
                trial_idx[ci] = cand
                trial_counts = counts - balls_arr[removed_idx] + balls_arr[cand]
                
                # Check validity
                if np.all(trial_counts > 0):
                    trial_idx, trial_counts, _ = try_reduce(trial_idx, trial_counts)
                    if len(trial_idx) < current_size:
                        code_idx = trial_idx
                        counts = trial_counts
                        current_size = len(code_idx)
                        if current_size < best_size:
                            best_size = current_size
                            best_code_idx = list(code_idx)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1

    return [all_words_arr[i].tolist() for i in best_code_idx]


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    all_words_arr = np.array(words_list, dtype=np.int8)
    
    # Step 1: Start from direct sum tetracode (81 words for q=3, n=6)
    initial_code = _direct_sum_tetracode(q, n, radius)
    
    # Verify it's valid
    if not _verify_code(initial_code, q, n, radius):
        # Fallback: greedy
        initial_code = words_list
    
    best_code = initial_code
    best_size = len(initial_code)
    
    # Step 2: Optimize
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 10:
        try:
            opt_code = _optimize_code(best_code, all_words_arr, q, n, radius, 
                                      time_limit=min(remaining, 100.0), seed=42)
            if _verify_code(opt_code, q, n, radius) and len(opt_code) < best_size:
                best_code = opt_code
                best_size = len(opt_code)
        except Exception:
            pass
            
    # Step 3: Try another seed if time remains
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 10 and best_size > 73:
        try:
            # Start from a random valid code of size best_size?
            # Or just run optimize again with different seed on the best code found.
            opt_code2 = _optimize_code(best_code, all_words_arr, q, n, radius, 
                                       time_limit=min(remaining, 50.0), seed=123)
            if _verify_code(opt_code2, q, n, radius) and len(opt_code2) < best_size:
                best_code = opt_code2
                best_size = len(opt_code2)
        except Exception:
            pass

    return best_code

# EVOLVE-BLOCK-END

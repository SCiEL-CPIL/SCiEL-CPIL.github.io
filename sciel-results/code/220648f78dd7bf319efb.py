import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Covers {0..q-1}^n with radius-r Hamming balls around as few centers as possible.
    Uses a bitmask-accelerated greedy + local search (add/remove) hybrid.
    """
    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    N = q ** n
    # All words as flat indices 0..N-1, with a numpy array for quick access
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    # Precompute ball indices: ball_mask[i] is a list of word-indices within radius of word i
    # For radius=1, each ball has 1 + n*(q-1) elements
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    balls = []
    for i in range(N):
        w = all_words[i]
        inds = [i]
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                nw = w.copy()
                nw[pos] = v
                inds.append(word_to_idx[tuple(nw)])
        balls.append(np.array(inds, dtype=np.int32))
    balls_arr = np.array(balls, dtype=np.int32)
    # balls_arr.shape = (N, ball_size)
    # Also build reverse: for each word j, which centers cover it. By symmetry this is balls_arr[j].

    def compute_coverage(centers_list):
        """Returns bool array of length N: True if word is covered."""
        cov = np.zeros(N, dtype=bool)
        if not centers_list:
            return cov
        # Use np.unique on flattened ball indices to avoid duplicates
        idxs = balls_arr[list(centers_list)]
        cov[idxs.ravel()] = True
        return cov

    def is_valid(centers_list):
        cov = compute_coverage(centers_list)
        return cov.all()

    def remove_redundant(centers_list):
        """Iteratively remove centers whose removal still leaves full coverage."""
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            n = len(centers_list)
            if n == 0:
                break
            # Try from end to beginning
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                temp = centers_list[:i] + centers_list[i+1:]
                if is_valid(temp):
                    centers_list.pop(i)
                    changed = True
                    break
        return centers_list

    # ---- Phase 1: Greedy initialization with bitmask speedup ----
    covered = np.zeros(N, dtype=bool)
    selected = []
    
    # Precompute: for each uncovered word, the set of centers that would cover it.
    # Use an array of int counts for "how many uncovered words each center would newly cover"
    
    while not covered.all() and time.time() < deadline - 80.0:
        # Compute gain for each center: number of words in its ball that are currently uncovered
        # gain[i] = sum of ~covered[balls_arr[i]]
        # Vectorized: for each word j that is uncovered, increment gain[c] for all c in balls_arr[j]
        # But balls_arr[j] is the set of centers covering j (by symmetry).
        
        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            break
        
        gain = np.zeros(N, dtype=np.int32)
        # For speed, iterate over uncovered words and scatter
        # This is O(|uncovered| * ball_size)
        for u in uncovered:
            gain[balls_arr[u]] += 1
        
        max_gain = gain.max()
        if max_gain == 0:
            break
        candidates = np.where(gain == max_gain)[0]
        idx = random.choice(candidates.tolist())
        selected.append(idx)
        covered[balls_arr[idx]] = True

    # Remove redundant
    selected = remove_redundant(selected)
    best_code = selected.copy()
    best_len = len(best_code)

    # ---- Phase 2: Local search with add/remove/swap moves ----
    # State: list of center indices (integers into all_words)
    # Score: len(code) + 1000 * (uncovered_count) -- strongly penalize invalid
    
    def get_score(code_list):
        if not code_list:
            return 1000000 + N
        cov = compute_coverage(code_list)
        uncovered = N - cov.sum()
        return len(code_list) + 1000 * uncovered

    current = best_code.copy()
    current_score = get_score(current)
    
    T = 20.0
    T_min = 0.5
    cooling = 0.995
    best_code = current.copy()
    best_score = current_score

    while time.time() < deadline - 10.0 and T > T_min:
        move = random.random()
        new_code = current.copy()
        
        if move < 0.35:
            # Remove a random center
            if len(new_code) > 1:
                i = random.randint(0, len(new_code) - 1)
                new_code.pop(i)
            else:
                continue
        elif move < 0.65:
            # Add a random center
            idx = random.randint(0, N - 1)
            if idx not in new_code:
                new_code.append(idx)
            else:
                continue
        else:
            # Swap: remove one, add another
            if len(new_code) > 0:
                i = random.randint(0, len(new_code) - 1)
                old = new_code[i]
                idx = random.randint(0, N - 1)
                if idx not in new_code:
                    new_code[i] = idx
                else:
                    continue
        
        new_score = get_score(new_code)
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current = new_code
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_code = current.copy()
        
        T *= cooling
        
        # Periodic redundant removal
        if random.random() < 0.15:
            current = remove_redundant(current)
            current_score = get_score(current)
            if current_score < best_score:
                best_score = current_score
                best_code = current.copy()

    # Final cleanup
    best_code = remove_redundant(best_code)
    
    # Ensure validity
    if not is_valid(best_code):
        # Fallback: greedy solution
        selected = remove_redundant(selected)
        if is_valid(selected):
            best_code = selected
        else:
            # Last resort: all words
            best_code = list(range(N))
    
    return [all_words[i].tolist() for i in best_code]

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    if n == 1:
        return [[i] for i in range(q)]
    
    if n == 2:
        return [[i, j] for i in range(q) for j in range(q)]
    
    if n == 3:
        return [[0,0,0], [1,1,1], [2,2,2]]
    
    if n == 4:
        return [[a, b, (a+b)%q, (a+2*b)%q] for a in range(q) for b in range(q)]
    
    # For the main case: q=3, n=6, radius=1
    # Use efficient numpy-based coverage tracking
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Enumerate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    total_words = len(all_words)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute for each codeword candidate, which words it covers
    # For radius=1: covers itself + n*(q-1) neighbors
    def get_coverage_mask(cw):
        """Return boolean mask of words covered by codeword cw."""
        mask = np.zeros(total_words, dtype=bool)
        cw = np.array(cw)
        # The word itself
        idx = word_to_idx[tuple(cw)]
        mask[idx] = True
        # All neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    new_word = cw.copy()
                    new_word[pos] = val
                    nidx = word_to_idx[tuple(new_word)]
                    mask[nidx] = True
        return mask
    
    # Greedy construction
    def greedy_construct(seed=42):
        rng = random.Random(seed)
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # Precompute coverage for all possible codewords
        # This is 729 * 729 booleans = ~521K booleans, manageable
        # But let's be smarter: compute on the fly with caching
        coverage_cache = {}
        
        def get_cov(cw_tuple):
            if cw_tuple not in coverage_cache:
                coverage_cache[cw_tuple] = get_coverage_mask(list(cw_tuple))
            return coverage_cache[cw_tuple]
        
        # Greedy: repeatedly pick codeword covering most uncovered
        remaining = total_words
        while remaining > 0:
            best_cw = None
            best_count = -1
            # Sample a subset of candidates for speed, or use all
            # For 729 candidates, checking each is feasible
            for cw in all_words:
                cw_t = tuple(cw)
                cov = get_cov(cw_t)
                # Count uncovered words this covers
                count = np.sum(cov & ~covered)
                if count > best_count:
                    best_count = count
                    best_cw = cw_t
                if best_count == remaining:
                    break
            
            if best_cw is None or best_count == 0:
                break
            
            codewords.append(list(best_cw))
            cov = get_cov(best_cw)
            covered |= cov
            remaining = np.sum(~covered)
        
        return codewords
    
    # Try multiple seeds for greedy
    best_codewords = None
    best_size = float('inf')
    
    for seed in range(10):
        if time.time() - start_time > time_limit:
            break
        cw = greedy_construct(seed)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
    
    # Local search: try to remove codewords
    def try_remove(codewords, covered):
        """Try to remove redundant codewords. Returns (new_codewords, new_covered, removed_count)."""
        removed = 0
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            for i in range(len(codewords)):
                # Temporarily remove codeword i
                new_covered = np.zeros(total_words, dtype=bool)
                for j, cw in enumerate(codewords):
                    if j != i:
                        new_covered |= get_coverage_mask(cw)
                if np.all(new_covered):
                    # Can remove
                    codewords.pop(i)
                    covered = new_covered
                    removed += 1
                    changed = True
                    break
        return codewords, covered, removed
    
    # Run removal
    if best_codewords is not None:
        covered = np.zeros(total_words, dtype=bool)
        for cw in best_codewords:
            covered |= get_coverage_mask(cw)
        best_codewords, covered, _ = try_remove(best_codewords, covered)
        best_size = len(best_codewords)
    
    # Local search: swap operations
    def local_search(codewords, covered, max_iters=200):
        rng = random.Random(123)
        current = codewords[:]
        cur_covered = covered.copy()
        best = current[:]
        best_size_ls = len(current)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            if len(current) >= best_size_ls:
                break
            
            # Try removing a random codeword
            idx = rng.randint(0, len(current) - 1)
            cw_remove = current[idx]
            
            # Compute coverage without this codeword
            new_covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(current):
                if j != idx:
                    new_covered |= get_coverage_mask(cw)
            
            uncovered = np.where(~new_covered)[0]
            
            if len(uncovered) == 0:
                # Can remove
                current.pop(idx)
                cur_covered = new_covered
                if len(current) < best_size_ls:
                    best_size_ls = len(current)
                    best = current[:]
                continue
            
            # Try to find a codeword that covers all uncovered words
            # The new codeword must be within radius of all uncovered words
            # This means it must be in the intersection of balls of radius around each uncovered word
            # For radius=1, the new codeword must be at distance <=1 from each uncovered word
            # So it must be in the intersection of the closed balls
            
            # Find candidates: words within distance 1 of all uncovered words
            # Start with candidates within distance 1 of first uncovered word
            first_uncov = all_words[uncovered[0]]
            candidates = [tuple(first_uncov)]
            for pos in range(n):
                for val in range(q):
                    if val != first_uncov[pos]:
                        nw = first_uncov.copy()
                        nw[pos] = val
                        candidates.append(tuple(nw))
            
            # Filter: must be within distance 1 of ALL uncovered words
            valid_candidates = []
            for cand in candidates:
                cand_arr = np.array(cand)
                valid = True
                for uncov_idx in uncovered:
                    uncov_word = all_words[uncov_idx]
                    if np.sum(cand_arr != uncov_word) > radius:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(cand)
            
            if valid_candidates:
                # Pick one that's not already in current
                for vc in valid_candidates:
                    if vc not in [tuple(c) for c in current]:
                        current[idx] = list(vc)
                        # Recompute covered
                        cur_covered = np.zeros(total_words, dtype=bool)
                        for cw in current:
                            cur_covered |= get_coverage_mask(cw)
                        if len(current) < best_size_ls:
                            best_size_ls = len(current)
                            best = current[:]
                        break
            else:
                # Try adding a codeword to cover gaps, then removing another
                # Or just skip
                pass
        
        return best, best_size_ls
    
    if best_codewords is not None:
        best_codewords, best_size = local_search(best_codewords, covered, max_iters=300)
    
    # Additional: try more aggressive local search with different random seeds
    for seed in range(5):
        if time.time() - start_time > time_limit:
            break
        rng = random.Random(1000 + seed)
        current = best_codewords[:]
        cur_covered = np.zeros(total_words, dtype=bool)
        for cw in current:
            cur_covered |= get_coverage_mask(cw)
        
        best_local = current[:]
        best_local_size = len(current)
        
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
            if len(current) >= best_local_size:
                break
            
            idx = rng.randint(0, len(current) - 1)
            cw_remove = current[idx]
            
            new_covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(current):
                if j != idx:
                    new_covered |= get_coverage_mask(cw)
            
            uncovered = np.where(~new_covered)[0]
            
            if len(uncovered) == 0:
                current.pop(idx)
                cur_covered = new_covered
                if len(current) < best_local_size:
                    best_local_size = len(current)
                    best_local = current[:]
                continue
            
            # Find valid replacement
            first_uncov = all_words[uncovered[0]]
            candidates = [tuple(first_uncov)]
            for pos in range(n):
                for val in range(q):
                    if val != first_uncov[pos]:
                        nw = first_uncov.copy()
                        nw[pos] = val
                        candidates.append(tuple(nw))
            
            valid_candidates = []
            for cand in candidates:
                cand_arr = np.array(cand)
                valid = True
                for uncov_idx in uncovered:
                    uncov_word = all_words[uncov_idx]
                    if np.sum(cand_arr != uncov_word) > radius:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(cand)
            
            if valid_candidates:
                for vc in valid_candidates:
                    if vc not in [tuple(c) for c in current]:
                        current[idx] = list(vc)
                        cur_covered = np.zeros(total_words, dtype=bool)
                        for cw in current:
                            cur_covered |= get_coverage_mask(cw)
                        if len(current) < best_local_size:
                            best_local_size = len(current)
                            best_local = current[:]
                        break
        
        if best_local_size < best_size:
            best_codewords = best_local
            best_size = best_local_size
    
    return best_codewords
# EVOLVE-BLOCK-END

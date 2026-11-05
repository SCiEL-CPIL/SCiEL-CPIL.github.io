"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Uses bitset-based local search (add/remove/swap) for the ternary case.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        return _greedy_construct_general(q, n, radius, deadline)
    
    return _construct_k3_6_1(deadline)


def _greedy_construct_general(q, n, radius, deadline):
    """General greedy set cover for covering codes (fallback)."""
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []
    
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_count = -1
        for w in uncovered:
            count = sum(1 for v in uncovered if hamming_dist(w, v) <= radius)
            if count > best_count:
                best_count = count
                best_word = w
        if best_word is None:
            break
        codewords.append(best_word)
        uncovered = {v for v in uncovered if hamming_dist(best_word, v) > radius}
    
    return [list(w) for w in codewords]


def _construct_k3_6_1(deadline):
    """Specialized construction for K_3(6,1) using optimized bitset local search."""
    start_time = time.time()
    
    # Precompute all 729 words as integers 0..728
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 729
    
    # Precompute balls as lists of indices
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    w2 = list(w)
                    w2[pos] = val
                    ball.append(word_to_idx[tuple(w2)])
        balls.append(ball)
    
    # Tetracode-based initial code (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Initial code set
    code_set = set(word_to_idx[tuple(cw)] for cw in init_codewords)
    
    # Coverage count array
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code_set:
        for w in balls[c]:
            cover_count[w] += 1
    
    best_size = len(code_set)
    best_code_set = set(code_set)
    
    # Function to compute critical counts for all c in code_set
    def get_critical_counts(code_set, cover_count):
        crit = {}
        for c in code_set:
            cnt = 0
            for w in balls[c]:
                if cover_count[w] == 1:
                    cnt += 1
            crit[c] = cnt
        return crit

    # Greedy deletion: remove all c with critical count 0
    def greedy_delete(code_set, cover_count):
        changed = True
        while changed:
            changed = False
            crit = get_critical_counts(code_set, cover_count)
            removable = [c for c, cnt in crit.items() if cnt == 0]
            if removable:
                for c in removable:
                    code_set.discard(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                changed = True
        return code_set, cover_count

    # Local search: try swaps to enable further deletions
    def local_search(code_set, cover_count, time_limit):
        nonlocal best_size, best_code_set
        while time.time() < time_limit:
            # 1. Greedy deletion
            code_set, cover_count = greedy_delete(code_set, cover_count)
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code_set = set(code_set)
            
            # Check if done
            if len(code_set) <= 71: # Lower bound
                break
            
            # 2. If no greedy deletions happened, try a swap
            # Check if any critical count is 0 (if so, greedy_delete would have removed them)
            # So now all critical counts >= 1
            
            # Find c with min critical count
            crit = get_critical_counts(code_set, cover_count)
            if not crit:
                break
            min_crit_val = min(crit.values())
            if min_crit_val == 0:
                # Should have been removed, but just in case
                continue
            
            # Pick a random c among those with min critical count
            candidates = [c for c, cnt in crit.items() if cnt == min_crit_val]
            c_remove = random.choice(candidates)
            
            # Identify U: words that would be uncovered
            U = [w for w in balls[c_remove] if cover_count[w] == 1]
            
            # Find best c_new
            # Score each c_new not in code_set by how many words in U it covers
            scores = np.zeros(N, dtype=np.int32)
            for w in U:
                for c_new in balls[w]:
                    scores[c_new] += 1
            
            # Find c_new with max score, not in code_set
            max_score = 0
            best_c_new = -1
            for c_new in range(N):
                if c_new in code_set:
                    continue
                if scores[c_new] > max_score:
                    max_score = scores[c_new]
                    best_c_new = c_new
            
            # If we found a c_new that covers all of U, swap is valid
            if best_c_new != -1 and max_score == len(U):
                # Perform swap
                code_set.discard(c_remove)
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                
                code_set.add(best_c_new)
                for w in balls[best_c_new]:
                    cover_count[w] += 1
                
                # Loop back to greedy delete
                
            else:
                # No valid 1-for-1 swap found.
                # We are stuck in a local minimum.
                break
        
        return code_set, cover_count

    # Main loop: Multiple restarts with different random coordinate permutations
    time_limit_total = deadline - 10
    restarts = 0
    max_restarts = 10 # Limited by time
    
    while time.time() < time_limit_total and restarts < max_restarts:
        # Randomly permute coordinates of the initial code
        perm = random.sample(range(6), 6)
        init_permuted = [ [cw[perm[i]] for i in range(6)] for cw in init_codewords ]
        
        code_set = set(word_to_idx[tuple(cw)] for cw in init_permuted)
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Run local search for a fraction of the time or until stuck
        # Give it up to 10 seconds per restart, or remaining time
        per_restart_limit = time.time() + 10
        if time_limit_total - time.time() < 10:
            per_restart_limit = time_limit_total
        
        code_set, cover_count = local_search(code_set, cover_count, per_restart_limit)
        
        # Final greedy deletion
        code_set, cover_count = greedy_delete(code_set, cover_count)
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code_set = set(code_set)
        
        restarts += 1
        
    # Convert back
    result = [list(all_words[i]) for i in best_code_set]
    
    # Verify
    # Re-check coverage for the best code set
    final_cover_count = np.zeros(N, dtype=np.int32)
    for c in best_code_set:
        for w in balls[c]:
            final_cover_count[w] += 1
    if not np.all(final_cover_count >= 1):
        # Fallback
        result = init_codewords
        
    return result
# EVOLVE-BLOCK-END

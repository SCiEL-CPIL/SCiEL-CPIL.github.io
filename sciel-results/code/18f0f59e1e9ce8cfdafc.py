"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius.
    
    Uses SA-based local search with numpy neighbor matrix for the ternary case.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        return _greedy_construct_general(q, n, radius, deadline)
    
    return _construct_k3_6_1_sa(deadline)


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


def _construct_k3_6_1_sa(deadline):
    """SA-based construction for K_3(6,1) using numpy neighbor matrix."""
    start_time = time.time()
    N = 729
    powers = [1, 3, 9, 27, 81, 243]
    
    # Precompute neighbors: each word connects to itself + all one-position changes
    num_neighbors = 1 + 6 * 2  # 13 neighbors per word
    neighbors = np.empty((N, num_neighbors), dtype=np.int32)
    for i in range(N):
        digits = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != digits[p]:
                    neighbors[i, k] = i + (alt - digits[p]) * powers[p]
                    k += 1
    
    # Tetracode-based initial code (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    def word_to_idx(w):
        idx = 0
        for p in range(6):
            idx += w[p] * powers[p]
        return idx
    
    # Convert initial codewords to indices
    init_indices = [word_to_idx(cw) for cw in init_codewords]
    
    best_size = N  # Start with worst case
    best_codewords = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Multiple restarts with different initializations
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # For restart 0, use tetracode; for others, use random or permuted
        if restart == 0:
            codewords = list(init_indices)
        else:
            # Random permutation of coordinates applied to tetracode
            perm = random.sample(range(6), 6)
            permuted = [[cw[perm[i]] for i in range(6)] for cw in init_codewords]
            codewords = [word_to_idx(cw) for cw in permuted]
        
        # Initialize coverage counts
        coverage = np.zeros(N, dtype=np.int32)
        in_set = set(codewords)
        for c in codewords:
            coverage[neighbors[c]] += 1
        
        # Greedy deletion first
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                if (coverage[neighbors[c]] >= 2).all():
                    codewords.pop(i)
                    in_set.discard(c)
                    coverage[neighbors[c]] -= 1
                    improved = True
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = list(codewords)
        
        # SA phase
        T = 1.0
        T_min = 0.001
        cooling = 0.9998
        steps = 0
        sa_deadline = deadline - 8
        
        while time.time() < sa_deadline:
            steps += 1
            
            # Try to find a redundant word to remove
            redundant_found = False
            for _ in range(3):  # Try a few random candidates
                if not codewords:
                    break
                idx = random.randint(0, len(codewords) - 1)
                c = codewords[idx]
                
                if (coverage[neighbors[c]] >= 2).all():
                    # Remove it
                    codewords.pop(idx)
                    in_set.discard(c)
                    coverage[neighbors[c]] -= 1
                    redundant_found = True
                    
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    break
            
            if redundant_found:
                T = max(T * cooling, T_min)
                continue
            
            # No redundant word found. Try a swap: remove one, add another
            # Pick a word with minimum critical count
            min_crit = float('inf')
            best_remove_idx = -1
            for i in range(len(codewords)):
                c = codewords[i]
                crit = (coverage[neighbors[c]] == 1).sum()
                if crit < min_crit:
                    min_crit = crit
                    best_remove_idx = i
            
            if min_crit == 0:
                # This word is redundant, should have been caught above
                continue
            
            c_remove = codewords[best_remove_idx]
            U = neighbors[c_remove][coverage[neighbors[c_remove]] == 1]
            
            if len(U) == 0:
                continue
            
            # Find best replacement: a word not in set that covers all of U
            # Score each candidate by how many of U it covers
            w0 = int(U[0])
            candidates = neighbors[w0]
            
            best_c_new = -1
            best_score = 0
            for cand in candidates:
                cand = int(cand)
                if cand == c_remove or cand in in_set:
                    continue
                score = sum(1 for w in U if w in neighbors[cand])
                if score > best_score:
                    best_score = score
                    best_c_new = cand
            
            if best_c_new == -1 or best_score < len(U):
                # No valid swap. Try a "move" that might temporarily increase size
                # Add a random word that helps most
                if random.random() < T:
                    # Accept a non-improving move: add a word
                    # Pick the word that covers the most uncovered-or-critical words
                    all_words_scores = np.zeros(N, dtype=np.int32)
                    for w in U:
                        for cand in neighbors[w]:
                            if cand not in in_set:
                                all_words_scores[cand] += 1
                    
                    # Also consider words that cover other critical words
                    for i in range(len(codewords)):
                        c = codewords[i]
                        for w in neighbors[c][coverage[neighbors[c]] == 1]:
                            for cand in neighbors[w]:
                                if cand not in in_set:
                                    all_words_scores[cand] += 1
                    
                    nonzero = all_words_scores[all_words_scores > 0]
                    if len(nonzero) > 0:
                        # Pick randomly among top scorers
                        max_s = nonzero.max()
                        top_candidates = np.where(all_words_scores == max_s)[0]
                        c_add = int(random.choice(top_candidates))
                        
                        codewords.append(c_add)
                        in_set.add(c_add)
                        coverage[neighbors[c_add]] += 1
                        T = max(T * cooling, T_min)
                        continue
            
            # Perform the swap
            codewords[best_remove_idx] = best_c_new
            in_set.discard(c_remove)
            in_set.add(best_c_new)
            coverage[neighbors[c_remove]] -= 1
            coverage[neighbors[best_c_new]] += 1
            
            T = max(T * cooling, T_min)
        
        # Final greedy deletion after SA
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                if (coverage[neighbors[c]] >= 2).all():
                    codewords.pop(i)
                    in_set.discard(c)
                    coverage[neighbors[c]] -= 1
                    improved = True
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = list(codewords)
        
        if best_size <= 71:
            break
    
    # Convert back to list of lists
    if best_codewords is None:
        best_codewords = init_indices
    
    result = []
    for idx in best_codewords:
        word = [(idx // powers[p]) % 3 for p in range(6)]
        result.append(word)
    
    # Verify coverage
    final_cover = np.zeros(N, dtype=np.int32)
    for idx in best_codewords:
        final_cover[neighbors[idx]] += 1
    
    if not (final_cover >= 1).all():
        # Fallback to tetracode
        result = init_codewords
    
    return result
# EVOLVE-BLOCK-END

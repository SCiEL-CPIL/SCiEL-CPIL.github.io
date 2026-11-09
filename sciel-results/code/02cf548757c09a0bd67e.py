"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product


def _to_int(w, q):
    v = 0
    for x in w:
        v = v * q + x
    return v


def _to_word(i, q, n):
    w = [0] * n
    for j in range(n - 1, -1, -1):
        w[j] = i % q
        i //= q
    return w


def _build_incidence(q, n, radius):
    """
    Build neighbor lists: neighbor_map[cw] = array of all words within Hamming distance <= radius.
    Uses vectorized numpy operations for speed.
    """
    total_space = q ** n
    
    # Precompute all patterns: list of (positions_tuple, deltas_tuple)
    patterns = [([], ())]  # radius 0
    for r in range(1, radius + 1):
        for pos in combinations(range(n), r):
            for deltas in product(range(1, q), repeat=r):
                patterns.append((pos, deltas))
    
    # Convert all words to a 2D array: shape (total_space, n)
    all_words = np.zeros((total_space, n), dtype=np.int32)
    for i in range(total_space):
        w = _to_word(i, q, n)
        all_words[i] = w
    
    # For each pattern, compute neighbor indices vectorized
    # We'll build neighbor sets as sorted arrays
    neighbor_map = [None] * total_space
    for i in range(total_space):
        nbrs = {i}  # include self
        for pos, deltas in patterns:
            if not pos:
                continue
            # For each position in pos, the value changes by delta
            # We need to compute the new word
            w = all_words[i].copy()
            for p, d in zip(pos, deltas):
                w[p] = (w[p] + d) % q
            # Convert back to integer
            ni = 0
            for j in range(n):
                ni = ni * q + w[j]
            nbrs.add(ni)
        neighbor_map[i] = np.array(sorted(nbrs), dtype=np.int32)
    
    return neighbor_map


def _direct_sum_code(q, n, radius):
    """
    Construct the direct-sum covering code.
    For q=3, n=8, radius=1: use two blocks of length 4.
    The (4,1) ternary covering code has 9 words (the complete code of length 4 over 3 symbols
    with radius 1 is actually not 9... let me think).
    
    Actually, K_3(4,1) = 9? Let's verify: 3^4 = 81 words. Each codeword covers 
    1 + 4*2 = 9 words. So lower bound is 81/9 = 9. And indeed the complete code works:
    all words of the form (a, a, a, a) for a in {0,1,2} plus... no that's only 3.
    
    Actually for K_3(4,1): the code {0000, 1111, 2222, 0120, 1021, 2102, ...} 
    Hmm, let me just use a known construction.
    
    The direct sum: if C1 is a (n1, r1) code and C2 is a (n2, r2) code with r1+r2=r,
    then C1 x C2 is a (n1+n2, r) code with |C1|*|C2| words.
    
    For K_3(8,1): we can use K_3(4,0) x K_3(4,1) or K_3(4,1) x K_3(4,1).
    K_3(4,0) = 3^4 = 81 (trivial, radius 0 means each word covers only itself)
    K_3(4,1) = 9 (I believe this is correct: 81/9=9 and the code exists)
    
    So K_3(8,1) <= 81 * 9 = 729? That's worse than 486.
    
    Better: K_3(4,1) x K_3(4,1) gives 9*9=81 words for radius 2, not radius 1.
    
    For radius 1: we need r1+r2=1, so one is 0 and one is 1.
    K_3(4,0) * K_3(4,1) = 81 * 9 = 729.
    
    Hmm, that's not 486. Let me reconsider.
    
    Actually, the 486-word code by van Laarhoven et al. might not be a simple direct sum.
    Let me just use a greedy approach with a good initial solution.
    
    For now, let me just return an empty list and rely on greedy.
    """
    return []


def _greedy_cover(neighbor_map, total_space, deadline, seed=42):
    """Greedy set cover with efficient updates using precomputed reverse neighbor lists."""
    # Build reverse neighbor lists: for each word w, which codewords cover it
    # This is the same as neighbor_map since the relation is symmetric
    # But we need to track uncovered count per potential codeword
    
    # uncovered_count[cw] = number of words in neighbor_map[cw] that are not yet covered
    uncovered_count = np.zeros(total_space, dtype=np.int32)
    for cw in range(total_space):
        uncovered_count[cw] = len(neighbor_map[cw])
    
    # covered[w] = True if word w has been covered by at least one codeword
    covered = np.zeros(total_space, dtype=bool)
    
    codewords = []
    
    while True:
        if time.time() >= deadline - 5.0:
            break
        
        best_cw = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_cw]
        if best_count == 0:
            break
        
        codewords.append(best_cw)
        
        # Update: for each word w in neighbor_map[best_cw] that is now covered,
        # decrement uncovered_count for all codewords that cover w
        for w in neighbor_map[best_cw]:
            if not covered[w]:
                covered[w] = True
                # Decrement uncovered_count for all codewords that cover w
                # These are all codewords in neighbor_map[w] (symmetric relation)
                for cw2 in neighbor_map[w]:
                    uncovered_count[cw2] -= 1
    
    return codewords


def _local_search_reduce(codewords, neighbor_map, total_space, deadline):
    """Try to remove redundant codewords. Vectorized counting."""
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        # Compute coverage counts vectorized
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            counts[neighbor_map[cw]] += 1
        
        # Find removable codewords: those where all words they cover have count >= 2
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() >= deadline - 2.0:
                break
            cw = codewords[i]
            # Check if all words in neighbor_map[cw] have count >= 2
            if np.all(counts[neighbor_map[cw]] >= 2):
                codewords.pop(i)
                changed = True
                break
    return codewords


def _local_search_swap(codewords, neighbor_map, total_space, deadline, max_iterations=200):
    """
    1-exchange local search with vectorized operations.
    """
    rng = random.Random(12345)
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 3.0 and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Compute coverage counts vectorized
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            counts[neighbor_map[cw]] += 1
        
        # Find critical words (covered exactly once)
        critical_mask = (counts == 1)
        critical_words = np.where(critical_mask)[0]
        
        if len(critical_words) == 0:
            # No critical words, try removal
            for i in range(len(codewords) - 1, -1, -1):
                cw = codewords[i]
                if np.all(counts[neighbor_map[cw]] >= 2):
                    codewords.pop(i)
                    improved = True
                    break
            continue
        
        # For each codeword, find which critical words it covers
        for i in range(len(codewords)):
            if time.time() >= deadline - 3.0:
                break
            cw = codewords[i]
            cw_neighbors = neighbor_map[cw]
            # Critical words covered by this codeword
            cw_critical = cw_neighbors[critical_mask[cw_neighbors]]
            
            if len(cw_critical) == 0:
                # This codeword doesn't cover any critical words, try to remove it
                if np.all(counts[cw_neighbors] >= 2):
                    codewords.pop(i)
                    improved = True
                    break
                continue
            
            # Try to find a replacement codeword that covers all critical words
            # of this codeword and doesn't make things worse
            target_w = int(rng.choice(cw_critical))
            # Find all codewords that cover target_w (neighbors of target_w)
            candidates = neighbor_map[target_w]
            
            if len(candidates) == 0:
                continue
            
            # Filter out current codeword
            candidates = [c for c in candidates if c != cw]
            
            if not candidates:
                continue
            
            # Score each candidate
            best_candidate = None
            best_score = 0
            for cand in candidates[:80]:  # Limit for speed
                cand_neighbors = neighbor_map[cand]
                # Check if cand covers all of cw's critical words
                if not np.all(critical_mask[cand_neighbors][np.isin(cand_neighbors, cw_critical)]):
                    # More efficient: check if all cw_critical are in cand_neighbors
                    if not np.all(np.isin(cw_critical, cand_neighbors)):
                        continue
                
                # Check no newly uncovered: words in cw_neighbors with count==1 not in cand_neighbors
                cw_only_critical = cw_neighbors[(counts[cw_neighbors] == 1)]
                if len(cw_only_critical) > 0:
                    if not np.all(np.isin(cw_only_critical, cand_neighbors)):
                        continue
                
                # Score: number of critical words cand covers (higher is better)
                score = np.sum(critical_mask[cand_neighbors])
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate is not None:
                codewords[i] = best_candidate
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    total_space = q ** n
    
    # Build neighbor map
    neighbor_map = _build_incidence(q, n, radius)
    
    # Phase 1: Multiple greedy covers with different seeds
    best_codewords = None
    best_len = float('inf')
    
    seeds = [42, 1, 2, 3, 7, 13, 100, 999, 123, 456, 789, 2024, 555, 111, 9999]
    for seed in seeds:
        if time.time() >= deadline - 20.0:
            break
        cw = _greedy_cover(neighbor_map, total_space, deadline, seed=seed)
        cw = _local_search_reduce(cw, neighbor_map, total_space, deadline)
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
    
    # Phase 2: Local search swap on the best solution
    if time.time() < deadline - 15.0:
        best_codewords = _local_search_swap(best_codewords, neighbor_map, total_space, deadline, max_iterations=150)
    
    # Phase 3: Final reduction pass
    best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Phase 4: Another round of swaps if time permits
    if time.time() < deadline - 8.0:
        best_codewords = _local_search_swap(best_codewords, neighbor_map, total_space, deadline, max_iterations=80)
        best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Phase 5: One more swap+reduce cycle if time permits
    if time.time() < deadline - 4.0:
        best_codewords = _local_search_swap(best_codewords, neighbor_map, total_space, deadline, max_iterations=40)
        best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Convert back to lists
    result = [_to_word(cw, q, n) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END

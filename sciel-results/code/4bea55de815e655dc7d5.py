"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from collections import deque

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a vectorized greedy construction followed by aggressive redundancy removal
    and swap-based local search.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    global_start = time.time()
    HARD_DEADLINE = 115.0  # Safety margin for 120s limit
    
    # --- Precomputation: Fast integer <-> word conversion and neighbor generation ---
    
    def int_to_word(val, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute the "delta" values for single-position changes
    # For each position i, and each value change, what is the integer difference?
    # This allows O(1) neighbor generation for radius 1.
    
    # neighbors[w] will be a list of ints
    neighbors = [None] * total_words
    
    # We can generate neighbors more efficiently using bit-manipulation-like logic for base-q
    # But for q=3, n=8, total_words=6561, even O(n*q) per word is fine.
    # Let's precompute the neighbors for all words.
    
    # Optimization: Precompute the "step" values for each position
    # step[i] = q^i
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i-1] * q
        
    for w in range(total_words):
        # Convert to word once
        word = [0] * n
        tmp = w
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
            
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                step_val = steps[pos]
                # Changing value at pos by delta changes w by delta * step_val
                # But we must handle wrap-around in base q? No, w is an integer representation.
                # w = sum(word[i] * steps[i])
                # new_w = w - orig_val*step_val + new_val*step_val
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = w + (new_val - orig_val) * step_val
                    neigh.append(new_w)
        
        if radius >= 2:
            # Add pairs. This is more complex.
            # For each pair of positions (p1, p2)
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = word[pos1]
                    v2 = word[pos2]
                    s1 = steps[pos1]
                    s2 = steps[pos2]
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            # Check if this word was already added by a single step?
                            # It's possible if q=2, but here q=3.
                            # A double change is distinct from single changes.
                            # However, we might add duplicates if we're not careful?
                            # No, (nv1, nv2) is unique.
                            # But we should avoid adding w itself (which is nv1==v1 and nv2==v2, skipped).
                            new_w = w + (nv1 - v1) * s1 + (nv2 - v2) * s2
                            # We need to ensure we don't add duplicates in the list.
                            # Since we iterate over unique pairs of (pos1, pos2, nv1, nv2),
                            # and the mapping to new_w is injective for fixed w?
                            # Yes, because the base-q representation is unique.
                            neigh.append(new_w)
        
        neighbors[w] = neigh

    # --- Greedy Initialization ---
    
    def greedy_init():
        covered = np.zeros(total_words, dtype=np.bool_)
        codewords = []
        uncovered_count = total_words
        
        # To speed up greedy, we can maintain a list of uncovered words.
        # But removing from a list is O(N).
        # Alternative: Just iterate over all words, but skip those already covered?
        # Or maintain a set of uncovered words?
        
        # For N=6561, iterating over all words each time is 6561 * K operations.
        # If K ~ 500, that's ~3M operations. Fast in Python? Maybe borderline.
        # Let's try a "candidate" approach:
        # Pick a random uncovered word u.
        # Find the word w that covers the most uncovered words among its neighbors.
        # This is the standard greedy.
        
        # To find "most uncovered neighbors", we need to count uncovered neighbors for each candidate.
        # Candidates can be restricted to neighbors of some uncovered word?
        # Yes, if a word w covers an uncovered word u, then w is a neighbor of u.
        # So we can pick a random uncovered word u, look at its neighbors (candidates),
        # and pick the one with max coverage of uncovered words.
        
        uncovered_set = set(range(total_words))
        
        while uncovered_set:
            # Pick a random uncovered word
            u = random.choice(list(uncovered_set))
            
            # Candidates are neighbors of u
            candidates = neighbors[u]
            
            best_cand = -1
            best_cov = -1
            
            for c in candidates:
                # Count how many neighbors of c are in uncovered_set
                # This is the expensive part.
                # We can optimize by pre-computing neighbor sets as numpy arrays?
                # Or just use a set intersection?
                # len(uncovered_set & set(neighbors[c]))
                
                # Since neighbors[c] is a list, converting to set every time is slow.
                # Let's precompute neighbor sets?
                # No, memory might be an issue or time.
                # Let's just iterate.
                cov = 0
                for v in neighbors[c]:
                    if v in uncovered_set:
                        cov += 1
                
                if cov > best_cov:
                    best_cov = cov
                    best_cand = c
            
            if best_cand == -1:
                break
            
            codewords.append(best_cand)
            
            # Update uncovered_set
            for v in neighbors[best_cand]:
                if v in uncovered_set:
                    uncovered_set.remove(v)
            
            # Early exit if time is tight
            if time.time() - global_start > HARD_DEADLINE - 10:
                break
        
        return codewords

    # --- Optimization: Redundancy Removal and Swaps ---
    
    def optimize(codewords, time_limit):
        # Use numpy array for coverage counts
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            # Vectorized addition?
            # neighbors[cw] is a list. We can convert to array.
            # But doing this for each cw is slow.
            # Let's stick to loops but use numpy for the array itself.
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        stuck_count = 0
        last_size = len(codewords)
        
        # Precompute neighbor sets for faster intersection?
        # neighbor_sets = [set(neighbors[w]) for w in range(total_words)]
        # This is expensive to build but fast to query.
        # For 6561 words, and ~25 neighbors each, it's manageable.
        # Let's do it once outside the loop if possible, or here.
        
        while time.time() - start_time < time_limit:
            # 1. Try to remove redundant codewords
            if try_remove_redundant():
                stuck_count = 0
                last_size = len(codewords)
                continue
            
            if len(codewords) == 0:
                break
            
            # 2. Identify codewords with unique coverage (cannot be removed)
            # And try to swap them with better codewords.
            
            # Find codewords that have at least one uniquely covered word
            unique_cw_indices = []
            for i, cw in enumerate(codewords):
                has_unique = False
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        has_unique = True
                        break
                if has_unique:
                    unique_cw_indices.append(i)
            
            if unique_cw_indices:
                # Pick one randomly
                idx = random.choice(unique_cw_indices)
                old_cw = codewords[idx]
                
                # Find words that are uniquely covered by old_cw
                would_uncover = []
                for u in neighbors[old_cw]:
                    if coverage_count[u] == 1:
                        would_uncover.append(u)
                
                # Candidates: Words that cover all would_uncover words
                if not would_uncover:
                    # This shouldn't happen if has_unique is True, but just in case
                    candidates = set(neighbors[old_cw]) - cw_set
                else:
                    # Intersection of neighbor sets of all would_uncover words
                    # Start with the smallest set?
                    # Sort by degree?
                    # For now, just intersect.
                    candidates = set(neighbors[would_uncover[0]])
                    for u in would_uncover[1:]:
                        candidates &= set(neighbors[u])
                        if not candidates:
                            break
                
                # Remove old_cw and existing codewords from candidates
                candidates -= cw_set
                candidates.discard(old_cw)
                
                if candidates:
                    # Score candidates: Minimize "new" uncovered words?
                    # Or maximize overlap with existing coverage?
                    # We want to replace old_cw with new_cw such that:
                    # 1. All would_uncover are still covered (guaranteed by construction).
                    # 2. The new_cw doesn't introduce too many "uncovered" words?
                    #    Actually, we want to maximize the number of words that are ALREADY covered
                    #    by other codewords, to increase redundancy and allow future removals.
                    
                    best_new_cw = -1
                    best_score = -1
                    
                    # Sample candidates if too many
                    cand_list = list(candidates)
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    for new_cw in cand_list:
                        # Score: number of neighbors of new_cw that are covered by >= 2 codewords
                        # (i.e., redundant coverage)
                        score = 0
                        for u in neighbors[new_cw]:
                            # coverage_count includes old_cw's contribution
                            # We need to see coverage WITHOUT old_cw
                            # If u is in would_uncover, its coverage drops by 1.
                            # If u is not in would_uncover, its coverage stays same (or increases if new_cw covers it).
                            # Let's just count how many neighbors of new_cw have coverage >= 2
                            # after removing old_cw and adding new_cw.
                            # This is complex.
                            
                            # Simpler heuristic: Maximize overlap with current covered set.
                            # Or minimize "loss" of coverage.
                            
                            # Let's try: Score = sum(1 for u in neighbors[new_cw] if coverage_count[u] > 1)
                            # This favors codewords that sit on top of other codewords.
                            if coverage_count[u] > 1:
                                score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_new_cw = new_cw
                    
                    if best_new_cw != -1:
                        # Perform swap
                        for u in neighbors[old_cw]:
                            coverage_count[u] -= 1
                        for u in neighbors[best_new_cw]:
                            coverage_count[u] += 1
                        codewords[idx] = best_new_cw
                        cw_set.discard(old_cw)
                        cw_set.add(best_new_cw)
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    stuck_count += 1
            else:
                # No unique coverage. Try to find a swap that reduces size?
                # Or just random walk?
                # If no unique coverage, we can try to remove any codeword?
                # But try_remove_redundant already did that.
                # So we are stuck.
                
                # Try a random swap to escape local minimum
                if len(codewords) > 0:
                    idx = random.randint(0, len(codewords) - 1)
                    old_cw = codewords[idx]
                    
                    # Find a random word that covers all words uniquely covered by old_cw
                    # (which is none, so any word is a candidate?)
                    # No, we must maintain coverage.
                    # If no word is uniquely covered by old_cw, then removing old_cw is safe.
                    # But try_remove_redundant failed, meaning some word IS uniquely covered.
                    # Contradiction?
                    # Ah, unique_cw_indices is empty means NO codeword has a uniquely covered word.
                    # This means every word is covered by at least 2 codewords.
                    # Then we CAN remove any codeword.
                    # Why did try_remove_redundant fail?
                    # Maybe it didn't check all?
                    # try_remove_redundant checks if coverage_count[u] == 1 for any neighbor.
                    # If unique_cw_indices is empty, then for all cw, for all u in neighbors[cw], coverage_count[u] != 1.
                    # So coverage_count[u] >= 2 for all u.
                    # Then try_remove_redundant SHOULD have removed one.
                    
                    # Unless... the sets are inconsistent?
                    # Let's just try to remove a random one here.
                    
                    # Actually, if unique_cw_indices is empty, we can remove ANY codeword.
                    # Let's remove a random one to reduce size.
                    
                    # Wait, if we remove one, we might create unique coverages.
                    # Let's just do it.
                    
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    stuck_count = 0
                else:
                    stuck_count += 1
            
            # Aggressive perturbation if stuck
            if stuck_count > 20:
                # Remove a random codeword and re-add?
                # Or just reset?
                # Let's try removing a random codeword and see if we can recover.
                if len(codewords) > 5:
                    idx = random.randint(0, len(codewords) - 1)
                    old_cw = codewords[idx]
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    stuck_count = 0
                else:
                    stuck_count = 0 # Reset to avoid infinite loop
        
        # Final polish
        for _ in range(50):
            if not try_remove_redundant():
                break
        
        return codewords

    # --- Main Loop ---
    
    best_codewords = []
    
    # Do a few restarts
    # Time budget: 115s
    # Each restart: greedy + optimize
    
    while time.time() - global_start < HARD_DEADLINE - 5:
        # Greedy
        cw = greedy_init()
        
        # Optimize
        # Allocate remaining time
        elapsed = time.time() - global_start
        opt_time = max(1, HARD_DEADLINE - elapsed - 2)
        
        if opt_time < 2:
            break
            
        cw = optimize(cw, opt_time)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        # Check time
        if time.time() - global_start > HARD_DEADLINE - 5:
            break
    
    # Fallback if no result
    if not best_codewords:
        cw = greedy_init()
        best_codewords = cw
    
    # Verify coverage
    covered = np.zeros(total_words, dtype=np.bool_)
    for cw in best_codewords:
        for u in neighbors[cw]:
            covered[u] = True
    
    if not all(covered):
        # Try to fix? Or just return?
        # If invalid, we might get 0 score.
        # Let's try a quick fix: add words to cover uncovered ones.
        uncovered = [i for i in range(total_words) if not covered[i]]
        for u in uncovered:
            # Add a neighbor of u
            # Pick the first one
            # This might be inefficient but ensures validity
            for v in neighbors[u]:
                if v not in best_codewords:
                    best_codewords.append(v)
                    for uu in neighbors[v]:
                        covered[uu] = True
                    break
    
    # Convert to words
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    
    # Ensure distinctness (should be, but just in case)
    seen = set()
    final_result = []
    for w in result:
        if w not in seen:
            seen.add(w)
            final_result.append(w)
            
    return final_result
# EVOLVE-BLOCK-END

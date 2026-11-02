# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    For the specific case q=3, n=7, radius=1, we use a specialized approach:
    1. Greedy set-cover initialization using fast bitset operations.
    2. Local search with redundancy removal and swap moves (tunneling across local minima).
    3. Simulated annealing style restarts if time permits.
    For other parameters, we fall back to a generic greedy + reduction approach.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    else:
        return _generic_construct(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """
    Specialized solver for q=3, n=7, radius=1.
    Uses greedy set cover with bitsets, then local search with swaps.
    """
    q, n, radius = 3, 7, 1
    total_words = q ** n  # 2187
    
    # Generate all words as tuples of 7 digits in {0,1,2}
    all_words = list(itertools.product(range(q), repeat=n))
    # Map each word to its index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball contents) for each word
    # Ball size is 1 + 7*2 = 15
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[w]]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val != orig_val:
                    w_list[pos] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val
        neighbors.append(covered)
    
    # Precompute coverers: coverers[point] = list of codeword indices that cover this point
    coverers = [[] for _ in range(total_words)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)
            
    # Convert neighbors to numpy arrays for faster vector ops
    neighbors_np = [np.array(b, dtype=np.int16) for b in neighbors]
    
    # 1. Greedy Set Cover using Bitsets
    ball_bitsets = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << total_words) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(total_words, dtype=bool)
    
    # Greedy Phase
    while current_cover_mask != all_mask:
        if time.time() > deadline - 60:
            break
            
        best_i = -1
        best_count = 0
        
        # Optimization: Only scan unchosen candidates
        # For speed, we can scan all, but skip chosen. 
        # N=2187, scanning all is ~2000 ops. 200 steps -> 400k ops. Fine.
        for i in range(total_words):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
                    
        if best_i == -1:
            break # Should not happen if greedy is correct
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure validity (safety net)
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, total_words-1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    # cover_count[point] = number of codewords covering this point
    cover_count = np.zeros(total_words, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors_np[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Helper: Try to remove redundant codewords
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            b = neighbors_np[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop with Swaps (Quantum Tunneling Analogy)
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 100
    temperature = 1.0 # For simulated annealing restarts
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart loop after modification
        
        # 2. Try Swaps
        # Pick a random codeword to try swapping
        if codewords:
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors_np[old_cw]
            critical_points = []
            for pt in b_old:
                if cover_count[pt] == 1:
                    critical_points.append(pt)
            
            if not critical_points:
                # Should have been removed by redundancy check
                # But if it wasn't (e.g. time limit), force remove
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            # Find candidates that cover ALL critical points
            # candidates = intersection of coverers[p] for p in critical_points
            # Start with coverers of the first critical point
            candidate_set = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set.intersection_update(coverers[p])
            
            # Remove old_cw and existing codewords from candidates
            candidate_set.discard(old_cw)
            for cw in codewords:
                candidate_set.discard(cw)
                
            if candidate_set:
                # Pick the best candidate
                # Heuristic: Pick the one with the lowest "cost"
                # Cost: sum of cover_count in its ball (lower is better, less redundancy)
                # Or simply pick random to explore
                candidates_list = list(candidate_set)
                random.shuffle(candidates_list)
                
                best_new_cw = -1
                min_cost = float('inf')
                
                # Limit candidates to check to avoid O(N) scan if set is huge
                # But intersection of coverers is usually small.
                for new_cw in candidates_list[:20]: # Check up to 20 candidates
                    b_new = neighbors_np[new_cw]
                    # Cost: How many new redundancies does this create?
                    # If cover_count[pt] is high, it's "cheap" to add.
                    # If cover_count[pt] is 1, it's "expensive" (we are stealing critical coverage? No, we are adding).
                    # Actually, we are replacing old_cw with new_cw.
                    # The critical points of old_cw are covered by new_cw.
                    # Other points in new_cw's ball will have their count increased by 1.
                    # Points in old_cw's ball (except criticals) will have their count decreased by 1.
                    
                    # Let's just pick the first valid one or random.
                    # A better heuristic: pick the one that covers the most points with count==1 
                    # (other than the criticals) to potentially enable future removals? 
                    # Or pick the one that minimizes the max count in its ball?
                    
                    # For now, random selection from candidates is a good "tunneling" move.
                    best_new_cw = new_cw
                    break
                
                if best_new_cw != -1:
                    # Perform swap
                    # Remove old_cw
                    cover_count[neighbors_np[old_cw]] -= 1
                    codewords[idx] = best_new_cw
                    # Add new_cw
                    cover_count[neighbors_np[best_new_cw]] += 1
                    is_chosen[old_cw] = False
                    is_chosen[best_new_cw] = True
                    improved = True
                    no_improvement_count = 0
                    continue
        
        # 3. If no improvement, increase stagnation counter
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart / Perturbation
                # Randomly remove a few non-critical codewords and re-greedy
                if codewords:
                    # Find non-critical codewords
                    removable = []
                    for i, cw in enumerate(codewords):
                        b = neighbors_np[cw]
                        if np.min(cover_count[b]) > 1:
                            removable.append(i)
                    
                    if removable:
                        # Remove 1-3 random non-critical codewords
                        num_remove = min(len(removable), random.randint(1, 3))
                        to_remove = random.sample(removable, num_remove)
                        to_remove.sort(reverse=True)
                        for i in to_remove:
                            cw = codewords[i]
                            cover_count[neighbors_np[cw]] -= 1
                            codewords.pop(i)
                            is_chosen[cw] = False
                        
                        # Re-greedy fill
                        current_cover_mask = 0
                        for cw in codewords:
                            current_cover_mask |= ball_bitsets[cw]
                        
                        while current_cover_mask != all_mask:
                            if time.time() > deadline - 10:
                                break
                            best_i = -1
                            best_count = 0
                            for i in range(total_words):
                                if is_chosen[i]:
                                    continue
                                new_bits = ball_bitsets[i] & ~current_cover_mask
                                if new_bits:
                                    count = new_bits.bit_count()
                                    if count > best_count:
                                        best_count = count
                                        best_i = i
                            if best_i == -1:
                                break
                            codewords.append(best_i)
                            is_chosen[best_i] = True
                            current_cover_mask |= ball_bitsets[best_i]
                            
                        # Update cover_count
                        cover_count = np.zeros(total_words, dtype=np.int16)
                        for cw in codewords:
                            cover_count[neighbors_np[cw]] += 1
                        
                        no_improvement_count = 0
                        max_no_improvement = 50 # Reset patience
                        temperature *= 0.9 # Cool down

    # Final redundancy removal
    try_remove_redundant(codewords, cover_count)
    
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Convert chosen indices back to codewords
    result = [list(all_words[idx]) for idx in best_codewords]
    
    # Ensure distinctness
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
            
    return final_result

def _generic_construct(q, n, radius, deadline):
    """
    Generic greedy + reduction for arbitrary q, n, radius.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    ball_members = []
    for i in range(total_words):
        w = all_words[i]
        members = set()
        # Enumerate all words within Hamming distance `radius`
        for dist in range(radius + 1):
            for positions in itertools.combinations(range(n), dist):
                for deltas in itertools.product(range(1, q), repeat=dist):
                    new_word = list(w)
                    for pos, d in zip(positions, deltas):
                        new_word[pos] = (w[pos] + d) % q
                    members.add(word_to_idx[tuple(new_word)])
        ball_members.append(list(members))
    
    # Greedy set cover
    uncovered_set = set(range(total_words))
    chosen = []
    
    while uncovered_set and time.time() < deadline - 30:
        best_ball = -1
        best_count = -1
        for i in range(total_words):
            count = sum(1 for m in ball_members[i] if m in uncovered_set)
            if count > best_count:
                best_count = count
                best_ball = i
        if best_ball == -1 or best_count <= 0:
            break
        chosen.append(best_ball)
        for m in ball_members[best_ball]:
            if m in uncovered_set:
                uncovered_set.discard(m)
        if not uncovered_set:
            break
    
    # Reduction
    coverage = [[] for _ in range(total_words)]
    for ci, ball_idx in enumerate(chosen):
        for m in ball_members[ball_idx]:
            coverage[m].append(ci)
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for ci in range(len(chosen) - 1, -1, -1):
            redundant = True
            for m in ball_members[chosen[ci]]:
                if len(coverage[m]) == 1 and coverage[m][0] == ci:
                    redundant = False
                    break
            if redundant:
                for m in ball_members[chosen[ci]]:
                    coverage[m].remove(ci)
                chosen.pop(ci)
                changed = True
    
    result = [list(all_words[idx]) for idx in chosen]
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
    return final_result
# EVOLVE-BLOCK-END

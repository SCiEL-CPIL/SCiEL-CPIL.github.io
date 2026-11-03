import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius, start_time, deadline)
    
    # Specific optimization for q=3, n=6, r=1
    total_words = 3 ** n
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * 3 + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute all neighbor indices for each word (13 neighbors: self + 12 at distance 1)
    neighbors = np.empty((total_words, 13), dtype=np.int32)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                nb_list.append(word_to_idx(tuple(new_w_list)))
        neighbors[idx] = nb_list
    
    # Precompute reverse neighbors: for each word, which words have it as a neighbor
    # This helps find candidate replacements quickly
    reverse_neighbors = [list() for _ in range(total_words)]
    for idx in range(total_words):
        for nb in neighbors[idx]:
            reverse_neighbors[nb].append(idx)
    # Convert to numpy arrays for fast access
    reverse_neighbors_arr = [np.array(rn, dtype=np.int32) for rn in reverse_neighbors]
    
    # Generate initial code using distance-2 independent set approach
    # Two codewords at Hamming distance 2 can be replaced by their 2 neighbors at distance 1
    # (the two words that differ from both in exactly 1 position), so we want to avoid
    # pairs at distance 2 in our code.
    
    # Build distance-2 adjacency: which words are at Hamming distance exactly 2 from each word
    dist2_adj = [set() for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                # Change positions pos1 and pos2
                for v1 in range(q):
                    if v1 == w[pos1]:
                        continue
                    for v2 in range(q):
                        if v2 == w[pos2]:
                            continue
                        new_w = list(w)
                        new_w[pos1] = v1
                        new_w[pos2] = v2
                        nb_idx = word_to_idx(tuple(new_w))
                        dist2_adj[idx].add(nb_idx)
                        dist2_adj[nb_idx].add(idx)
    
    # Convert to numpy arrays for fast access
    dist2_adj_arr = [np.array(sorted(s), dtype=np.int32) for s in dist2_adj]
    
    # Greedy independent set on distance-2 graph:
    # We want a large set of words such that no two are at distance 2.
    # Start with all words, greedily remove to make independent set.
    # Actually, we want a COVERING code, so we need every word within distance 1.
    # Strategy: start with all 729 words (full cover), then greedily remove words
    # while maintaining coverage AND maintaining distance-2 independence.
    
    # Better strategy: use the tetracode-based seed but also try greedy from scratch
    # with distance-2 constraint.
    
    # Approach: Greedy set cover with distance-2 independence constraint
    # Start empty, add words that cover most uncovered, preferring those not at dist 2 from existing
    
    # Actually, let's try a different approach:
    # 1. Start with tetracode seed (81 words)
    # 2. Remove pairs at distance 2 by replacing with better alternatives
    # 3. Then do local search
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Use NumPy for coverage tracking
    coverage = np.zeros(total_words, dtype=np.int32)
    current_code = list(seed_codewords)
    current_code_idx = [word_to_idx(cw) for cw in current_code]
    
    for cw_idx in current_code_idx:
        coverage[neighbors[cw_idx]] += 1
    
    # Phase 1: Remove distance-2 conflicts
    # For each pair at distance 2, try to eliminate one by finding a replacement
    # that covers what it uniquely covers, without creating new distance-2 conflicts
    dist2_conflict_removed = True
    max_conflict_rounds = 50
    for _ in range(max_conflict_rounds):
        if time.time() > deadline:
            break
        dist2_conflict_removed = False
        
        code_set = set(current_code_idx)
        
        # Find all distance-2 pairs in current code
        pairs_to_fix = []
        for i, cw_idx in enumerate(current_code_idx):
            for j in range(i + 1, len(current_code_idx)):
                other_idx = current_code_idx[j]
                if other_idx in dist2_adj[cw_idx]:
                    pairs_to_fix.append((i, j))
        
        if not pairs_to_fix:
            break
        
        # For each conflicting pair, try to remove the less "useful" one
        # and replace with a word that covers its unique coverage
        for (i, j) in pairs_to_fix:
            if time.time() > deadline:
                break
            if i >= len(current_code_idx) or j >= len(current_code_idx):
                continue
            
            idx_i = current_code_idx[i]
            idx_j = current_code_idx[j]
            
            # Check if they're still at distance 2 (might have been removed)
            if idx_j not in dist2_adj[idx_i]:
                continue
            
            # Try removing i first
            coverage[neighbors[idx_i]] -= 1
            uncovered_i = np.where(coverage == 0)[0]
            
            if len(uncovered_i) == 0:
                # i is redundant, remove it
                current_code.pop(i)
                current_code_idx.pop(i)
                dist2_conflict_removed = True
                continue
            else:
                # Find replacement for i that doesn't conflict with j
                best_rep = -1
                best_covers = 0
                candidate_reps = set()
                for uidx in uncovered_i[:20]:
                    for r in reverse_neighbors_arr[uidx]:
                        if r not in code_set and r != idx_j and r not in dist2_adj[idx_j]:
                            candidate_reps.add(r)
                
                for rep in list(candidate_reps)[:50]:
                    covers = np.count_nonzero(coverage[neighbors[rep]] == 0)
                    if covers > best_covers:
                        best_covers = covers
                        best_rep = rep
                
                if best_rep >= 0 and best_covers >= len(uncovered_i):
                    coverage[neighbors[best_rep]] += 1
                    current_code[i] = idx_to_word(best_rep)
                    current_code_idx[i] = best_rep
                    code_set.discard(idx_i)
                    code_set.add(best_rep)
                    dist2_conflict_removed = True
                else:
                    # Revert
                    coverage[neighbors[idx_i]] += 1
            
            # If still conflicting, try removing j
            if i < len(current_code_idx) and j < len(current_code_idx):
                idx_i = current_code_idx[i]
                idx_j = current_code_idx[j]
                if idx_j in dist2_adj[idx_i]:
                    coverage[neighbors[idx_j]] -= 1
                    uncovered_j = np.where(coverage == 0)[0]
                    
                    if len(uncovered_j) == 0:
                        current_code.pop(j)
                        current_code_idx.pop(j)
                        dist2_conflict_removed = True
                        continue
                    else:
                        best_rep = -1
                        best_covers = 0
                        candidate_reps = set()
                        for uidx in uncovered_j[:20]:
                            for r in reverse_neighbors_arr[uidx]:
                                if r not in code_set and r != idx_i and r not in dist2_adj[idx_i]:
                                    candidate_reps.add(r)
                        
                        for rep in list(candidate_reps)[:50]:
                            covers = np.count_nonzero(coverage[neighbors[rep]] == 0)
                            if covers > best_covers:
                                best_covers = covers
                                best_rep = rep
                        
                        if best_rep >= 0 and best_covers >= len(uncovered_j):
                            coverage[neighbors[best_rep]] += 1
                            current_code[j] = idx_to_word(best_rep)
                            current_code_idx[j] = best_rep
                            code_set.discard(idx_j)
                            code_set.add(best_rep)
                            dist2_conflict_removed = True
                        else:
                            coverage[neighbors[idx_j]] += 1
    
    # Helper: check if all neighbors of a codeword have coverage > 1
    def is_redundant(cw_idx):
        return np.all(coverage[neighbors[cw_idx]] > 1)
    
    # Helper: remove a codeword
    def remove_codeword(pos):
        nonlocal current_code, current_code_idx
        cw_idx = current_code_idx[pos]
        coverage[neighbors[cw_idx]] -= 1
        current_code.pop(pos)
        current_code_idx.pop(pos)
    
    # Helper: add a codeword
    def add_codeword(cw_idx):
        nonlocal current_code, current_code_idx
        coverage[neighbors[cw_idx]] += 1
        current_code.append(idx_to_word(cw_idx))
        current_code_idx.append(cw_idx)
    
    # 4. Redundancy Removal Loop - vectorized check
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        new_code_idx = []
        for i, cw_idx in enumerate(current_code_idx):
            if time.time() > deadline:
                # Keep remaining
                new_code_idx.extend(current_code_idx[i:])
                break
            if np.all(coverage[neighbors[cw_idx]] > 1):
                coverage[neighbors[cw_idx]] -= 1
                improved = True
            else:
                new_code_idx.append(cw_idx)
                
        current_code_idx = new_code_idx
        current_code = [idx_to_word(ci) for ci in current_code_idx]
    
    # 5. Local Search with faster swap moves
    code_set = set(current_code_idx)
    
    # Track distance-2 conflicts for penalty-based local search
    def count_dist2_conflicts(code_idx_list):
        """Count number of pairs at distance 2 in the code."""
        conflicts = 0
        s = set(code_idx_list)
        for idx in code_idx_list:
            for nb in dist2_adj_arr[idx]:
                if nb in s and nb > idx:
                    conflicts += 1
        return conflicts
    
    # Precompute for each word: how many codewords cover it (already in coverage)
    
    # Try multiple rounds of swap-based improvement
    swap_improved = True
    max_swap_rounds = 200
    for round_num in range(max_swap_rounds):
        if time.time() > deadline:
            break
        if not swap_improved:
            break
        swap_improved = False
        
        n_codewords = len(current_code_idx)
        if n_codewords == 0:
            break
        
        # Shuffle candidates
        candidates = list(range(n_codewords))
        random.shuffle(candidates)
        
        best_swap = None  # (remove_pos, add_idx, net_gain)
        best_gain = 0
        
        # Check all candidates (up to ~80, fast enough)
        for cand_pos in candidates:
            if time.time() > deadline:
                break
            
            remove_idx = current_code_idx[cand_pos]
            remove_nbs = neighbors[remove_idx]
            
            # Temporarily remove
            coverage[remove_nbs] -= 1
            uncovered_mask = (coverage == 0)
            n_uncovered = np.count_nonzero(uncovered_mask)
            
            if n_uncovered == 0:
                # Redundant - just remove
                current_code.pop(cand_pos)
                current_code_idx.pop(cand_pos)
                code_set.discard(remove_idx)
                swap_improved = True
                break
            else:
                # Find best replacement using reverse neighbors of uncovered words
                # Prefer replacements that reduce distance-2 conflicts
                
                # Collect candidate replacements from uncovered words' reverse neighbors
                candidate_reps = set()
                uncovered_indices = np.where(uncovered_mask)[0]
                # Sample if too many uncovered
                if len(uncovered_indices) > 30:
                    sampled = random.sample(list(uncovered_indices), 30)
                else:
                    sampled = list(uncovered_indices)
                
                for uidx in sampled:
                    for r in reverse_neighbors_arr[uidx]:
                        if r not in code_set:
                            candidate_reps.add(r)
                
                # Score candidates with distance-2 conflict penalty
                if candidate_reps:
                    rep_list = list(candidate_reps)
                    if len(rep_list) > 150:
                        rep_list = random.sample(rep_list, 150)
                    
                    best_rep_idx = -1
                    best_score = -1000000
                    
                    # Count current distance-2 conflicts for scoring
                    current_conflicts = 0
                    temp_code_set = set(current_code_idx)
                    temp_code_set.discard(remove_idx)
                    for idx in current_code_idx:
                        if idx == remove_idx:
                            continue
                        for nb in dist2_adj_arr[idx]:
                            if nb in temp_code_set and nb > idx:
                                current_conflicts += 1
                    
                    for rep_idx in rep_list:
                        rep_nbs = neighbors[rep_idx]
                        covers = np.count_nonzero(coverage[rep_nbs] == 0)
                        
                        # Only consider if it covers all uncovered
                        if covers < n_uncovered:
                            continue
                        
                        # Count distance-2 conflicts if we add this rep
                        new_conflicts = current_conflicts
                        for nb in dist2_adj_arr[rep_idx]:
                            if nb in temp_code_set:
                                new_conflicts += 1
                        
                        # Score: prefer fewer conflicts, then more coverage
                        score = -new_conflicts * 100 + covers
                        if score > best_score:
                            best_score = score
                            best_rep_idx = rep_idx
                    
                    if best_rep_idx >= 0:
                        gain = 1  # Any valid swap is an improvement
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (cand_pos, best_rep_idx)
                
                # Revert
                coverage[remove_nbs] += 1
        
        # Apply best swap
        if best_swap is not None and best_gain >= 0:
            cand_pos, add_idx = best_swap
            remove_idx = current_code_idx[cand_pos]
            
            # Remove old
            coverage[neighbors[remove_idx]] -= 1
            current_code.pop(cand_pos)
            current_code_idx.pop(cand_pos)
            code_set.discard(remove_idx)
            
            # Add new
            coverage[neighbors[add_idx]] += 1
            current_code.append(idx_to_word(add_idx))
            current_code_idx.append(add_idx)
            code_set.add(add_idx)
            
            swap_improved = True
    
    # 6. Final redundant removal pass
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        new_code_idx = []
        for i, cw_idx in enumerate(current_code_idx):
            if time.time() > deadline:
                new_code_idx.extend(current_code_idx[i:])
                break
            if np.all(coverage[neighbors[cw_idx]] > 1):
                coverage[neighbors[cw_idx]] -= 1
                improved = True
            else:
                new_code_idx.append(cw_idx)
                
        current_code_idx = new_code_idx
        current_code = [idx_to_word(ci) for ci in current_code_idx]
    
    # 7. Additional local search rounds with perturbation
    best_code_found = list(current_code_idx)
    best_code_size = len(current_code_idx)
    
    for pert_round in range(3):
        if time.time() > deadline - 10:
            break
        
        # Perturb: randomly remove 3-5 codewords and re-add to escape local optima
        if len(current_code_idx) > 80:
            pert_size = min(5, len(current_code_idx) - 75)
            pert_positions = random.sample(range(len(current_code_idx)), pert_size)
            pert_positions.sort(reverse=True)
            for pos in pert_positions:
                cw_idx = current_code_idx[pos]
                coverage[neighbors[cw_idx]] -= 1
                current_code.pop(pos)
                current_code_idx.pop(pos)
            
            # Re-add to restore coverage
            while True:
                if time.time() > deadline - 5:
                    break
                uncovered = np.where(coverage == 0)[0]
                if len(uncovered) == 0:
                    break
                
                # Find best word to add
                best_add = -1
                best_covers = 0
                for uidx in uncovered[:15]:
                    for r in reverse_neighbors_arr[uidx]:
                        if r not in set(current_code_idx):
                            covers = np.count_nonzero(coverage[neighbors[r]] == 0)
                            if covers > best_covers:
                                best_covers = covers
                                best_add = r
                
                if best_add < 0:
                    break
                
                coverage[neighbors[best_add]] += 1
                current_code.append(idx_to_word(best_add))
                current_code_idx.append(best_add)
        
        # Run another round of local search
        code_set = set(current_code_idx)
        swap_improved = True
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if not swap_improved:
                break
            swap_improved = False
            
            n_codewords = len(current_code_idx)
            if n_codewords == 0:
                break
            
            candidates = list(range(n_codewords))
            random.shuffle(candidates)
            
            best_swap = None
            best_gain = 0
            
            for cand_pos in candidates:
                if time.time() > deadline - 5:
                    break
                
                remove_idx = current_code_idx[cand_pos]
                remove_nbs = neighbors[remove_idx]
                
                coverage[remove_nbs] -= 1
                uncovered_mask = (coverage == 0)
                n_uncovered = np.count_nonzero(uncovered_mask)
                
                if n_uncovered == 0:
                    current_code.pop(cand_pos)
                    current_code_idx.pop(cand_pos)
                    code_set.discard(remove_idx)
                    swap_improved = True
                    break
                else:
                    candidate_reps = set()
                    uncovered_indices = np.where(uncovered_mask)[0]
                    if len(uncovered_indices) > 25:
                        sampled = random.sample(list(uncovered_indices), 25)
                    else:
                        sampled = list(uncovered_indices)
                    
                    for uidx in sampled:
                        for r in reverse_neighbors_arr[uidx]:
                            if r not in code_set:
                                candidate_reps.add(r)
                    
                    if candidate_reps:
                        rep_list = list(candidate_reps)
                        if len(rep_list) > 80:
                            rep_list = random.sample(rep_list, 80)
                        
                        best_rep_idx = -1
                        best_rep_covers = 0
                        
                        for rep_idx in rep_list:
                            rep_nbs = neighbors[rep_idx]
                            covers = np.count_nonzero(coverage[rep_nbs] == 0)
                            if covers > best_rep_covers:
                                best_rep_covers = covers
                                best_rep_idx = rep_idx
                        
                        if best_rep_idx >= 0 and best_rep_covers >= n_uncovered:
                            gain = best_rep_covers - n_uncovered
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (cand_pos, best_rep_idx)
                    
                    coverage[remove_nbs] += 1
            
            if best_swap is not None:
                cand_pos, add_idx = best_swap
                remove_idx = current_code_idx[cand_pos]
                
                coverage[neighbors[remove_idx]] -= 1
                current_code.pop(cand_pos)
                current_code_idx.pop(cand_pos)
                code_set.discard(remove_idx)
                
                coverage[neighbors[add_idx]] += 1
                current_code.append(idx_to_word(add_idx))
                current_code_idx.append(add_idx)
                code_set.add(add_idx)
                
                swap_improved = True
        
        # Final redundant removal
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            new_code_idx = []
            for i, cw_idx in enumerate(current_code_idx):
                if np.all(coverage[neighbors[cw_idx]] > 1):
                    coverage[neighbors[cw_idx]] -= 1
                    improved = True
                else:
                    new_code_idx.append(cw_idx)
            current_code_idx = new_code_idx
            current_code = [idx_to_word(ci) for ci in current_code_idx]
        
        # Keep best
        if len(current_code_idx) < best_code_size:
            best_code_size = len(current_code_idx)
            best_code_found = list(current_code_idx)
    
    # 8. Verify coverage (safety check)
    if time.time() < deadline - 3.0:
        final_coverage = np.zeros(total_words, dtype=np.int32)
        for cw_idx in current_code_idx:
            final_coverage[neighbors[cw_idx]] += 1
        if np.any(final_coverage == 0):
            # Coverage broken! Use best valid code found
            if best_code_found:
                current_code_idx = list(best_code_found)
                current_code = [idx_to_word(ci) for ci in current_code_idx]
            else:
                current_code = list(seed_codewords)
                current_code_idx = [word_to_idx(cw) for cw in current_code]
        
        # Also verify best_code_found
        if best_code_found and len(best_code_found) < len(current_code_idx):
            bc_coverage = np.zeros(total_words, dtype=np.int32)
            for cw_idx in best_code_found:
                bc_coverage[neighbors[cw_idx]] += 1
            if not np.any(bc_coverage == 0):
                current_code_idx = list(best_code_found)
                current_code = [idx_to_word(ci) for ci in current_code_idx]
    
    # Convert back to list of lists
    return [list(cw) for cw in current_code]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback implementation for other parameters
    # Basic greedy set cover
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors (words within Hamming distance radius)
    # For small n and q, enumerate all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_id = {w: i for i, w in enumerate(all_words)}
    
    # Build neighbor lists
    neighbors = []
    for w in all_words:
        nb_set = set()
        # All words within Hamming distance radius
        for u in all_words:
            dist = sum(1 for a, b in zip(w, u) if a != b)
            if dist <= radius:
                nb_set.add(word_to_id[u])
        neighbors.append(sorted(nb_set))
    
    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline:
        # Find word covering most uncovered
        best_word = -1
        best_count = 0
        for w_idx in range(total_words):
            if len(uncovered) == 0:
                break
            covered = len(uncovered & set(neighbors[w_idx]))
            if covered > best_count:
                best_count = covered
                best_word = w_idx
        
        if best_word < 0:
            break
        
        code.append(all_words[best_word])
        uncovered -= set(neighbors[best_word])
    
    return [list(w) for w in code]

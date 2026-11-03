import time
import random
import sys

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute neighbors and bitsets
    # neighbors[i] is a list of indices covered by codeword i
    # cover_bitsets[i] is an integer where bit j is 1 if word j is covered by codeword i
    # reverse_cover[j] is a list of codewords that cover word j
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    neighbors = []
    cover_bitsets = []
    reverse_cover = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        
        neighbors.append(idxs)
        
        bs = 0
        for j in idxs:
            bs |= (1 << j)
            reverse_cover[j].append(i)
            
        cover_bitsets.append(bs)
    
    # Precompute a "full" mask for all words
    full_mask = (1 << total_words) - 1

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            # Optimization: Only check candidates that might have high gain?
            # For 2187 words, checking all is fast enough with bitsets.
            for i in range(total_words):
                # Gain is number of currently uncovered words covered by i
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code

    def reduce_code(code):
        # Counts based reduction
        # counts[w] is number of codewords in 'code' that cover word w
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Tentatively remove c
                for w in neighbors[c]:
                    counts[w] -= 1
                
                # Check if any word becomes uncovered
                can_remove = True
                for w in neighbors[c]:
                    if counts[w] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    changed = True
                else:
                    # Restore counts
                    for w in neighbors[c]:
                        counts[w] += 1
                    new_code.append(c)
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        # Maintain counts for efficiency
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        improved = True
        # Track time
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # 1. Try Removal
                # Subtract counts
                for w in neighbors[c_remove]:
                    counts[w] -= 1
                
                # Check for holes
                holes = []
                for w in neighbors[c_remove]:
                    if counts[w] == 0:
                        holes.append(w)
                
                if not holes:
                    # Successful removal
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore counts
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # 2. Try Swap
                    # Find intersection of reverse_cover for all holes
                    # Start with the hole with smallest reverse_cover to minimize work
                    holes.sort(key=lambda x: len(reverse_cover[x]))
                    candidates = set(reverse_cover[holes[0]])
                    
                    for h in holes[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if candidates:
                        # Score candidates
                        # Score = sum of counts of words covered by candidate
                        # Higher score means candidate covers more already-covered words (redundant)
                        # Wait, we want to minimize redundancy?
                        # Actually, if we swap in a candidate, we want it to cover the holes.
                        # Any candidate in the intersection covers all holes.
                        # We should prefer candidates that cover *more* words overall?
                        # Or candidates that cover words with high counts?
                        # Let's look at the "excess" coverage.
                        # If we swap c_remove -> c_new:
                        # Words covered by c_remove but not c_new: counts decrease.
                        # Words covered by c_new but not c_remove: counts increase.
                        # We must ensure no count drops to 0.
                        
                        # Let's use a simpler heuristic: 
                        # Pick a candidate that maximizes the minimum count of words it covers?
                        # Or just random choice from candidates is often good enough.
                        # Let's try scoring by "potential to enable future removals".
                        # A good swap is one where the new codeword covers words that are "critical" (count 1) 
                        # or words that are covered by many others (high count) - actually, 
                        # covering high count words is "wasteful" if we can avoid it, 
                        # but covering count 1 words is necessary.
                        
                        # Let's just pick a random candidate for now, but maybe biased towards 
                        # candidates that are "less similar" to existing code?
                        # For simplicity and speed, random choice from valid swaps is a strong baseline.
                        # Let's add a slight bias: prefer candidates that cover more words in total (higher degree)
                        # No, all codewords have the same degree (1 + n*(q-1)).
                        
                        # Let's try to minimize the "waste".
                        # Waste = sum of (counts[w] - 1) for w in neighbors[c_new] \ neighbors[c_remove]
                        # We want to minimize waste? Or maximize it?
                        # If we add a codeword that covers words with count 1, we increase redundancy for those words.
                        # If we add a codeword that covers words with count > 1, we increase redundancy further.
                        # We want to keep counts as low as possible (ideally 1).
                        # So we prefer c_new that covers words with LOW counts?
                        # Actually, we just need to cover the holes.
                        # The holes had count 0. Now they will have count 1 (from c_new).
                        # The other words covered by c_new will have their count increased.
                        # We want to minimize the increase in total sum of counts?
                        # Sum of counts is fixed by number of codewords * degree.
                        # So we want to distribute counts evenly?
                        
                        # Let's just pick randomly.
                        best_cand = random.choice(list(candidates))
                        
                        # Perform Swap
                        for w in neighbors[c_remove]:
                            counts[w] -= 1
                        for w in neighbors[best_cand]:
                            counts[w] += 1
                        code[i] = best_cand
                        improved = True
                        break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        # Calculate scores to determine which to remove
        # Score = number of words covered by this codeword that are covered by NO OTHER codeword?
        # i.e., Critical coverage.
        # We want to remove codewords with LOW critical coverage.
        
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        critical_scores = []
        for i, c in enumerate(code):
            crit = 0
            for w in neighbors[c]:
                if counts[w] == 1:
                    crit += 1
            critical_scores.append(crit)
        
        # Remove k codewords with lowest critical scores
        # Sort indices by score
        indices = list(range(len(code)))
        indices.sort(key=lambda x: critical_scores[x])
        
        remove_indices = indices[:k]
        
        # Remove them
        # To avoid index shifting issues, remove in reverse order
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
        uncovered_mask = full_mask
        # Rebuild uncovered mask from current code
        covered_mask = 0
        for c in code:
            covered_mask |= cover_bitsets[c]
        uncovered_mask = full_mask & ~covered_mask
        
        while uncovered_mask != 0 and time.time() < deadline:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
            
        return code

    start_time = time.time()
    # Leave 5 seconds margin for safety
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        # Local search
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # Perturbation
        if time.time() < deadline - 15 and best_code is not None:
            # Adapt k based on remaining time or code size
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 10)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 5:
                perturbed = local_search(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
        
    # Fallback if no code found (should not happen)
    if best_code is None:
        best_code = greedy_construct(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result

import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 120s limit, leave 5s for overhead/output
    deadline = start_time + 115.0 

    # Handle general case fallback if not the specific task
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy fallback for robustness
        N = q ** n
        if N > 100000:
            # Too large, return all words (trivial upper bound)
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific implementation for q=3, n=8, radius=1
    N = q ** n  # 6561
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors efficiently
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                new_w = w + diff * powers[i]
                neighbors[w, idx] = new_w
                idx += 1

    # Helper to compute coverage from a boolean mask
    def compute_coverage(in_code):
        cov = np.zeros(N, dtype=np.int32)
        for w in np.where(in_code)[0]:
            cov[neighbors[w, :]] += 1
        return cov

    def greedy_init(rng):
        """
        Greedy set cover with bucket-based max gain selection.
        Returns a valid code (in_code boolean array).
        """
        in_code = np.zeros(N, dtype=np.bool_)
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_count = N
        gains = np.full(N, neigh_size, dtype=np.int32)
        
        # Buckets for fast max-gain retrieval
        buckets = [[] for _ in range(neigh_size + 1)]
        for w in range(N):
            buckets[gains[w]].append(w)
        max_gain = neigh_size
        
        # Deadline for greedy phase: use first 5 seconds or until covered
        greedy_deadline = start_time + 5.0
        
        while uncovered_count > 0 and time.time() < greedy_deadline:
            # Find current max gain
            while max_gain > 0 and len(buckets[max_gain]) == 0:
                max_gain -= 1
            if max_gain == 0:
                break
            
            candidates = buckets[max_gain]
            if not candidates:
                continue
                
            # Pick random candidate from max gain bucket
            w = int(rng.choice(candidates))
            
            if in_code[w]:
                buckets[max_gain].remove(w)
                continue
                
            in_code[w] = True
            buckets[max_gain].remove(w)
            nbs = neighbors[w, :]
            
            for u in nbs:
                old_cov = coverage[u]
                coverage[u] = old_cov + 1
                
                if old_cov == 0:
                    # u just became covered.
                    # Decrement gains for all v that cover u
                    for v in neighbors[u, :]:
                        old_g = gains[v]
                        new_g = old_g - 1
                        if new_g < 0:
                            new_g = 0
                        gains[v] = new_g
                        if in_code[v]:
                            continue
                        if new_g != old_g:
                            buckets[old_g].remove(v)
                            buckets[new_g].append(v)
                    uncovered_count -= 1
            
            if uncovered_count == 0:
                break
        
        # If greedy didn't finish in time, force finish to ensure validity
        if uncovered_count > 0:
            # Add all remaining uncovered words to ensure valid code
            for w in range(N):
                if not in_code[w]:
                    # Check if it covers any uncovered word
                    if np.any(coverage[neighbors[w, :]] == 0):
                         in_code[w] = True
                         # Update coverage for final check
                         coverage[neighbors[w, :]] += 1
            # Recalculate coverage to be safe
            coverage = compute_coverage(in_code)
            
        return in_code

    def local_search(in_code, rng, deadline):
        """
        Simulated Annealing to minimize code size.
        Moves: 
        1. Deletion (if valid)
        2. Swap (remove 1, add 1)
        3. Double-swap (remove 2, add 1)
        Optimized with explicit lists for code words.
        """
        current_in_code = in_code.copy()
        current_coverage = compute_coverage(current_in_code)
        current_size = np.sum(current_in_code)
        
        best_in_code = current_in_code.copy()
        best_size = current_size
        
        # SA Parameters
        T = 10.0
        alpha = 0.9995
        min_T = 0.01
        M = 10000  # Penalty for uncovered words
        
        sa_deadline = deadline - 5.0 # Leave 5s for final polish
        
        # Maintain list of code words for O(1) selection
        code_words = np.where(current_in_code)[0].tolist()
        
        while time.time() < sa_deadline:
            if T < min_T:
                if rng.random() < 0.05:
                    T = 1.0
                else:
                    break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # 1. Deletion
                if len(code_words) == 0:
                    break
                idx = int(rng.integers(len(code_words)))
                w = code_words[idx]
                
                nbs = neighbors[w, :]
                prev_cov = current_coverage[nbs].copy()
                current_coverage[nbs] -= 1
                newly_uncovered = np.sum((prev_cov == 1) & (current_coverage[nbs] == 0))
                
                delta_uncovered = newly_uncovered
                delta_size = -1
                delta_cost = M * delta_uncovered + delta_size
                
                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_in_code[w] = False
                    current_size -= 1
                    code_words.pop(idx)
                    
                    if delta_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    # Revert
                    current_coverage[nbs] = prev_cov
                    
            elif move_type < 0.8:
                # 2. Swap: Remove w1, Add w2
                if len(code_words) == 0:
                    break
                idx1 = int(rng.integers(len(code_words)))
                w1 = code_words[idx1]
                nbs1 = neighbors[w1, :]
                
                # Identify essential neighbors of w1 (covered only by w1)
                prev_cov1 = current_coverage[nbs1].copy()
                current_coverage[nbs1] -= 1
                essential_mask = (prev_cov1 == 1) & (current_coverage[nbs1] == 0)
                essential_words = nbs1[essential_mask]
                
                if len(essential_words) == 0:
                    # No essential words. Try to swap with a random non-code word to explore.
                    w2 = int(rng.integers(N))
                    if not current_in_code[w2]:
                        nbs2 = neighbors[w2, :]
                        prev_cov2 = current_coverage[nbs2].copy()
                        current_coverage[nbs2] += 1
                        
                        if np.all(current_coverage > 0):
                            # Valid swap
                            current_in_code[w1] = False
                            current_in_code[w2] = True
                            code_words[idx1] = w2
                            # Size unchanged
                        else:
                            # Revert
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        current_coverage[nbs1] = prev_cov1
                else:
                    # Need w2 to cover ALL essential_words
                    # Intersection of neighbors of essential_words
                    cand_set = set(neighbors[essential_words[0], :].tolist())
                    for ew in essential_words[1:]:
                        cand_set.intersection_update(neighbors[ew, :].tolist())
                    
                    # Filter candidates: must not be in code
                    candidates = [c for c in cand_set if not current_in_code[c]]
                    
                    if candidates:
                        w2 = candidates[int(rng.integers(len(candidates)))]
                        nbs2 = neighbors[w2, :]
                        prev_cov2 = current_coverage[nbs2].copy()
                        current_coverage[nbs2] += 1
                        
                        # Check validity
                        if np.all(current_coverage > 0):
                            current_in_code[w1] = False
                            current_in_code[w2] = True
                            code_words[idx1] = w2
                        else:
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        # No valid swap found, revert
                        current_coverage[nbs1] = prev_cov1
                        
            else:
                # 3. Double-swap: Remove w1, w2, Add w3
                if len(code_words) < 2:
                    T *= alpha
                    continue
                
                idx1 = int(rng.integers(len(code_words)))
                idx2 = int(rng.integers(len(code_words) - 1))
                if idx2 >= idx1:
                    idx2 += 1
                
                w1 = code_words[idx1]
                w2 = code_words[idx2]
                
                nbs1 = neighbors[w1, :]
                nbs2 = neighbors[w2, :]
                
                prev_cov1 = current_coverage[nbs1].copy()
                prev_cov2 = current_coverage[nbs2].copy()
                
                # Apply removals
                current_coverage[nbs1] -= 1
                current_coverage[nbs2] -= 1
                
                newly_uncovered_mask = (current_coverage == 0)
                newly_uncovered_words = np.where(newly_uncovered_mask)[0]
                
                if len(newly_uncovered_words) == 0:
                    # Both removals safe (size -2)
                    current_in_code[w1] = False
                    current_in_code[w2] = False
                    current_size -= 2
                    # Remove from code_words list
                    if idx1 > idx2:
                        code_words.pop(idx1)
                        code_words.pop(idx2)
                    else:
                        code_words.pop(idx2)
                        code_words.pop(idx1)
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    # Need to add a word that covers all newly_uncovered_words
                    if len(newly_uncovered_words) <= 5:
                        cand_set = set(neighbors[newly_uncovered_words[0], :].tolist())
                        for uw in newly_uncovered_words[1:]:
                            cand_set.intersection_update(neighbors[uw, :].tolist())
                        candidates = [c for c in cand_set if not current_in_code[c]]
                        
                        if candidates:
                            w3 = candidates[int(rng.integers(len(candidates)))]
                            nbs3 = neighbors[w3, :]
                            prev_cov3 = current_coverage[nbs3].copy()
                            current_coverage[nbs3] += 1
                            
                            if np.all(current_coverage > 0):
                                current_in_code[w1] = False
                                current_in_code[w2] = False
                                current_in_code[w3] = True
                                current_size -= 1
                                # Update code_words
                                if idx1 > idx2:
                                    code_words.pop(idx1)
                                    code_words.pop(idx2)
                                else:
                                    code_words.pop(idx2)
                                    code_words.pop(idx1)
                                code_words.append(w3)
                                
                                if current_size < best_size:
                                    best_size = current_size
                                    best_in_code = current_in_code.copy()
                            else:
                                # Revert
                                current_coverage[nbs1] = prev_cov1
                                current_coverage[nbs2] = prev_cov2
                                current_coverage[nbs3] = prev_cov3
                        else:
                            # Revert
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        # Too many uncovered, revert
                        current_coverage[nbs1] = prev_cov1
                        current_coverage[nbs2] = prev_cov2
                        
            T *= alpha

        return best_in_code, best_size

    def final_polish(in_code):
        """
        Greedy deletion pass to remove any redundant codewords.
        """
        current_in_code = in_code.copy()
        current_coverage = compute_coverage(current_in_code)
        current_size = np.sum(current_in_code)
        
        # Iterate until no more changes
        changed = True
        while changed:
            changed = False
            code_words = np.where(current_in_code)[0]
            for w in code_words:
                nbs = neighbors[w, :]
                # If all neighbors have coverage > 1, w is redundant
                if np.min(current_coverage[nbs]) > 1:
                    current_in_code[w] = False
                    current_coverage[nbs] -= 1
                    current_size -= 1
                    changed = True
                    
        return current_in_code, current_size

    # Main Logic
    rng = np.random.default_rng(42)
    
    # 1. Greedy Initialization
    initial_code = greedy_init(rng)
    
    # 2. Local Search (SA)
    best_code, best_size = local_search(initial_code, rng, deadline)
    
    # 3. Final Polish
    polished_code, polished_size = final_polish(best_code)
    
    if polished_size < best_size:
        best_code = polished_code
        best_size = polished_size
        
    # If we have time, run a second restart with different seed?
    # Given the tight time, one good restart with strong SA is usually sufficient.
    # Let's check if we have significant time left.
    if time.time() < deadline - 10:
        rng2 = np.random.default_rng(123)
        initial_code2 = greedy_init(rng2)
        code2, size2 = local_search(initial_code2, rng2, deadline)
        polished_code2, polished_size2 = final_polish(code2)
        
        if polished_size2 < best_size:
            best_code = polished_code2
            best_size = polished_size2

    # Construct output
    final_code = []
    for w in range(N):
        if best_code[w]:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
            
    return final_code

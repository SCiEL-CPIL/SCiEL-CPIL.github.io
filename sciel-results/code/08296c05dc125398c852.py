import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate all words and map to indices
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)
    
    # Convert neighbors to numpy arrays for faster membership checks
    neighbors_set = [set(neigh) for neigh in neighbors]
    
    # Greedy Set Cover with vectorized counting
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            count = np.sum(uncovered[neighbors[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() > deadline - 30.0:
            break
    
    # Maintain coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        coverage_count[neighbors[j]] += 1
    
    def is_redundant(j, cov):
        return np.all(cov[neighbors[j]] > 1)

    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set) - 1, -1, -1):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    cov[neighbors[j]] -= 1
                    selected_set.pop(idx)
                    changed = True
        return selected_set

    best_selected = list(selected)
    best_cov = coverage_count.copy()
    
    # First, reduce the initial greedy solution
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    # Local Search: Swap Moves with proper accept/reject
    random.seed(42)
    
    max_iters = 5000
    iter_count = 0
    no_improve_count = 0
    current_size = len(best_selected)
    
    def compute_swap_score(c_out, c_in, cov, selected_set):
        """Compute a score for a swap. Higher is better."""
        # Words that become uncovered if c_out removed
        U = [nb for nb in neighbors[c_out] if cov[nb] == 1]
        if not U:
            return 1000  # Pure removal, very good
        
        # Check c_in covers all of U
        for w in U:
            if w not in neighbors_set[c_in]:
                return -1000  # Invalid swap
        
        # Score components:
        # 1. Extra critical words covered by c_in (makes other codewords redundant)
        extra_critical = 0
        for nb in neighbors[c_in]:
            if nb not in neighbors_set[c_out] and cov[nb] == 1:
                extra_critical += 1
        
        # 2. Reduce over-coverage: count how many words go from cov>2 to cov-1
        #    (c_in adds coverage, c_out removes it; net change per word)
        #    For words in both neighbors: no net change
        #    For words only in c_in: +1 coverage
        #    For words only in c_out: -1 coverage
        #    We want to reduce words with high coverage
        
        # 3. Diversity: prefer c_in that is far from existing codewords
        #    (hard to compute quickly, skip for now)
        
        score = extra_critical * 10 - len(U)
        return score

    def attempt_swap(best_selected, best_cov, deadline):
        """Try to find and apply a beneficial swap. Returns (success, new_size)."""
        candidates_out = list(best_selected)
        random.shuffle(candidates_out)
        
        best_swap = None
        best_swap_score = -1000
        
        for c_out in candidates_out:
            if time.time() > deadline - 5.0:
                break
                
            U = [nb for nb in neighbors[c_out] if best_cov[nb] == 1]
            
            if not U:
                # Redundant codeword, just remove it
                return True, len(best_selected) - 1
            
            # Find best c_in
            w0 = U[0]
            for c_in in neighbors[w0]:
                if c_in in best_selected:
                    continue
                
                # Quick check: does c_in cover all of U?
                covers_all = all(w in neighbors_set[c_in] for w in U)
                if not covers_all:
                    continue
                
                score = compute_swap_score(c_out, c_in, best_cov, best_selected)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (c_out, c_in)
        
        if best_swap is None:
            return False, len(best_selected)
        
        c_out, c_in = best_swap
        
        # Apply the swap
        best_cov[neighbors[c_out]] -= 1
        best_selected.remove(c_out)
        best_cov[neighbors[c_in]] += 1
        best_selected.append(c_in)
        
        # Try to reduce
        best_selected = try_reduce(best_selected, best_cov, deadline)
        
        return True, len(best_selected)

    while time.time() < deadline - 10.0 and iter_count < max_iters:
        iter_count += 1
        
        success, new_size = attempt_swap(best_selected, best_cov, deadline)
        
        if success:
            if new_size < current_size:
                current_size = new_size
                no_improve_count = 0
            elif new_size == current_size:
                # Accept with probability to escape local minima
                if random.random() < 0.3:
                    no_improve_count = 0
                else:
                    # Revert: this is complex, so we just accept
                    # The try_reduce should have cleaned up
                    pass
            else:
                # Size increased, should not happen with our scoring
                # But try_reduce might have failed to clean up
                pass
        else:
            no_improve_count += 1
        
        # Periodic perturbation if stuck
        if no_improve_count > 200 and random.random() < 0.1:
            # Remove a random codeword and re-greedy fill
            if len(best_selected) > 50:
                idx_to_remove = random.randint(0, len(best_selected) - 1)
                c_removed = best_selected[idx_to_remove]
                best_cov[neighbors[c_removed]] -= 1
                best_selected.pop(idx_to_remove)
                
                # Re-greedy fill
                uncovered = np.zeros(N, dtype=bool)
                uncovered[best_cov == 0] = True
                fill_count = 0
                while uncovered.any() and fill_count < 10:
                    best_idx = -1
                    best_count = 0
                    for j in range(N):
                        count = np.sum(uncovered[neighbors[j]])
                        if count > best_count:
                            best_count = count
                            best_idx = j
                    if best_idx == -1:
                        break
                    best_selected.append(best_idx)
                    best_cov[neighbors[best_idx]] += 1
                    uncovered[neighbors[best_idx]] = False
                    fill_count += 1
                
                best_selected = try_reduce(best_selected, best_cov, deadline)
                no_improve_count = 0
    
    # Final high-precision polish pass
    # Do multiple rounds of reduction with tighter criteria
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        best_selected = try_reduce(best_selected, best_cov, deadline)
        
        # Try single swaps with stricter acceptance
        candidates_out = list(best_selected)
        random.shuffle(candidates_out)
        for c_out in candidates_out:
            if time.time() > deadline - 2.0:
                break
            U = [nb for nb in neighbors[c_out] if best_cov[nb] == 1]
            if not U:
                best_cov[neighbors[c_out]] -= 1
                best_selected.remove(c_out)
                continue
            
            w0 = U[0]
            best_c_in = -1
            best_extra = -1
            for c_in in neighbors[w0]:
                if c_in in best_selected:
                    continue
                if not all(w in neighbors_set[c_in] for w in U):
                    continue
                extra = sum(1 for nb in neighbors[c_in] 
                           if nb not in neighbors_set[c_out] and best_cov[nb] == 1)
                if extra > best_extra:
                    best_extra = extra
                    best_c_in = c_in
            
            if best_c_in != -1 and best_extra > 0:
                best_cov[neighbors[c_out]] -= 1
                best_selected.remove(c_out)
                best_cov[neighbors[best_c_in]] += 1
                best_selected.append(best_c_in)
                best_selected = try_reduce(best_selected, best_cov, deadline)
    
    # Convert indices back to words
    codewords = [words[i] for i in best_selected]
    
    return codewords

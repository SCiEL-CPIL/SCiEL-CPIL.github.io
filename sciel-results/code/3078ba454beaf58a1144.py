"""Vectorized greedy set cover + local search for ternary covering code K_3(8,1)."""
import time
import numpy as np

DEADLINE_SECONDS = 140.0


def construct(q, n, radius):
    start_time = time.time()
    
    # Precompute powers of q for index conversion
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i - 1] * q
    
    total_words = q ** n
    
    def idx_to_digits(idx):
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * powers[n - 1 - i]
        return idx
    
    # Build neighbor lists: for each word, list of words within Hamming distance radius
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbor_list = [[] for _ in range(total_words)]
    for i in range(total_words):
        digits = idx_to_digits(i)
        neighbor_list[i].append(i)
        for pos in range(n):
            for new_val in range(q):
                if new_val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                j = digits_to_idx(new_digits)
                neighbor_list[i].append(j)
    
    # Greedy set cover with random tie-breaking
    def greedy_set_cover(neighbor_list, total_words, rng):
        uncovered = np.ones(total_words, dtype=bool)
        uncovered_count = total_words
        chosen = []
        
        while uncovered_count > 0:
            # Find all words that are still uncovered
            uncovered_indices = np.where(uncovered)[0]
            
            # For each candidate codeword, count how many uncovered words it covers
            best_cw = -1
            best_count = 0
            
            # Use a random subset of candidates for speed if needed, but try all first
            for cw in range(total_words):
                if uncovered_count <= 0:
                    break
                # Count uncovered neighbors
                count = 0
                for nb in neighbor_list[cw]:
                    if uncovered[nb]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_cw = cw
                    if best_count == uncovered_count:
                        break
            
            if best_cw < 0:
                break
            
            # Add best_cw to chosen
            chosen.append(best_cw)
            for nb in neighbor_list[best_cw]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    uncovered_count -= 1
        
        return chosen
    
    # Local search: remove redundant codewords
    def local_search_remove(chosen, neighbor_list, total_words):
        if not chosen:
            return chosen
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in chosen:
            for nb in neighbor_list[cw]:
                coverage_count[nb] += 1
        
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for cw in list(chosen_set):
                removable = True
                for nb in neighbor_list[cw]:
                    if coverage_count[nb] == 1:
                        removable = False
                        break
                if removable:
                    for nb in neighbor_list[cw]:
                        coverage_count[nb] -= 1
                    chosen_set.remove(cw)
                    changed = True
        
        return list(chosen_set)
    
    # Local search: try to replace a codeword with another to enable more removals
    def local_search_swap(chosen, neighbor_list, total_words, rng):
        if len(chosen) < 2:
            return chosen
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in chosen:
            for nb in neighbor_list[cw]:
                coverage_count[nb] += 1
        
        chosen_set = set(chosen)
        improved = True
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removing a random codeword and see if we can cover the gap
            for cw in list(chosen_set):
                # Check if removing cw leaves any word uncovered
                removable = True
                for nb in neighbor_list[cw]:
                    if coverage_count[nb] == 1:
                        removable = False
                        break
                
                if removable:
                    for nb in neighbor_list[cw]:
                        coverage_count[nb] -= 1
                    chosen_set.remove(cw)
                    improved = True
                    break
        
        return list(chosen_set)
    
    # Run multiple restarts of greedy set cover
    best_result = None
    best_size = float('inf')
    
    # First, try the direct sum construction as a baseline
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    num_tails = q ** (n - 4)
    baseline_codewords = set()
    for h in tetracode:
        for tail_idx in range(num_tails):
            tail_digits = [0] * (n - 4)
            temp = tail_idx
            for i in range(n - 5, -1, -1):
                tail_digits[i] = temp % q
                temp //= q
            full_word = h + tail_digits
            idx = digits_to_idx(full_word)
            baseline_codewords.add(idx)
    
    baseline_list = list(baseline_codewords)
    baseline_reduced = local_search_remove(baseline_list, neighbor_list, total_words)
    if len(baseline_reduced) < best_size:
        best_size = len(baseline_reduced)
        best_result = baseline_reduced[:]
    
    # Now try greedy set cover from scratch with different random seeds
    num_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS - 20:
        num_restarts += 1
        rng = np.random.default_rng(seed=num_restarts * 12345 + 678)
        
        # Greedy set cover
        chosen = greedy_set_cover(neighbor_list, total_words, rng)
        
        if len(chosen) < best_size:
            # Local search to improve
            chosen = local_search_remove(chosen, neighbor_list, total_words)
            chosen = local_search_swap(chosen, neighbor_list, total_words, rng)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = chosen[:]
        
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
    
    if best_result is None:
        best_result = baseline_list
    
    # Convert to list of digit lists
    result = [idx_to_digits(idx) for idx in best_result]
    return result


# Quick self-test (optional, disabled for speed in evaluation)
if __name__ == "__main__":
    t0 = time.time()
    res = construct(3, 8, 1)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Codewords: {len(res)}")

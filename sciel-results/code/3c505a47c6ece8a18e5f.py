import time
import random
import numpy as np
import sys

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1) using Greedy Initialization 
    followed by Simulated Annealing.
    
    Returns a list of codewords (lists of integers) such that every word in 
    {0..q-1}^n is within Hamming distance 'radius' of some codeword.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Handle non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    # For q=3, n=6, radius=1, each word has 1 + 6*2 = 13 neighbors.
    neighbors = np.zeros((total_words, 13), dtype=np.int32)
    powers = np.array([3**i for i in range(n)], dtype=np.int32)
    
    for i in range(total_words):
        w = [(i // p) % 3 for p in powers]
        neigh_list = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                delta = (new_val - orig) * powers[pos]
                neighbor_idx = (i + delta) % total_words
                neigh_list.append(neighbor_idx)
        neighbors[i] = neigh_list

    # Convert indices to words for output
    def idx_to_word(idx):
        return [(idx // p) % 3 for p in powers]

    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * powers[i]
        return idx

    # --- Phase 1: Greedy Initialization ---
    # Start with empty code.
    # Coverage array: count of how many codewords cover each word.
    # We want to add codewords that cover the most words with coverage 0.
    
    code = []
    coverage = np.zeros(total_words, dtype=np.int32)
    uncovered_mask = np.ones(total_words, dtype=bool) # True if coverage == 0
    
    # Precompute list of all indices for random sampling
    all_indices = np.arange(total_words)
    
    # Greedy steps
    # To speed up, we don't pick the absolute best every time (O(N^2)), 
    # but a random sample of candidates is often sufficient for a good start.
    # However, for N=729, a full greedy pass is feasible if optimized.
    # Let's do a few rounds of greedy.
    
    # Optimization: Instead of checking all 729 candidates for the best add,
    # we can just pick random ones until coverage is full, then prune?
    # No, greedy add is better for size.
    
    # Simple Greedy:
    # While there are uncovered words:
    #   Pick the codeword that covers the most uncovered words.
    #   Add it to code.
    #   Update coverage.
    
    # This is O(K * 729 * 13) roughly. K is ~75. 75*729*13 ~ 700k ops. Very fast.
    
    while np.any(uncovered_mask):
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = -1
        
        # Sample candidates to speed up? No, 729 is small. Check all.
        # We can vectorize the counting if we had a matrix, but we don't.
        # Looping 729 times is fine.
        
        # Optimization: If a word is uncovered, any codeword in its neighborhood covers it.
        # We want to maximize sum(coverage[c] == 0 for c in neighbors[x]).
        
        # Let's just iterate through all possible codewords
        for x in all_indices:
            # Count how many neighbors of x are currently uncovered
            # neighbors[x] is an array of 13 indices
            # We need sum(uncovered_mask[neighbors[x]])
            count = np.sum(uncovered_mask[neighbors[x]])
            if count > best_count:
                best_count = count
                best_idx = x
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        
        # Update coverage
        for nb in neighbors[best_idx]:
            coverage[nb] += 1
            if coverage[nb] == 1:
                uncovered_mask[nb] = False
        
        if time.time() > deadline:
            break

    # --- Phase 2: Local Search / Simulated Annealing ---
    # Current state: code (list of indices), coverage (array)
    # Objective: Minimize len(code) subject to min(coverage) >= 1.
    # Energy E = len(code) + L * sum(coverage == 0)
    # We want to minimize this.
    
    L = 10000  # Large penalty for uncovered words
    
    def get_energy(code_list, cov_arr):
        uncovered_count = np.sum(cov_arr == 0)
        return len(code_list) + L * uncovered_count

    current_code = code[:]
    current_coverage = coverage.copy()
    current_energy = get_energy(current_code, current_coverage)
    
    best_code = current_code[:]
    best_energy = current_energy
    best_coverage = current_coverage.copy()
    
    # Simulated Annealing
    # T starts high, cools down.
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    rng = np.random.default_rng(42)
    
    while T > T_min and time.time() < deadline:
        
        # Generate a move
        # 1. Add a random codeword
        # 2. Remove a random codeword
        # 3. Swap a random codeword with another random codeword
        
        move_type = rng.choice(3)
        
        new_code = current_code[:]
        new_coverage = current_coverage.copy()
        
        if move_type == 0: # Add
            # Pick a random index not in code? Or any index?
            # Adding a duplicate doesn't change coverage or size effectively for energy?
            # Actually, duplicates increase len(code) but not coverage. Bad move.
            # So pick an index NOT in current_code.
            # For simplicity, just pick a random index. If it's in code, it's a waste of time.
            # Better: pick random index. If in code, skip.
            idx = rng.integers(0, total_words)
            if idx in new_code:
                continue
            
            new_code.append(idx)
            for nb in neighbors[idx]:
                new_coverage[nb] += 1
                
        elif move_type == 1: # Remove
            if not new_code:
                continue
            idx = new_code[rng.integers(0, len(new_code))]
            new_code.remove(idx)
            for nb in neighbors[idx]:
                new_coverage[nb] -= 1
                
        else: # Swap
            if not new_code:
                continue
            # Remove one
            idx_out = new_code[rng.integers(0, len(new_code))]
            new_code.remove(idx_out)
            for nb in neighbors[idx_out]:
                new_coverage[nb] -= 1
            
            # Add another
            idx_in = rng.integers(0, total_words)
            # Ensure idx_in is not idx_out (obviously) and not already in new_code
            if idx_in in new_code:
                # Revert or skip? Let's just skip this move.
                continue
            
            new_code.append(idx_in)
            for nb in neighbors[idx_in]:
                new_coverage[nb] += 1
        
        new_energy = get_energy(new_code, new_coverage)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            current_code = new_code
            current_coverage = new_coverage
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = current_code[:]
                best_coverage = current_coverage.copy()
        
        T *= cooling_rate
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break

    # Final verification and cleanup
    # Ensure best_code is valid
    final_coverage = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        for nb in neighbors[idx]:
            final_coverage[nb] += 1
    
    # If any uncovered, we have a problem.
    # This shouldn't happen if we accepted only valid states or started valid.
    # Greedy start is valid.
    # SA moves might create invalid states (uncovered words).
    # But we track best_energy which includes penalty.
    # If best_energy has penalty, it's invalid.
    # We must return a VALID code.
    
    # If the best found was invalid (had uncovered words), we need to fix it.
    # But usually, the greedy start is valid and we only accept moves that don't increase energy too much.
    # However, if we accepted a move that created uncovered words, energy went up by L.
    # We might have accepted it if T was high.
    # So best_code might be invalid.
    
    # Strategy: If best_code is invalid, fall back to the greedy start (which is valid).
    # Or, repair best_code by adding back missing words.
    
    if np.any(final_coverage == 0):
        # Repair: Add words from the greedy solution that cover the gaps
        # Or just return the greedy solution which is definitely valid.
        # Let's check if the greedy solution is still valid (it was).
        # The greedy solution is `code` from Phase 1.
        # Let's verify `code` is valid.
        g_coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code:
            for nb in neighbors[idx]:
                g_coverage[nb] += 1
        if not np.any(g_coverage == 0):
            # Greedy is valid. If best_code is invalid, use greedy.
            # But maybe best_code is valid and just larger?
            # get_energy penalizes uncovered words heavily.
            # If best_code was invalid, its energy is >= L.
            # Greedy energy is len(code).
            # So if best_code is invalid, greedy is likely better (lower energy).
            # But we tracked best_energy.
            # If best_energy < L, then best_code was valid at some point.
            # Wait, if best_code is the state with min energy, and it's invalid,
            # then all valid states had higher energy? Unlikely.
            # Most likely, best_code IS valid.
            # Let's just check.
            pass
        
        # To be safe, if the tracked best_code is invalid, we return the greedy code.
        # But we need to be sure greedy code is valid.
        # Let's just return the first valid code found.
        # Actually, the greedy code `code` is valid.
        # Let's compare sizes.
        if len(code) < len(best_code):
            best_code = code[:]
        else:
            # Try to repair best_code? No, just use greedy if best is invalid.
            # But how do I know best is invalid? I checked final_coverage above.
            # So if final_coverage has 0s, best_code is invalid.
            # In that case, we should return `code` (greedy).
            best_code = code[:]
    
    # Convert indices to words
    return [idx_to_word(idx) for idx in best_code]

if __name__ == "__main__":
    # Quick self-test
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")

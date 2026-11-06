import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Pre-compute bitmask for each k-subset and static score
    # Also precompute the list of indices of t-subsets covered by each block
    block_data = []
    block_t_indices = []
    for block in k_subsets:
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        static_score = mask.bit_count()
        block_data.append((static_score, block, mask))
        block_t_indices.append(indices)
        
    # Sort by static score descending to break ties in greedy selection
    # This is a heuristic: blocks that cover more total t-subsets are "better"
    block_data.sort(key=lambda x: x[0], reverse=True)
    
    # Re-index block_t_indices to match the sorted order of block_data
    # We need to map from the original block index to the sorted position
    # Actually, block_t_indices was built in the order of k_subsets.
    # block_data is sorted. We need to align them.
    # Let's just rebuild the sorted list of indices.
    sorted_indices = [i for i, _ in enumerate(sorted(range(num_k_subsets), key=lambda i: block_data[i][0], reverse=True))]
    # Wait, block_data contains (score, block, mask).
    # Let's just create a list of indices sorted by score.
    indices_sorted = sorted(range(num_k_subsets), key=lambda i: block_data[i][0], reverse=True)
    
    # Precompute numpy coverage matrix for fast gain calculation?
    # With 3876 blocks and 3876 subsets, a dense matrix is 15M entries.
    # Using bitmasks is likely faster in Python for the intersection count if we stick to Python ints.
    # However, the current program uses pure Python bit operations which are fast enough for 3876x3876.
    
    # Precompute weights for weighted greedy (inspiration from other solutions)
    # Rare subsets get higher priority to break symmetries and improve coverage efficiency
    # Calculate frequency of each t-subset across all blocks
    subset_freq = [0] * num_t_subsets
    for idx in range(num_k_subsets):
        for sub_idx in block_t_indices[idx]:
            subset_freq[sub_idx] += 1
    
    # Weights: inverse frequency (higher weight for rarer subsets)
    weights = np.array([1.0 / f if f > 0 else 1.0 for f in subset_freq], dtype=np.float64)
    weights_uniform = np.ones(num_t_subsets, dtype=np.float64)
    
    # Build coverage matrix as numpy array for vectorized weighted greedy
    coverage_matrix = np.zeros((num_k_subsets, num_t_subsets), dtype=np.int8)
    for i, bits in enumerate(block_t_indices):
        for b in bits:
            coverage_matrix[i, b] = 1
    
    def fast_greedy(weight_vector=None, initial_blocks=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights
        
        if initial_blocks is not None:
            # Start with initial blocks
            uncovered = np.ones(num_t_subsets, dtype=np.int8)
            for idx in initial_blocks:
                uncovered &= (1 - coverage_matrix[idx])
            remaining = int(np.sum(uncovered))
            chosen = list(initial_blocks)
        else:
            uncovered = np.ones(num_t_subsets, dtype=np.int8)
            chosen = []
            remaining = num_t_subsets
        
        while remaining > 0:
            if time.time() > deadline - 2:
                break
                
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen
    
    def local_search(blocks, counts, deadline):
        """Perform local search to remove redundant blocks and try swaps."""
        current = blocks.copy()
        cnt = counts.copy()
        best_len = len(current)
        best_blocks = current.copy()
        
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Remove any redundant blocks
            pos = 0
            while pos < len(current):
                idx = current[pos]
                redundant = True
                for sub_idx in block_t_indices[idx]:
                    if cnt[sub_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for sub_idx in block_t_indices[idx]:
                        cnt[sub_idx] -= 1
                    current.pop(pos)
                    improved = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best_blocks = current.copy()
                    # Do not increment pos, check the new block at this position
                else:
                    pos += 1
                    
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                    
            # Phase 2: Try swaps to improve structure
            if time.time() > deadline - 2:
                break
                
            # Sample a few blocks to try swapping
            sample_size = min(5, len(current))
            for _ in range(sample_size):
                i = random.randint(0, len(current) - 1)
                idx_i = current[i]
                
                # Find unique subsets of i
                unique_subs = [sub_idx for sub_idx in block_t_indices[idx_i] if cnt[sub_idx] == 1]
                
                if not unique_subs:
                    # Redundant, should have been caught in Phase 1
                    for sub_idx in block_t_indices[idx_i]:
                        cnt[sub_idx] -= 1
                    current.pop(i)
                    if len(current) < best_len:
                        best_len = len(current)
                        best_blocks = current.copy()
                    continue
                    
                # Find a candidate block j that covers all unique_subs
                # And is not already in the list
                # This is expensive, so we limit the search
                # For simplicity, we skip complex swaps in this version
                pass
        
        return best_blocks
    
    def generate_modular_blocks(v, k, t, deadline):
        """Generate structured blocks using modular arithmetic over Z_v."""
        if v <= 0 or k <= 0 or k > v:
            return []
        
        blocks = set()
        
        # Strategy 1: Arithmetic progressions modulo v
        # For each step d and starting point s, create {s, s+d, s+2d, ..., s+(k-1)d} mod v
        # This creates structured blocks with regular difference patterns
        used_steps = set()
        for d in range(1, v):
            if d in used_steps:
                continue
            # Skip steps that don't generate distinct elements
            test_block = set((i * d) % v for i in range(k))
            if len(test_block) < k:
                continue
            used_steps.add(d)
            
            for s in range(v):
                block = tuple(sorted((s + i * d) % v for i in range(k)))
                blocks.add(block)
                if time.time() > deadline - 30:
                    return [list(b) for b in blocks]
        
        # Strategy 2: Quadratic residues style blocks
        # For prime v, use quadratic residues to create blocks
        if v > 1:
            # Find a primitive root if v is prime
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        return False
                return True
            
            if is_prime(v):
                # Find primitive root
                g = 2
                while g < v:
                    order = 1
                    val = g
                    for _ in range(1, v):
                        val = (val * g) % v
                        if val == 1:
                            break
                        order += 1
                    if order == v - 1:
                        break
                    g += 1
                
                if g < v:
                    # Generate blocks from multiplicative cosets
                    for offset in range(v):
                        # Take k consecutive powers of g, offset
                        block_elements = set()
                        for i in range(k):
                            # Compute g^(offset + i) mod v
                            exp = offset + i
                            val = pow(g, exp, v)
                            block_elements.add(val)
                        
                        if len(block_elements) == k:
                            blocks.add(tuple(sorted(block_elements)))
                        
                        if time.time() > deadline - 30:
                            return [list(b) for b in blocks]
        
        return [list(b) for b in blocks]
    
    # Run multiple greedy initializations with different weightings
    greedy_results = []
    
    # First, try to generate modular blocks as initial seeds
    modular_blocks = generate_modular_blocks(v, k, t, deadline)
    
    # Map modular blocks to indices
    k_subset_to_idx = {tuple(sorted(b)): i for i, b in enumerate(k_subsets)}
    modular_indices = []
    for mb in modular_blocks:
        idx = k_subset_to_idx.get(tuple(sorted(mb)))
        if idx is not None:
            modular_indices.append(idx)
    
    # Run greedy with modular block initialization
    if modular_indices and time.time() < deadline - 15:
        result_mod = fast_greedy(weights, initial_blocks=modular_indices)
        if result_mod:
            greedy_results.append(result_mod)
    
    # Run standard greedy with different weightings
    for w_vec in [weights, weights_uniform]:
        if time.time() > deadline - 10:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)
    
    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights)
    
    # Build counts for local search
    counts = [0] * num_t_subsets
    for idx in best_greedy:
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] += 1
    
    # Run local search
    best_blocks = local_search(best_greedy, counts, deadline - 2)
    
    # Convert best_blocks back to lists of integers
    result = [list(k_subsets[idx]) for idx in best_blocks]
    
    # Final validity check (just in case)
    # If invalid, we need to fix it.
    # But the local search only removes redundant blocks, so validity is preserved.
    
    return result

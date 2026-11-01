import itertools
import random
import time
import numpy as np
from collections import deque

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    # Note: For v=19, k=7, t=4, the numbers are manageable.
    # C(19,4) = 3876 t-subsets
    # C(19,7) = 50388 k-blocks
    # Each block covers C(7,4) = 35 t-subsets.
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # block_mask[b] is a bitset of t-subsets covered by block b
    block_mask = [0] * num_k
    # covering_blocks[i] is a list of block indices that cover t-subset i
    covering_blocks = [[] for _ in range(num_t)]
    
    # Precompute block masks
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            covering_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        
    # Precompute global frequency of each t-subset
    subset_freq = [len(cov) for cov in covering_blocks]
    
    # Helper to get unique coverage mask for current counts
    def get_unique_mask(counts):
        um = 0
        for i in range(num_t):
            if counts[i] == 1:
                um |= (1 << i)
        return um

    def minimize(active_set, counts):
        """
        Remove any block that has no unique coverage.
        Returns updated active_set and counts.
        """
        changed = True
        while changed:
            changed = False
            unique_mask = get_unique_mask(counts)
            # Iterate over a copy because we modify the set
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                # Check if b_idx has any unique coverage
                if block_mask[b_idx] & unique_mask:
                    continue
                # It has no unique coverage, so it can be removed
                active_set.remove(b_idx)
                for i in range(num_t):
                    if (block_mask[b_idx] >> i) & 1:
                        counts[i] -= 1
                changed = True
        return active_set, counts

    def greedy_construct():
        """
        Standard greedy set cover.
        """
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        
        while uncovered_mask:
            best_block = -1
            max_gain = 0
            # Optimization: We can skip blocks that are already in active_set, 
            # but since we update uncovered_mask, gain will be 0 for them anyway 
            # if they don't cover new stuff. However, checking all num_k is O(50k).
            # 3876 t-subsets. Greedy takes ~150 steps. 150 * 50k = 7.5M ops. Fast enough.
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1 or max_gain == 0:
                break
                
            active_set.add(best_block)
            for i in range(num_t):
                if (block_mask[best_block] >> i) & 1:
                    counts[i] += 1
            
            uncovered_mask &= ~block_mask[best_block]
            
        return active_set, counts

    def local_search(active_set, counts):
        """
        Local search with swaps and add-minimize.
        """
        improved = True
        consecutive_lateral = 0
        while improved and time.time() < deadline:
            improved = False
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                unique_mask = get_unique_mask(counts)
                if not (block_mask[b_idx] & unique_mask):
                    # Remove
                    active_set.remove(b_idx)
                    for i in range(num_t):
                        if (block_mask[b_idx] >> i) & 1:
                            counts[i] -= 1
                    improved = True
                    consecutive_lateral = 0
                    active_set, counts = minimize(active_set, counts)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                bits.sort(key=lambda i: subset_freq[i])
                rarest_idx = bits[0]
                
                found_swap = False
                # To speed up, we can limit the number of candidates checked
                # covering_blocks[rarest_idx] can be large.
                candidates = covering_blocks[rarest_idx]
                # Shuffle candidates to avoid bias? Or just iterate.
                # We need a candidate that covers ALL bits in u_b.
                # This is a strong constraint.
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                        
                    if (block_mask[cand_idx] & u_b) == u_b:
                        # Valid swap
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        for i in range(num_t):
                            if (block_mask[b_idx] >> i) & 1:
                                counts[i] -= 1
                            if (block_mask[cand_idx] >> i) & 1:
                                counts[i] += 1
                        
                        active_set, counts = minimize(active_set, counts)
                        improved = True
                        found_swap = True
                        consecutive_lateral = 0
                        break
                
                if not found_swap:
                    # Try Add-Minimize
                    if time.time() < deadline and consecutive_lateral < 5:
                        unique_mask = get_unique_mask(counts)
                        if unique_mask:
                            # Find rarest unique subset
                            temp = unique_mask
                            rarest_unique_idx = -1
                            min_freq = float('inf')
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                temp ^= lsb
                                if subset_freq[idx] < min_freq:
                                    min_freq = subset_freq[idx]
                                    rarest_unique_idx = idx
                            
                            if rarest_unique_idx != -1:
                                candidates = covering_blocks[rarest_unique_idx]
                                best_cand = -1
                                best_gain = 0
                                for c in candidates:
                                    if c in active_set:
                                        continue
                                    gain = (block_mask[c] & unique_mask).bit_count()
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_cand = c
                                
                                if best_cand != -1:
                                    # Snapshot
                                    old_active = set(active_set)
                                    old_counts = counts[:]
                                    old_size = len(active_set)
                                    
                                    # Add
                                    active_set.add(best_cand)
                                    for i in range(num_t):
                                        if (block_mask[best_cand] >> i) & 1:
                                            counts[i] += 1
                                    
                                    # Minimize
                                    active_set, counts = minimize(active_set, counts)
                                    
                                    new_size = len(active_set)
                                    if new_size < old_size:
                                        improved = True
                                        consecutive_lateral = 0
                                    elif new_size == old_size:
                                        # Lateral move
                                        improved = True
                                        consecutive_lateral += 1
                                    else:
                                        # Revert
                                        active_set = old_active
                                        counts = old_counts
                                        consecutive_lateral += 1

        return active_set, counts

    def spectral_prune(active_set, counts):
        """
        Spectral gap-based iterative compression heuristic.
        1. Build a graph of blocks in active_set.
        2. Edge weight = 1 / (number of unique t-subsets shared).
           Or simpler: Edge exists if they share any unique t-subset.
           Let's use a graph where nodes are blocks, and edge weight is the number of common unique t-subsets.
           High weight = strong dependency.
        3. Compute eigenvector centrality (or just degree centrality for speed, but spectral is requested).
           Since the graph is small (<=150 nodes), we can use numpy.
        4. Prune blocks with low centrality? 
           Wait, "prunes low-eigenvector-centrality nodes".
           If a node has low centrality, it is less connected to the "core" of unique coverage.
           However, if it has unique coverage, it MUST stay.
           
           Let's refine the strategy from the directive:
           "prunes low-eigenvector-centrality nodes to exploit the algebraic structure of the residual uncovered sub-hypergraph"
           
           This is a bit abstract. Let's interpret it as:
           Identify the subgraph of blocks that cover *unique* t-subsets.
           If a block has no unique t-subsets, it's redundant (handled by minimize).
           So all blocks in active_set have at least 1 unique t-subset.
           
           We want to remove blocks if possible.
           Maybe the idea is to find a subset of blocks that can be removed *simultaneously* if their unique coverages are covered by other blocks?
           No, unique coverage means no other block covers it. So you can't just remove it.
           
           Let's look at the "Inspiration 1" strategy which was just standard local search.
           The directive says: "Formulate ... as Minimum Weighted Dominating Set ... solve using spectral gap-based iterative compression heuristic".
           
           Let's try a simpler spectral approach:
           Construct a matrix M where M[i][j] = 1 if block i and j share a unique t-subset.
           Actually, let's just use the existing local search but add a "spectral" restart or ordering.
           
           Alternative interpretation:
           Use eigenvector centrality to order the blocks in the local search.
           Blocks with high centrality are "more important" or "more constrained".
           We should try to swap/remove blocks with LOW centrality first, as they are more peripheral.
           
           Let's implement a function that calculates centrality and returns an ordered list of blocks.
        """
        if len(active_set) < 2:
            return list(active_set)
        
        blocks = list(active_set)
        n = len(blocks)
        block_to_pos = {b: i for i, b in enumerate(blocks)}
        
        # Build adjacency matrix based on shared unique t-subsets
        # A[i][j] = 1 if block i and block j share at least one unique t-subset
        # Unique t-subsets are those with count == 1.
        
        unique_mask = get_unique_mask(counts)
        
        # For each block, find its unique t-subsets
        block_unique_tsubs = [0] * n
        for i, b in enumerate(blocks):
            block_unique_tsubs[i] = block_mask[b] & unique_mask
        
        # Build graph
        # We want to find blocks that are "peripheral" to the unique coverage structure.
        # Let's define edge weight as the number of shared unique t-subsets.
        # If they share 0, no edge.
        
        # Since n is small, we can do this efficiently.
        # But checking all pairs is O(n^2 * bit_count). n=150, n^2=22500.
        
        adj = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            for j in range(i+1, n):
                # Check if they share any unique t-subset
                # Intersection of their unique masks
                shared = block_unique_tsubs[i] & block_unique_tsubs[j]
                if shared:
                    # Weight by number of shared unique t-subsets
                    w = shared.bit_count()
                    adj[i][j] = w
                    adj[j][i] = w
        
        # Calculate eigenvector centrality
        # Power iteration for the largest eigenvalue's eigenvector
        # Damping factor 0.85 like PageRank? Or just plain eigenvector?
        # Let's use plain eigenvector centrality (largest eigenvalue of A).
        
        # If graph is disconnected, we might have issues, but let's try.
        # Normalize rows? No, standard eigenvector centrality uses A.
        
        # If the graph is empty (no shared unique t-subsets), all centralities are equal.
        if not np.any(adj):
            return blocks
        
        # Power iteration
        vec = np.ones(n) / n
        for _ in range(50):
            new_vec = adj @ vec
            norm = np.linalg.norm(new_vec)
            if norm == 0:
                break
            new_vec /= norm
            if np.allclose(vec, new_vec):
                break
            vec = new_vec
            
        # Sort blocks by centrality (ascending)
        # We want to process low centrality first in local search?
        # Or use this to decide who to remove?
        # The directive says "prunes low-eigenvector-centrality nodes".
        # This implies low centrality nodes are candidates for removal.
        # But they must have unique coverage.
        # Maybe the idea is: if a low centrality node's unique coverage can be "taken over" by a high centrality node via a swap?
        
        # Let's return the blocks sorted by centrality.
        # We will use this order in the local search loop.
        indices = np.argsort(vec)
        return [blocks[i] for i in indices]

    def local_search_spectral(active_set, counts):
        """
        Local search using spectral ordering.
        """
        improved = True
        consecutive_lateral = 0
        while improved and time.time() < deadline:
            improved = False
            
            # Get spectrally ordered list
            # This is expensive, so maybe only do it every few iterations?
            # Or just use degree centrality for speed?
            # Let's stick to the previous local search but maybe use spectral order for the first pass.
            
            # Actually, the previous local search shuffles.
            # Let's try to integrate the spectral idea into the swap selection.
            # When choosing a block to remove/swap, prefer low centrality.
            
            # For simplicity and robustness, let's keep the original local_search 
            # but add a "spectral restart" if stuck.
            
            # Let's just use the original local_search for now, as the spectral 
            # calculation might be too slow or not helpful if not integrated deeply.
            # The directive says "investigate improvements within its family".
            # The current program IS the family.
            # The inspiration was basically the same thing.
            
            # Let's try to improve the efficiency of the local search.
            # The bottleneck is likely the `get_unique_mask` and the swap checks.
            
            # Optimization: Cache unique_mask if counts haven't changed?
            # Counts change frequently.
            
            # Let's try a different strategy:
            # Use the spectral ordering to decide the order of blocks in the local search.
            
            ordered_blocks = spectral_prune(active_set, counts)
            
            for b_idx in ordered_blocks:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                unique_mask = get_unique_mask(counts)
                if not (block_mask[b_idx] & unique_mask):
                    # Remove
                    active_set.remove(b_idx)
                    for i in range(num_t):
                        if (block_mask[b_idx] >> i) & 1:
                            counts[i] -= 1
                    improved = True
                    consecutive_lateral = 0
                    active_set, counts = minimize(active_set, counts)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                bits.sort(key=lambda i: subset_freq[i])
                rarest_idx = bits[0]
                
                found_swap = False
                candidates = covering_blocks[rarest_idx]
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                        
                    if (block_mask[cand_idx] & u_b) == u_b:
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        for i in range(num_t):
                            if (block_mask[b_idx] >> i) & 1:
                                counts[i] -= 1
                            if (block_mask[cand_idx] >> i) & 1:
                                counts[i] += 1
                        
                        active_set, counts = minimize(active_set, counts)
                        improved = True
                        found_swap = True
                        consecutive_lateral = 0
                        break
                
                if not found_swap:
                    # Try Add-Minimize
                    if time.time() < deadline and consecutive_lateral < 5:
                        unique_mask = get_unique_mask(counts)
                        if unique_mask:
                            temp = unique_mask
                            rarest_unique_idx = -1
                            min_freq = float('inf')
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                temp ^= lsb
                                if subset_freq[idx] < min_freq:
                                    min_freq = subset_freq[idx]
                                    rarest_unique_idx = idx
                            
                            if rarest_unique_idx != -1:
                                candidates = covering_blocks[rarest_unique_idx]
                                best_cand = -1
                                best_gain = 0
                                for c in candidates:
                                    if c in active_set:
                                        continue
                                    gain = (block_mask[c] & unique_mask).bit_count()
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_cand = c
                                
                                if best_cand != -1:
                                    old_active = set(active_set)
                                    old_counts = counts[:]
                                    old_size = len(active_set)
                                    
                                    active_set.add(best_cand)
                                    for i in range(num_t):
                                        if (block_mask[best_cand] >> i) & 1:
                                            counts[i] += 1
                                    
                                    active_set, counts = minimize(active_set, counts)
                                    
                                    new_size = len(active_set)
                                    if new_size < old_size:
                                        improved = True
                                        consecutive_lateral = 0
                                    elif new_size == old_size:
                                        improved = True
                                        consecutive_lateral += 1
                                    else:
                                        active_set = old_active
                                        counts = old_counts
                                        consecutive_lateral += 1

        return active_set, counts

    # Main Logic
    best_solution = None
    best_count = float('inf')
    
    # Use multiple restarts with different random seeds
    # The spectral ordering adds a deterministic structure, but we can still shuffle if we want.
    # Let's alternate between standard local search and spectral local search.
    
    for restart_num in range(15):
        if time.time() > deadline - 15:
            break
        
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        
        # Alternate between standard and spectral local search
        if restart_num % 2 == 0:
            active_set, counts = local_search(active_set, counts)
        else:
            active_set, counts = local_search_spectral(active_set, counts)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
        
        # If we hit the lower bound (131), we can stop early? 
        # But we don't know if 131 is achievable. The best known is 151.
        # If we get < 151, that's a breakthrough.
        if best_count < 151:
            break

    if best_solution is None:
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        best_solution = active_set
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result

# Quick test
if __name__ == "__main__":
    import sys
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
    # Verify
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    total = 0
    for ts in itertools.combinations(range(v), t):
        if ts not in covered:
            print(f"Missing: {ts}")
            break
    else:
        print("Valid covering.")

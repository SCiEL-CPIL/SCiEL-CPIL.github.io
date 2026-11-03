import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence data
    # block_tsubs: list of t-subset indices for each block
    # tsub_blocks: list of block indices for each t-subset
    block_tsubs = []
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            tsubs.append(idx)
            tsub_blocks[idx].append(bi)
        block_tsubs.append(tsubs)

    best_blocks = None
    best_len = float('inf')

    def time_left():
        return deadline - time.time()

    def remove_redundant(blocks):
        """Remove blocks that do not cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Map block index in 'blocks' list to its t-subsets
        # We need to track which blocks are in the current solution
        # For efficiency, we work with indices into all_blocks
        block_indices = [all_blocks.index(b) for b in blocks]
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for bi in block_indices:
            tsubs = block_tsubs[bi]
            block_tsub_indices.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        return [blocks[i] for i in range(len(blocks)) if kept[i]]

    def local_search_polish(blocks, seed=0):
        """Apply 1-for-1 swaps and 2-for-1 removals to reduce block count."""
        rng = random.Random(seed)
        
        # Convert to indices for faster lookup
        block_indices = [all_blocks.index(b) for b in blocks]
        block_set = set(block_indices)
        
        improved = True
        while improved and time_left() > 5:
            improved = False
            
            # 1. Remove redundant blocks
            new_blocks = remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                block_indices = [all_blocks.index(b) for b in blocks]
                block_set = set(block_indices)
                improved = True
                continue
            
            # 2. Try 2-for-1 removals
            # For each pair of blocks (i, j), check if their unique subsets can be covered by a single block k
            # This is expensive O(N^2 * N_blocks * t). 
            # Optimization: Only try pairs that are "close" or just sample pairs?
            # Given N ~ 150, N^2 = 22500. For each pair, we need to find a block k.
            # We can precompute unique subsets for each block.
            
            # Calculate unique subsets for each block in the current solution
            # unique_tsubs[i] = set of t-subset indices covered by block i but no other block in solution
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in block_indices:
                for idx in block_tsubs[bi]:
                    cover_count[idx] += 1
            
            unique_tsubs = [set() for _ in range(len(blocks))]
            for i, bi in enumerate(block_indices):
                for idx in block_tsubs[bi]:
                    if cover_count[idx] == 1:
                        unique_tsubs[i].add(idx)
            
            # Try to find a pair (i, j) such that unique_tsubs[i] U unique_tsubs[j] is covered by some block k not in solution
            # To make this faster, we can iterate through all blocks k not in solution and see if it covers the union of unique subsets of some pair.
            # Actually, iterating pairs is better if we can quickly check if a block k covers a set S.
            
            # Let's try a heuristic: For each block k not in solution, check if it can replace a pair.
            # This is still expensive. Let's stick to 1-for-1 swaps first, which are cheaper.
            
            # 3. Try 1-for-1 swaps
            # For each block i in solution, find its unique subsets U_i.
            # Find a block j not in solution such that U_i is a subset of T_j (t-subsets of j).
            # If found, swap i with j.
            
            # We need a fast way to check if U_i subset of T_j.
            # |U_i| is usually small.
            
            swapped = False
            for i in range(len(blocks)):
                if time_left() < 5: break
                U_i = unique_tsubs[i]
                if not U_i:
                    # If U_i is empty, block i is redundant, which should have been caught by remove_redundant.
                    # But if not, we can swap it with anything or remove it.
                    # remove_redundant handles this.
                    continue
                
                # We want to find j not in block_set such that U_i <= T_j
                # T_j is block_tsubs[j]
                
                # Heuristic: Check random blocks not in solution? Or scan all?
                # Scanning all 21k blocks for each of 150 blocks is 3M checks.
                # Each check is |U_i| lookups in a set.
                
                # Let's try to find a candidate j.
                # We can precompute for each t-subset s, the list of blocks covering it.
                # But U_i is a set of subsets.
                
                # Alternative: For each block j not in solution, check if it covers all of U_i.
                # To speed this up, we can check the "rarest" subset in U_i first.
                # Sort U_i by the number of blocks covering them (ascending).
                
                # Let's just scan a subset of blocks if it's too slow, or scan all if time permits.
                # Given the deadline, let's scan all blocks not in solution.
                
                # Optimization: If |U_i| == 1, we just need any block covering that one subset.
                # If |U_i| > 1, we need intersection of block lists.
                
                # Let's try a simple scan.
                found_j = None
                # To avoid checking the same j repeatedly for different i, we could maintain a list, but for now, just scan.
                
                # We can break early if we find a good j.
                # What makes a j "good"? Maybe one that covers many other unique subsets of other blocks?
                # For now, any j that works is an improvement in structure.
                
                # Iterate through blocks not in solution
                # We can shuffle the order to avoid bias
                candidates = [j for j in range(n_blocks) if j not in block_set]
                rng.shuffle(candidates)
                
                for j in candidates:
                    # Check if T_j covers U_i
                    T_j = block_tsubs[j]
                    # T_j is a list. Convert to set? Or just check membership?
                    # |U_i| is small. |T_j| is C(7,4)=35.
                    # Checking if U_i subset of T_j:
                    # for u in U_i: if u not in T_j_set: break
                    
                    # To make this faster, we can precompute T_j as a set? 
                    # That's 21k sets of size 35. Memory is fine.
                    # But creating them on the fly is slow.
                    
                    # Let's just do a linear scan in T_j for each u in U_i.
                    # 35 * |U_i| operations.
                    
                    covers_all = True
                    T_j_set = set(T_j) # This is expensive inside the loop.
                    # Better: Precompute sets for all blocks?
                    # Let's do that outside if needed, or just accept the cost.
                    # Actually, let's just check membership in the list.
                    # Since |T_j| is small (35), linear search is okay.
                    
                    for u in U_i:
                        if u not in T_j:
                            covers_all = False
                            break
                            
                    if covers_all:
                        found_j = j
                        break
                        
                if found_j is not None:
                    # Swap i with found_j
                    blocks[i] = list(all_blocks[found_j])
                    block_indices[i] = found_j
                    block_set.remove(block_indices[i])
                    block_set.add(found_j)
                    improved = True
                    swapped = True
                    # Recalculate unique_tsubs and cover_count? 
                    # Yes, because the structure changed.
                    # We can break out of the i-loop and restart the while loop to recalculate.
                    break
                
            if swapped:
                continue
            
            # If no swaps, try 2-for-1?
            # Given time constraints, let's skip 2-for-1 for now and rely on multiple restarts with different LP duals.
            
        return blocks

    def lp_guided_greedy(duals, seed=0):
        rng = random.Random(seed)
        covered = np.zeros(n_tsubsets, dtype=bool)
        picked = np.zeros(n_blocks, dtype=bool)
        blocks = []
        
        # Initialize weights based on duals
        block_weights = np.zeros(n_blocks)
        for j in range(n_blocks):
            if block_tsubs[j]:
                block_weights[j] = np.sum(duals[block_tsubs[j]])
                
        # Heap of (-weight, index)
        heap = [(-block_weights[j], j) for j in range(n_blocks) if block_weights[j] > 1e-9]
        heapq.heapify(heap)
        
        while heap and time_left() > 5:
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            # Check if weight is stale (should have decreased)
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                continue
            
            # Check if it covers any uncovered subset
            covers_uncovered = False
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covers_uncovered = True
                    break
            if not covers_uncovered:
                continue
                
            picked[j] = True
            blocks.append(list(all_blocks[j]))
            
            # Update weights
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in tsub_blocks[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-9:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
            
            if not np.any(~covered):
                break
        
        # Fallback for any remaining uncovered
        if not np.all(covered):
            # Simple greedy for remaining
            rem_uncovered = np.where(~covered)[0]
            uncovered_set = set(rem_uncovered)
            
            while uncovered_set and time_left() > 5:
                best_block = None
                best_count = -1
                for j in range(n_blocks):
                    count = 0
                    for s_idx in block_tsubs[j]:
                        if s_idx in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_block = j
                
                if best_block is None or best_count == 0:
                    break
                    
                blocks.append(list(all_blocks[best_block]))
                for s_idx in block_tsubs[best_block]:
                    if s_idx in uncovered_set:
                        uncovered_set.remove(s_idx)
                        
        return blocks

    # Solve LP to get duals
    duals = None
    try:
        # Build sparse matrix A for LP: M.T @ x >= 1 => -M.T @ x <= -1
        rows = []
        cols = []
        data = []
        for j in range(n_blocks):
            for s_idx in block_tsubs[j]:
                rows.append(s_idx)
                cols.append(j)
                data.append(1.0)
        
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
        
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
    except Exception as e:
        duals = np.ones(n_tsubsets)

    # Run multiple trials with different seeds and perturbations
    num_trials = 10 # Increased trials as local search is fast
    for i in range(num_trials):
        if time_left() < 15:
            break
            
        # Perturb duals slightly to break symmetry
        if i > 0:
            # Use a fixed seed for perturbation to be reproducible? No, random is fine.
            perturbation_scale = 0.2
            perturbed_duals = duals * (1 + perturbation_scale * np.random.random(n_tsubsets))
        else:
            perturbed_duals = duals
            
        blocks = lp_guided_greedy(perturbed_duals, seed=i)
        
        if blocks:
            # Polish with local search
            blocks = local_search_polish(blocks, seed=i)
            blocks = remove_redundant(blocks)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
        # If we have a very good solution, stop early
        if best_len <= 140:
            break

    if best_blocks is None:
        # Final fallback: standard greedy
        duals = np.ones(n_tsubsets)
        best_blocks = lp_guided_greedy(duals, seed=0)
        best_blocks = local_search_polish(best_blocks, seed=0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks

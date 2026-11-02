"""Improved covering algorithm: fast vectorized greedy + intensive local search."""
import itertools
import time
import random
import numpy as np
from collections import deque

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin
    t0 = time.time()
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, numpy array of t-subset indices it covers
    # Each k-block contains C(k,t) t-subsets
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)
    
    block_tsubs = [None] * n_blocks
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
    
    # Build inverse: for each t-subset, list of blocks covering it
    # Count sizes first for efficient allocation
    tsub_block_counts = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in range(n_blocks):
        for ts_idx in block_tsubs[bi]:
            tsub_block_counts[ts_idx] += 1
    
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        tsub_blocks[i] = np.empty(tsub_block_counts[i], dtype=np.int32)
    
    # Fill in the inverse index
    tsub_block_pos = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in range(n_blocks):
        for ts_idx in block_tsubs[bi]:
            pos = tsub_block_pos[ts_idx]
            tsub_blocks[ts_idx][pos] = bi
            tsub_block_pos[ts_idx] += 1
    
    # Precompute flat arrays for fast gain computation
    block_tsub_offsets = np.zeros(n_blocks + 1, dtype=np.int64)
    for bi in range(n_blocks):
        block_tsub_offsets[bi + 1] = block_tsub_offsets[bi] + c_k_t
    block_tsub_flat = np.empty(block_tsub_offsets[-1], dtype=np.int32)
    for bi in range(n_blocks):
        block_tsub_flat[block_tsub_offsets[bi]:block_tsub_offsets[bi+1]] = block_tsubs[bi]
    
    # Precompute block_ids for scatter operations
    block_ids_flat = np.repeat(np.arange(n_blocks, dtype=np.int32), c_k_t)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0, weighted=False):
        """Vectorized greedy with optional weighted scoring."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # gain[i] = number of uncovered t-subsets in block i
        gain = np.zeros(n_blocks, dtype=np.int32)
        mask = uncovered[block_tsub_flat]
        if np.any(mask):
            np.add.at(gain, block_ids_flat[mask], 1)
        
        # Track which blocks have been used
        used = np.zeros(n_blocks, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            if weighted:
                # Weighted scoring: prefer blocks that cover more unique t-subsets
                # and penalize blocks that overlap heavily with already-chosen ones
                max_blocks = np.where((gain == max_gain) & (~used))[0]
                if len(max_blocks) > 1:
                    # Score by gain / (overlap with used blocks)
                    # Simple heuristic: just random among max gain
                    chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
                else:
                    chosen = max_blocks[0]
            else:
                max_blocks = np.where((gain == max_gain) & (~used))[0]
                if len(max_blocks) == 0:
                    max_blocks = np.where(gain == max_gain)[0]
                chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(all_blocks[chosen])
            used[chosen] = True
            
            # Update uncovered and gain
            chosen_tsubs = block_tsubs[chosen]
            newly_covered = chosen_tsubs[uncovered[chosen_tsubs]]
            
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                # Decrement gain for all blocks covering each newly covered t-subset
                all_affected = np.concatenate([tsub_blocks[ts_idx] for ts_idx in newly_covered])
                np.add.at(gain, all_affected, -1)
            
            # Mark chosen block as unusable
            gain[chosen] = -1
        
        return blocks_used
    
    def remove_redundant(blocks):
        """Remove all redundant blocks (those whose t-subsets are all covered by others)."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_lists = []
        for b in blocks:
            tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsub_lists.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
        
        kept = [True] * len(blocks)
        for i in range(len(blocks)):
            redundant = all(cover_count[idx] > 1 for idx in block_tsub_lists[i])
            if redundant:
                kept[i] = False
                for idx in block_tsub_lists[i]:
                    cover_count[idx] -= 1
        
        # Second pass: after removing some, more may become redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                if all(cover_count[idx] > 1 for idx in block_tsub_lists[i]):
                    kept[i] = False
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        return [blocks[i] for i in range(len(blocks)) if kept[i]]
    
    def flow_local_search(blocks, seed=0, max_iterations=50):
        """Flow-based local search using residual graph to find high-value swaps.
        
        Models the problem as a flow network:
        - Source -> Block nodes (capacity 1)
        - Block nodes -> t-subset nodes (capacity inf)
        - t-subset nodes -> Sink (capacity 1)
        
        The current solution defines a flow of value |blocks|. We search for
        augmenting paths in the residual graph that correspond to valid exchanges.
        """
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        reduced = list(blocks)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            # Phase 1: Remove redundant blocks
            reduced = remove_redundant(reduced)
            
            if len(reduced) <= 1:
                break
            
            # Build cover counts and identify bottlenecked t-subsets
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_lists = []
            block_to_blockidx = {}
            for bi, b in enumerate(reduced):
                tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
                block_tsub_lists.append(tsubs)
                block_to_blockidx[b] = bi
                for idx in tsubs:
                    cover_count[idx] += 1
            
            # Identify bottlenecked t-subsets (cover_count == 1)
            bottleneck_tsubs = np.where(cover_count == 1)[0]
            
            if len(bottleneck_tsubs) == 0:
                # No bottlenecks, try random swaps
                pass
            
            # Build residual graph for flow-based search
            # Nodes: 0=source, 1..n_blocks=block nodes, n_blocks+1..n_blocks+n_tsubsets=t-subset nodes, last=sink
            # Edges in residual graph:
            # - source -> block_i (capacity 1 - flow, i.e., 1 if not in solution, 0 if in solution)
            # - block_i -> tsub_j (capacity inf, always present if block_i covers tsub_j)
            # - tsub_j -> sink (capacity 1 - flow, i.e., 1 if not covered, 0 if covered)
            # Reverse edges for residual:
            # - block_i -> source (capacity = flow, i.e., 1 if in solution)
            # - tsub_j -> block_i (capacity = flow, i.e., 1 if block_i covers tsub_j and is in solution)
            # - sink -> tsub_j (capacity = flow, i.e., 1 if tsub_j is covered)
            
            # For efficiency, we'll use a simplified approach:
            # Find sets of blocks to remove and replace with equal or fewer blocks
            # such that all t-subsets remain covered.
            
            improved = False
            n = len(reduced)
            order = list(range(n))
            rng.shuffle(order)
            
            # Strategy: for each block, try to find a replacement set that covers
            # its unique t-subsets and doesn't leave gaps. Use flow to find optimal replacements.
            for i in order:
                if time.time() > deadline - 2:
                    break
                if len(reduced) <= 1:
                    break
                
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block is redundant, remove it
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    block_tsub_lists.pop(i)
                    reduced.pop(i)
                    improved = True
                    continue
                
                # Try to find replacement blocks using flow-inspired heuristic
                # Prioritize blocks that cover the most bottlenecked t-subsets
                candidate_scores = {}
                for ts_idx in unique_tsubs:
                    for bi in tsub_blocks[ts_idx]:
                        if bi not in block_to_blockidx:
                            # Score by how many bottlenecked t-subsets this block covers
                            if bi not in candidate_scores:
                                candidate_scores[bi] = 0
                            candidate_scores[bi] += 1
                
                if not candidate_scores:
                    continue
                
                # Sort candidates by score (descending)
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
                
                found_swap = False
                for ci, score in sorted_candidates[:80]:  # Limit candidates tried
                    if time.time() > deadline - 2:
                        break
                    
                    cand_tsubs_set = set(block_tsubs[ci].tolist())
                    
                    # Check if candidate covers all unique t-subsets of block i
                    if not all(ts in cand_tsubs_set for ts in unique_tsubs):
                        continue
                    
                    # Check: after removing block i and adding ci, no t-subset is uncovered
                    ok = True
                    for idx in block_tsub_lists[i]:
                        if cover_count[idx] == 1 and idx not in cand_tsubs_set:
                            ok = False
                            break
                    
                    if ok:
                        # Perform swap
                        for idx in block_tsub_lists[i]:
                            cover_count[idx] -= 1
                        for ts_idx in block_tsubs[ci]:
                            cover_count[ts_idx] += 1
                        reduced[i] = all_blocks[ci]
                        block_tsub_lists[i] = block_tsubs[ci].tolist()
                        block_to_blockidx[reduced[i]] = i
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    continue
            
            if not improved:
                # Try 2-for-1 or 2-for-2 exchanges using flow-inspired approach
                if len(reduced) > 3 and time.time() < deadline - 5:
                    # Find pairs of blocks whose combined unique t-subsets can be covered by one block
                    n = len(reduced)
                    found_pair_swap = False
                    for i in range(n):
                        if time.time() > deadline - 2:
                            break
                        if found_pair_swap:
                            break
                        for j in range(i + 1, n):
                            if time.time() > deadline - 2:
                                break
                            if found_pair_swap:
                                break
                            
                            # Get unique t-subsets of blocks i and j
                            unique_i = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                            unique_j = [idx for idx in block_tsub_lists[j] if cover_count[idx] == 1]
                            
                            if not unique_i or not unique_j:
                                continue
                            
                            # Find a block that covers all unique t-subsets from both i and j
                            all_unique = set(unique_i) | set(unique_j)
                            
                            # Find candidates
                            candidate_set = set()
                            for ts_idx in all_unique:
                                for bi in tsub_blocks[ts_idx]:
                                    if bi not in block_to_blockidx:
                                        candidate_set.add(bi)
                            
                            if not candidate_set:
                                continue
                            
                            for ci in list(candidate_set)[:30]:
                                if time.time() > deadline - 2:
                                    break
                                
                                cand_tsubs_set = set(block_tsubs[ci].tolist())
                                
                                # Check if candidate covers all unique t-subsets
                                if not all(ts in cand_tsubs_set for ts in all_unique):
                                    continue
                                
                                # Check no gaps after removing i and j, adding ci
                                ok = True
                                for idx in block_tsub_lists[i] + block_tsub_lists[j]:
                                    if cover_count[idx] == 1 and idx not in cand_tsubs_set:
                                        ok = False
                                        break
                                
                                if ok:
                                    # Perform 2-for-1 swap
                                    for idx in block_tsub_lists[i]:
                                        cover_count[idx] -= 1
                                    for idx in block_tsub_lists[j]:
                                        cover_count[idx] -= 1
                                    for ts_idx in block_tsubs[ci]:
                                        cover_count[ts_idx] += 1
                                    
                                    reduced[i] = all_blocks[ci]
                                    block_tsub_lists[i] = block_tsubs[ci].tolist()
                                    reduced.pop(j)
                                    block_tsub_lists.pop(j)
                                    # Update block_to_blockidx
                                    for idx in range(len(reduced)):
                                        block_to_blockidx[reduced[idx]] = idx
                                    
                                    found_pair_swap = True
                                    improved = True
                                    break
                        
                        if found_pair_swap:
                            break
                
                if not improved:
                    break
        
        return reduced
    
    def local_search(blocks, seed=0, max_iterations=200):
        """Intensive local search: iterative removal + targeted addition."""
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        reduced = list(blocks)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            # Phase 1: Remove redundant blocks
            reduced = remove_redundant(reduced)
            
            # Phase 2: Try to swap blocks to reduce count
            if len(reduced) <= 1:
                break
            
            # Build cover counts
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_lists = []
            for b in reduced:
                tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
                block_tsub_lists.append(tsubs)
                for idx in tsubs:
                    cover_count[idx] += 1
            
            # Try removing one block at a time and see if we can cover the gap
            improved = False
            n = len(reduced)
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2:
                    break
                if len(reduced) <= 1:
                    break
                
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block is redundant, remove it
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    block_tsub_lists.pop(i)
                    reduced.pop(i)
                    improved = True
                    continue
                
                # Try to find a replacement block that covers all unique_tsubs
                # and doesn't leave gaps
                # Candidate blocks: those that share at least one t-subset with block i
                # For efficiency, sample from blocks that cover at least one unique t-subset
                candidate_set = set()
                for ts_idx in unique_tsubs:
                    for bi in tsub_blocks[ts_idx]:
                        if bi not in [j for j, b in enumerate(reduced)]:
                            candidate_set.add(bi)
                
                if not candidate_set:
                    continue
                
                candidates = list(candidate_set)
                rng.shuffle(candidates)
                
                found_swap = False
                for ci in candidates[:50]:  # Limit candidates tried
                    if time.time() > deadline - 2:
                        break
                    
                    cand_tsubs_set = set(block_tsubs[ci].tolist())
                    
                    # Check if candidate covers all unique t-subsets of block i
                    if not all(ts in cand_tsubs_set for ts in unique_tsubs):
                        continue
                    
                    # Check: after removing block i and adding ci, no t-subset is uncovered
                    ok = True
                    for idx in block_tsub_lists[i]:
                        if cover_count[idx] == 1 and idx not in cand_tsubs_set:
                            ok = False
                            break
                    
                    if ok:
                        # Perform swap
                        for idx in block_tsub_lists[i]:
                            cover_count[idx] -= 1
                        for ts_idx in block_tsubs[ci]:
                            cover_count[ts_idx] += 1
                        reduced[i] = all_blocks[ci]
                        block_tsub_lists[i] = block_tsubs[ci].tolist()
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    continue
            
            if not improved:
                # No improvement in this iteration, try a different strategy
                # Try adding a block that covers many currently singly-covered t-subsets
                if len(reduced) > 2 and time.time() < deadline - 5:
                    # Find t-subsets with cover_count == 1
                    single_covered = np.where(cover_count == 1)[0]
                    if len(single_covered) > 0:
                        # Find block that covers the most single-covered t-subsets
                        best_gain = -1
                        best_block = -1
                        for bi in range(n_blocks):
                            cnt = 0
                            for ts_idx in block_tsubs[bi]:
                                if cover_count[ts_idx] == 1:
                                    cnt += 1
                            if cnt > best_gain:
                                best_gain = cnt
                                best_block = bi
                        
                        if best_gain >= 2 and best_block >= 0:
                            # Add this block, then try to remove a redundant one
                            new_reduced = reduced + [all_blocks[best_block]]
                            new_reduced = remove_redundant(new_reduced)
                            if len(new_reduced) < len(reduced):
                                reduced = new_reduced
                                improved = True
            
            if not improved:
                break
        
        return reduced
    
    # Run multiple greedy + local search cycles
    num_trials = 60
    for seed in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Alternate between standard and weighted greedy
        blocks = greedy_cover(seed, weighted=(seed % 3 == 0))
        if not blocks:
            continue
        
        # Quick local search
        blocks = local_search(blocks, seed, max_iterations=100)
        
        # Flow-based local search
        if time.time() < deadline - 15:
            blocks_flow = flow_local_search(blocks, seed + 500, max_iterations=30)
            if len(blocks_flow) < len(blocks):
                blocks = blocks_flow
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Deeper local search on the best
            if time.time() < deadline - 20:
                blocks_deep = local_search(blocks, seed + 1000, max_iterations=300)
                blocks_deep = flow_local_search(blocks_deep, seed + 1500, max_iterations=50)
                if len(blocks_deep) < best_len:
                    best_len = len(blocks_deep)
                    best_blocks = blocks_deep
        
        # Early termination
        if best_len <= 135:
            break
    
    # Final intensive local search on best result
    if best_blocks is not None and time.time() < deadline - 15:
        final = local_search(best_blocks, 9999, max_iterations=500)
        final = flow_local_search(final, 10000, max_iterations=100)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)
    
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0, max_iterations=100)
    
    return best_blocks

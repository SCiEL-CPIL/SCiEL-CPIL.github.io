import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    t0 = time.time()
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_tsubs = [None] * n_blocks
    rows = []
    cols = []
    data = []
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        for ts_idx in tsubs:
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    # Build sparse matrix M: M[ts_idx, bi] = 1 if block bi covers ts_idx
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # Build inverse index: for each t-subset, list of blocks covering it
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        tsub_blocks[i] = M[i, :].indices.astype(np.int32)
    
    # 2. LP Relaxation to get dual weights (bottleneck indicators)
    dual_weights = None
    try:
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            dual_weights = np.abs(res.ineqlin.marginals)
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
    except Exception:
        dual_weights = np.ones(n_tsubsets)
    
    # 3. LP-Guided Greedy with bottleneck awareness
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Initial weights: sum of dual_weights of t-subsets each block covers
        block_weights = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            block_weights[bi] = np.sum(dual_weights[block_tsubs[bi]])
        
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            max_w = block_weights.max()
            if max_w <= 1e-9:
                break
                
            # Pick randomly among top 5% of candidates for diversity
            threshold = max_w * 0.95
            candidates = np.where(block_weights >= threshold)[0]
            if len(candidates) > 0:
                chosen = candidates[rng.randint(0, len(candidates)-1)]
            else:
                chosen = np.argmax(block_weights)
            
            selected_blocks.append(chosen)
            
            # Update weights
            ts_list = block_tsubs[chosen]
            active_mask = uncovered[ts_list]
            
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    blocks_to_update = tsub_blocks[ts_idx]
                    block_weights[blocks_to_update] -= w_sub
                    block_weights[blocks_to_update] = np.maximum(block_weights[blocks_to_update], 0)
            
            block_weights[chosen] = -1e10
        
        return selected_blocks

    def walksat_refine(block_indices, seed=0):
        """WalkSAT-style local search with pair-replacement."""
        rng = random.Random(seed)
        selected = set(block_indices)
        
        # Compute coverage count
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            cover_count[block_tsubs[bi]] += 1
        
        best_selected = set(selected)
        best_count = len(selected)
        
        p_remove = 0.4
        p_pair_replace = 0.3
        max_steps = 30000
        no_improve_limit = 3000
        
        step = 0
        no_improve = 0
        
        while step < max_steps and time.time() < deadline - 2:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > no_improve_limit:
                    break
            
            if not selected:
                break
            
            sel_list = list(selected)
            r = rng.random()
            
            if r < p_remove:
                # Try to remove a random block
                bi = sel_list[rng.randint(0, len(sel_list)-1)]
                ts_indices = block_tsubs[bi]
                min_count = np.min(cover_count[ts_indices])
                if min_count > 1:
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    
            elif r < p_remove + p_pair_replace:
                # Pair replacement: try to replace 2 blocks with 1
                if len(selected) < 2:
                    continue
                    
                # Pick two blocks to potentially replace
                b1, b2 = rng.sample(sel_list, 2)
                
                # Find t-subsets uniquely covered by b1 or b2
                ts1 = block_tsubs[b1]
                ts2 = block_tsubs[b2]
                
                # Combined unique coverage: t-subsets covered by exactly one of b1, b2
                # and not covered by any other selected block
                unique_mask = np.zeros(n_tsubsets, dtype=bool)
                for ts in ts1:
                    if cover_count[ts] == 1 and ts in set(ts1):
                        unique_mask[ts] = True
                for ts in ts2:
                    if cover_count[ts] == 1 and ts in set(ts2):
                        unique_mask[ts] = True
                        
                unique_tsubs = np.where(unique_mask)[0]
                
                if len(unique_tsubs) == 0:
                    # Both blocks are redundant
                    selected.discard(b1)
                    selected.discard(b2)
                    cover_count[ts1] -= 1
                    cover_count[ts2] -= 1
                    continue
                
                # Try to find a single block that covers all unique_tsubs
                # Sample candidates
                found = False
                for _ in range(30):
                    ci = rng.randint(0, n_blocks - 1)
                    if ci in selected:
                        continue
                    
                    cand_set = set(block_tsubs[ci])
                    if all(ts in cand_set for ts in unique_tsubs):
                        # Perform pair replacement
                        selected.discard(b1)
                        selected.discard(b2)
                        cover_count[ts1] -= 1
                        cover_count[ts2] -= 1
                        
                        selected.add(ci)
                        cover_count[block_tsubs[ci]] += 1
                        found = True
                        break
                
                if not found:
                    # With small probability, make a random move to escape
                    if rng.random() < 0.03:
                        ci = rng.randint(0, n_blocks - 1)
                        if ci not in selected:
                            selected.add(ci)
                            cover_count[block_tsubs[ci]] += 1
            else:
                # Single swap: remove one, add another
                bi_to_remove = sel_list[rng.randint(0, len(sel_list)-1)]
                ts_indices = block_tsubs[bi_to_remove]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    selected.discard(bi_to_remove)
                    cover_count[ts_indices] -= 1
                    continue
                
                # Find replacement
                found_swap = False
                for _ in range(20):
                    ci = rng.randint(0, n_blocks - 1)
                    if ci in selected:
                        continue
                    
                    cand_set = set(block_tsubs[ci])
                    if all(ts in cand_set for ts in unique_tsubs):
                        selected.discard(bi_to_remove)
                        cover_count[ts_indices] -= 1
                        selected.add(ci)
                        cover_count[block_tsubs[ci]] += 1
                        found_swap = True
                        break
                
                if not found_swap and rng.random() < 0.05:
                    ci = rng.randint(0, n_blocks - 1)
                    if ci not in selected:
                        selected.add(ci)
                        cover_count[block_tsubs[ci]] += 1
        
        return [all_blocks[bi] for bi in best_selected]

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 15
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        init_blocks = lp_greedy(seed)
        if not init_blocks:
            continue
        
        refined = walksat_refine(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
            
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
            max_gain = max(gain)
            if max_gain == 0:
                break
            candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
            chosen = candidates[0]
            blocks_used.append(chosen)
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks

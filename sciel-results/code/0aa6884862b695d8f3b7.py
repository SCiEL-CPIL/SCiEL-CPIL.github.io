import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = [tset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_indices)
    
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)
            
    # Convert block_tsets to numpy arrays for faster ops
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    
    # 2. Greedy Set Cover
    def run_greedy(randomize_ties=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)
        
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time_left() < 30.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            if randomize_ties:
                candidates = np.where(gain == max_gain)[0]
                bi = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                bi = int(np.argmax(gain))
            
            selected.append(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1
        
        return selected
    
    # 3. Fast reduction using numpy
    def reduce_solution(sel):
        if not sel:
            return []
        
        if time_left() < 5.0:
            return sel
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            cov[block_tsets_np[bi]] += 1
        
        new_sel = []
        for bi in sel:
            bts = block_tsets_np[bi]
            if np.any(cov[bts] == 1):
                new_sel.append(bi)
            else:
                cov[bts] -= 1
        
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
    
    # 4. Local Search with 1-out-1 and 2-out-1 swaps
    def local_search(selected, max_rounds=50):
        if not selected or time_left() < 15.0:
            return selected
        
        best_selected = list(selected)
        best_size = len(selected)
        
        for rnd in range(max_rounds):
            if time_left() < 15.0:
                break
            
            # Phase 1: Reduction
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                
            if not selected:
                break
            
            # Rebuild coverage
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1
            
            improved = False
            
            # Phase 2: 1-out-1 Swaps
            # Try to remove one block and add another
            n_swaps_to_try = min(len(selected), 20)
            for _ in range(n_swaps_to_try):
                if time_left() < 15.0:
                    break
                if not selected:
                    break
                
                idx_remove = random.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]
                
                bts_remove = block_tsets_np[bi_remove]
                # Tsets that would become uncovered if we remove bi_remove
                # These are tsets where cov[ti] == 1 and bi_remove is in tset_to_blocks[ti]
                # Since bi_remove is in selected, cov[ti] >= 1.
                # If cov[ti] == 1, it's uniquely covered by bi_remove (among selected).
                newly_uncovered = bts_remove[cov[bts_remove] == 1]
                
                if len(newly_uncovered) == 0:
                    # Block is redundant
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    improved = True
                    continue
                
                # Find candidate blocks that cover ALL newly_uncovered tsets
                # Intersect tset_to_blocks for each newly_uncovered tset
                if len(newly_uncovered) > 5: # Heuristic: too many constraints, skip
                    continue
                
                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])
                
                cand_set.discard(bi_remove)
                cand_set -= set(selected)
                
                if not cand_set:
                    continue
                
                # Pick a candidate
                cand_list = list(cand_set)
                # Prefer blocks that cover more total tsets (greedy heuristic)
                if len(cand_list) > 3:
                    scores = [(len(block_tsets[bj]), bj) for bj in cand_list]
                    scores.sort(key=lambda x: -x[0])
                    cand_list = [s[1] for s in scores[:3]]
                
                bi_add = random.choice(cand_list)
                
                # Apply swap
                bts_add = block_tsets_np[bi_add]
                
                # Check validity: all tsets in bts_remove must still be covered
                # cov[bts_remove] will decrease by 1.
                # We need cov[bts_remove] - 1 >= 1 for all ti in bts_remove that are NOT in bts_add?
                # No, if ti is in bts_add, cov[ti] increases by 1, so net change is 0.
                # If ti is NOT in bts_add, cov[ti] decreases by 1.
                # We need cov[ti] - 1 >= 1 => cov[ti] >= 2 for ti in bts_remove \ bts_add.
                # And for ti in bts_remove \cap bts_add, cov[ti] >= 1 is sufficient.
                
                # Efficient check:
                # For ti in newly_uncovered (cov==1), they MUST be in bts_add.
                # We already ensured this by construction of cand_set.
                # For ti in bts_remove not in newly_uncovered (cov>=2), if they are not in bts_add, cov becomes cov-1 >= 1. OK.
                # If they are in bts_add, cov becomes cov. OK.
                # So the swap is always valid if cand_set was constructed correctly.
                
                cov[bts_remove] -= 1
                cov[bts_add] += 1
                
                selected[idx_remove] = bi_add
                improved = True
                
                # Quick reduction after swap
                selected = reduce_solution(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                
                # Rebuild coverage for next iteration of swaps
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1
                
                if not selected:
                    break
            
            # Phase 3: 2-out-1 Swaps
            # Try to remove two blocks and add one
            if time_left() < 15.0:
                break
                
            n_pairs_to_try = min(len(selected) * (len(selected) - 1) // 2, 100)
            # Randomly sample pairs to avoid O(N^2) full scan if N is large
            if len(selected) > 50:
                pairs = []
                for _ in range(n_pairs_to_try):
                    i1 = random.randint(0, len(selected) - 1)
                    i2 = random.randint(0, len(selected) - 1)
                    if i1 != i2:
                        pairs.append((i1, i2))
            else:
                pairs = [(i1, i2) for i1 in range(len(selected)) for i2 in range(i1+1, len(selected))]
                
            for i1, i2 in pairs:
                if time_left() < 15.0:
                    break
                if len(selected) < 3:
                    break
                
                bi1 = selected[i1]
                bi2 = selected[i2]
                
                # Find tsets that are covered ONLY by bi1 or bi2
                # cov[ti] == 1 and ti in bts(bi1) OR cov[ti] == 1 and ti in bts(bi2)
                bts1 = block_tsets_np[bi1]
                bts2 = block_tsets_np[bi2]
                
                # Union of tsets covered by bi1 or bi2
                # We need the set of tsets that would become uncovered if we remove both.
                # This is { ti | cov[ti] == 1 and (ti in bts1 or ti in bts2) }
                
                # Efficiently:
                # tsets_uncovered_by_removal = (bts1[cov[bts1] == 1]) union (bts2[cov[bts2] == 1])
                # Note: if a tset is in both, and cov==1, it's counted once.
                
                u1 = bts1[cov[bts1] == 1]
                u2 = bts2[cov[bts2] == 1]
                
                if len(u1) == 0 and len(u2) == 0:
                    # Both are redundant?
                    # Try removing one
                    cov[bts1] -= 1
                    selected.pop(i1)
                    # Adjust index if i2 > i1
                    if i2 > i1:
                        i2 -= 1
                    improved = True
                    continue
                    
                # Combine unique tsets
                # Use set union for correctness, though numpy unique is faster
                # For small sets, set is fine
                unique_tsets = np.unique(np.concatenate([u1, u2]))
                
                if len(unique_tsets) == 0:
                    continue
                    
                # Find a block bi3 that covers all unique_tsets
                # Intersect tset_to_blocks for each ti in unique_tsets
                if len(unique_tsets) > 6: # Heuristic
                    continue
                    
                cand_set = set(tset_to_blocks[int(unique_tsets[0])])
                for ti in unique_tsets[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])
                
                cand_set.discard(bi1)
                cand_set.discard(bi2)
                cand_set -= set(selected)
                
                if not cand_set:
                    continue
                
                cand_list = list(cand_set)
                if len(cand_list) > 3:
                    scores = [(len(block_tsets[bj]), bj) for bj in cand_list]
                    scores.sort(key=lambda x: -x[0])
                    cand_list = [s[1] for s in scores[:3]]
                    
                bi3 = random.choice(cand_list)
                
                # Apply swap: remove bi1, bi2, add bi3
                bts3 = block_tsets_np[bi3]
                
                cov[bts1] -= 1
                cov[bts2] -= 1
                cov[bts3] += 1
                
                # Update selected list
                # Remove bi1 and bi2
                # Careful with indices
                idxs = sorted([i1, i2], reverse=True)
                for idx in idxs:
                    selected.pop(idx)
                selected.append(bi3)
                
                improved = True
                
                # Quick reduction
                selected = reduce_solution(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                
                # Rebuild coverage
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1
                
                if not improved:
                    break
                    
            if not improved:
                break
        
        return best_selected
    
    # 5. Main
    best_overall = None
    best_size_overall = float('inf')
    
    # Deterministic greedy
    if time_left() > 20.0:
        sel = run_greedy(randomize_ties=False)
        sel = reduce_solution(sel)
        sel = local_search(sel)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
    
    # Randomized greedy restarts
    n_restarts = 0
    while time_left() > 15.0 and n_restarts < 5:
        n_restarts += 1
        seed = n_restarts * 7919
        sel = run_greedy(randomize_ties=True, seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search(sel)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
            
    if best_overall is None:
        best_overall = run_greedy(randomize_ties=False)
        best_overall = reduce_solution(best_overall)
        
    # Final reduction
    best_overall = reduce_solution(best_overall)
    
    return [list(all_blocks[bi]) for bi in best_overall]

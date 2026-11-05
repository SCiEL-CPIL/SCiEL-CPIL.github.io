import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0

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
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)

    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Convert block_tsets to numpy arrays for faster ops
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    block_tsets_set = [set(ts) for ts in block_tsets]
    
    # Precompute hardness: how many blocks cover each tset
    tset_hardness = np.zeros(n_tsub, dtype=np.int32)
    for ti in range(n_tsub):
        tset_hardness[ti] = len(tset_to_blocks[ti])
    
    # Inverse hardness for weighting
    tset_inv_hardness = np.zeros(n_tsub, dtype=np.float64)
    for ti in range(n_tsub):
        if tset_hardness[ti] > 0:
            tset_inv_hardness[ti] = 1.0 / tset_hardness[ti]
        else:
            tset_inv_hardness[ti] = 0.0

    # 2. Greedy Set Cover with hardness-weighted scoring
    def run_greedy(seed=None, use_weighted=False):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)
        
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time_left() < 20.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            
            if use_weighted and len(candidates) > 1:
                # Score by sum of inverse hardness of covered tsets
                best_score = -1
                best_bi = int(candidates[0])
                for ci in candidates:
                    bts = block_tsets_np[ci]
                    score = np.sum(tset_inv_hardness[bts])
                    if score > best_score:
                        best_score = score
                        best_bi = int(ci)
                bi = best_bi
            else:
                bi = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            selected.append(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1
        
        return selected

    # 3. Fast reduction
    def reduce_solution(sel):
        if not sel:
            return []
        
        if time_left() < 3.0:
            return sel
        
        sel_set = set(sel)
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

    # 4. Local search with 1-exchange and simulated annealing
    def local_search(selected, max_rounds=80, use_sa=False, seed=None):
        if not selected or time_left() < 8.0:
            return selected
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            cov[block_tsets_np[bi]] += 1
        
        best_selected = list(selected)
        best_size = len(selected)
        current_size = len(selected)
        
        # Simulated annealing parameters
        if use_sa:
            T_start = 1.0
            T_end = 0.01
            alpha = 0.95  # cooling rate
        
        for rnd in range(max_rounds):
            if time_left() < 8.0:
                break
            
            # Reduction pass
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                current_size = len(selected)
                
            if not selected:
                break
            
            # Rebuild coverage
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1
            
            # Try multiple swap attempts
            n_attempts = min(len(selected), 60)
            improved_this_round = False
            
            for _ in range(n_attempts):
                if time_left() < 8.0:
                    break
                if not selected:
                    break
                
                idx_remove = rng.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]
                
                bts_remove = block_tsets_np[bi_remove]
                newly_uncovered = bts_remove[cov[bts_remove] == 1]
                
                if len(newly_uncovered) == 0:
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    current_size -= 1
                    improved_this_round = True
                    if current_size < best_size:
                        best_size = current_size
                        best_selected = list(selected)
                    continue
                
                # Find candidates covering all newly_uncovered
                if len(newly_uncovered) > 3:
                    continue  # Too hard to find a single replacement
                
                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])
                
                cand_set.discard(bi_remove)
                cand_set -= set(selected)
                
                if not cand_set:
                    continue
                
                cand_list = list(cand_set)
                
                # Score candidates by additional coverage
                if len(cand_list) <= 3:
                    bi_add = rng.choice(cand_list)
                else:
                    scores = []
                    bts_remove_set = set(bts_remove.tolist())
                    for bj in cand_list:
                        bts_bj = block_tsets_np[bj]
                        additional = np.sum(~np.isin(bts_bj, bts_remove))
                        scores.append((additional, bj))
                    scores.sort(key=lambda x: -x[0])
                    top_n = min(3, len(scores))
                    bi_add = rng.choice([s[1] for s in scores[:top_n]])
                
                bts_add = block_tsets_np[bi_add]
                
                cov[bts_remove] -= 1
                cov[bts_add] += 1
                
                valid = np.all(cov[bts_remove] >= 1)
                
                if not valid:
                    cov[bts_add] -= 1
                    cov[bts_remove] += 1
                    continue
                
                selected[idx_remove] = bi_add
                improved_this_round = True
                
                # Quick reduction after swap
                selected = reduce_solution(selected)
                current_size = len(selected)
                if current_size < best_size:
                    best_size = current_size
                    best_selected = list(selected)
                
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1
                
                if not selected:
                    break
            
            if not improved_this_round and rnd > 10:
                break
        
        return best_selected

    # 5. Main: multiple strategies
    best_overall = None
    best_size_overall = float('inf')

    # Strategy 1: Deterministic greedy + local search
    if time_left() > 25.0:
        sel = run_greedy(seed=None, use_weighted=True)
        sel = reduce_solution(sel)
        sel = local_search(sel, max_rounds=100, use_sa=False, seed=1)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    # Strategy 2: Randomized greedy restarts
    n_restarts = 0
    while time_left() > 20.0 and n_restarts < 10:
        n_restarts += 1
        seed = n_restarts * 7919 + 42
        sel = run_greedy(seed=seed, use_weighted=True)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search(sel, max_rounds=80, use_sa=False, seed=seed)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    # Strategy 3: Different greedy variants
    for hseed in [123, 456, 789, 1011, 1013]:
        if time_left() < 15.0:
            break
        sel = run_greedy(seed=hseed, use_weighted=False)
        sel = reduce_solution(sel)
        sel = local_search(sel, max_rounds=60, use_sa=False, seed=hseed)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    if best_overall is None:
        best_overall = run_greedy(seed=None, use_weighted=True)
        best_overall = reduce_solution(best_overall)
    
    best_overall = reduce_solution(best_overall)
    
    return [list(all_blocks[bi]) for bi in best_overall]

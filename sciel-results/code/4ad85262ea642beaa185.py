import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    tsets = list(itertools.combinations(range(v), t))
    N = len(tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(tsets)}
    
    point_tsets = [[] for _ in range(v)]
    for i, combo in enumerate(tsets):
        for x in combo:
            point_tsets[x].append(i)
            
    # Precompute tset for each block? No, blocks change.
    
    def greedy_construct(seed):
        random.seed(seed)
        coverage = np.zeros(N, dtype=np.int32)
        blocks = []
        uncovered_count = N
        
        while uncovered_count > 0:
            # Find uncovered
            while True:
                idx = random.randint(0, N - 1)
                if coverage[idx] == 0:
                    break
            
            anchor = tsets[idx]
            block_set = set(anchor)
            block_list = list(anchor)
            
            # Calculate gain
            gain = np.zeros(v, dtype=np.int32)
            for x in range(v):
                if x in block_set:
                    continue
                g = 0
                for ti in point_tsets[x]:
                    if coverage[ti] == 0:
                        g += 1
                gain[x] = g
            
            while len(block_list) < k:
                best_g = -1
                candidates = []
                for x in range(v):
                    if x not in block_set:
                        if gain[x] > best_g:
                            best_g = gain[x]
                            candidates = [x]
                        elif gain[x] == best_g:
                            candidates.append(x)
                
                if not candidates:
                    break
                
                best_x = random.choice(candidates)
                block_set.add(best_x)
                block_list.append(best_x)
                
                # Recalculate gain for next step
                if len(block_list) < k:
                    for x in range(v):
                        if x in block_set:
                            gain[x] = 0
                            continue
                        g = 0
                        for ti in point_tsets[x]:
                            if coverage[ti] == 0:
                                g += 1
                        gain[x] = g
            
            block_list.sort()
            blocks.append(block_list)
            
            for tcombo in itertools.combinations(block_list, t):
                i = tset_to_idx[tcombo]
                if coverage[i] == 0:
                    uncovered_count -= 1
                coverage[i] += 1
        
        return blocks, coverage

    def remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t):
        block_tsets = []
        for bl in blocks:
            ts = [tset_to_idx[tc] for tc in itertools.combinations(bl, t)]
            block_tsets.append(ts)
        
        changed = True
        while changed:
            changed = False
            for bi in range(len(blocks) - 1, -1, -1):
                redundant = True
                for i in block_tsets[bi]:
                    if coverage[i] <= 1:
                        redundant = False
                        break
                if redundant:
                    for i in block_tsets[bi]:
                        coverage[i] -= 1
                    blocks.pop(bi)
                    block_tsets.pop(bi)
                    changed = True
        
        return blocks, coverage, block_tsets

    def local_search(blocks, coverage, block_tsets, tsets, tset_to_idx, v, k, t, max_steps=50000):
        steps = 0
        improved = True
        while improved and steps < max_steps:
            improved = False
            steps += 1
            
            # 1. Single block redundancy
            bi = random.randint(0, len(blocks) - 1)
            holes = []
            for i in block_tsets[bi]:
                if coverage[i] == 1:
                    holes.append(i)
            
            if not holes:
                for i in block_tsets[bi]:
                    coverage[i] -= 1
                blocks.pop(bi)
                block_tsets.pop(bi)
                improved = True
                continue
            
            U = set()
            for h_idx in holes:
                U.update(tsets[h_idx])
            
            if len(U) <= k:
                remaining_points = [x for x in range(v) if x not in U]
                need = k - len(U)
                if need >= 0 and need <= len(remaining_points):
                    best_new_block = None
                    best_score = -1
                    for _ in range(5):
                        if need == 0:
                            extra = []
                        else:
                            extra = random.sample(remaining_points, need)
                        new_block_set = U.union(extra)
                        new_block = sorted(new_block_set)
                        if new_block in blocks:
                            continue
                        score = 0
                        for tc in itertools.combinations(new_block, t):
                            i = tset_to_idx[tc]
                            if coverage[i] == 2:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_new_block = new_block
                    
                    if best_new_block is not None:
                        for i in block_tsets[bi]:
                            coverage[i] -= 1
                        new_block_tsets = [tset_to_idx[tc] for tc in itertools.combinations(best_new_block, t)]
                        for i in new_block_tsets:
                            coverage[i] += 1
                        blocks[bi] = best_new_block
                        block_tsets[bi] = new_block_tsets
                        improved = True
                        # Check redundancies
                        for ri in range(len(blocks) - 1, -1, -1):
                            if ri == bi: continue
                            redundant = True
                            for i in block_tsets[ri]:
                                if coverage[i] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                for i in block_tsets[ri]:
                                    coverage[i] -= 1
                                blocks.pop(ri)
                                block_tsets.pop(ri)
                                # Don't break, keep checking

            if improved:
                continue

            # 2. Two block removal / one block addition
            if len(blocks) >= 2:
                bi = random.randint(0, len(blocks) - 1)
                bj = random.randint(0, len(blocks) - 1)
                if bi == bj:
                    continue
                
                holes = set()
                for i in block_tsets[bi]:
                    if coverage[i] == 1:
                        holes.add(i)
                    elif coverage[i] == 2 and i in block_tsets[bj]:
                        holes.add(i)
                for i in block_tsets[bj]:
                    if coverage[i] == 1:
                        holes.add(i)
                    elif coverage[i] == 2 and i in block_tsets[bi]:
                        holes.add(i)
                
                if not holes:
                    # Remove both
                    if bi < bj:
                        for i in block_tsets[bi]: coverage[i] -= 1
                        blocks.pop(bi); block_tsets.pop(bi)
                        for i in block_tsets[bj-1]: coverage[i] -= 1
                        blocks.pop(bj-1); block_tsets.pop(bj-1)
                    else:
                        for i in block_tsets[bj]: coverage[i] -= 1
                        blocks.pop(bj); block_tsets.pop(bj)
                        for i in block_tsets[bi]: coverage[i] -= 1
                        blocks.pop(bi); block_tsets.pop(bi)
                    improved = True
                    continue
                
                U = set()
                for h_idx in holes:
                    U.update(tsets[h_idx])
                
                if len(U) <= k:
                    remaining_points = [x for x in range(v) if x not in U]
                    need = k - len(U)
                    if need >= 0 and need <= len(remaining_points):
                        best_new_block = None
                        best_score = -1
                        for _ in range(5):
                            if need == 0:
                                extra = []
                            else:
                                extra = random.sample(remaining_points, need)
                            new_block_set = U.union(extra)
                            new_block = sorted(new_block_set)
                            if new_block in blocks:
                                continue
                            score = 0
                            for tc in itertools.combinations(new_block, t):
                                i = tset_to_idx[tc]
                                if coverage[i] == 2:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_new_block = new_block
                        
                        if best_new_block is not None:
                            # Remove bi, bj
                            if bi < bj:
                                for i in block_tsets[bi]: coverage[i] -= 1
                                for i in block_tsets[bj]: coverage[i] -= 1
                                blocks.pop(bi); block_tsets.pop(bi)
                                blocks.pop(bj-1); block_tsets.pop(bj-1)
                            else:
                                for i in block_tsets[bj]: coverage[i] -= 1
                                for i in block_tsets[bi]: coverage[i] -= 1
                                blocks.pop(bj); block_tsets.pop(bj)
                                blocks.pop(bi); block_tsets.pop(bi)
                            
                            # Add new
                            new_block_tsets = [tset_to_idx[tc] for tc in itertools.combinations(best_new_block, t)]
                            for i in new_block_tsets:
                                coverage[i] += 1
                            blocks.append(best_new_block)
                            block_tsets.append(new_block_tsets)
                            
                            improved = True
                            # Check redundancies
                            for ri in range(len(blocks) - 1, -1, -1):
                                redundant = True
                                for i in block_tsets[ri]:
                                    if coverage[i] <= 1:
                                        redundant = False
                                        break
                                if redundant:
                                    for i in block_tsets[ri]:
                                        coverage[i] -= 1
                                    blocks.pop(ri)
                                    block_tsets.pop(ri)

        return blocks, coverage, block_tsets

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        seed = 200 + restart * 11
        blocks, coverage = greedy_construct(seed)
        blocks, coverage, block_tsets = remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t)
        
        if time.time() < deadline - 10.0:
            blocks, coverage, block_tsets = local_search(blocks, coverage, block_tsets, tsets, tset_to_idx, v, k, t, max_steps=30000)
        
        blocks, coverage, block_tsets = remove_redundant(blocks, coverage, tset_to_idx, tsets, v, k, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        if best_count <= 310:
            break

    if best_blocks is None:
        blocks, _ = greedy_construct(0)
        return blocks

    return best_blocks

import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    # --- Strategy 1: Algebraic Construction via PSL(2,11) ---
    
    # PSL(2,11) generators acting on 12 points (0..10, 11=infinity)
    T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    S = [11, 10, 5, 7, 8, 2, 9, 3, 4, 6, 1, 0]
    
    def compose(p, q):
        return [p[q[i]] for i in range(12)]
    
    def generate_group(gens):
        e = list(range(12))
        group = {tuple(e)}
        frontier = [e]
        while frontier:
            new_frontier = []
            for g in frontier:
                for h in gens:
                    gh = compose(g, h)
                    gh_t = tuple(gh)
                    if gh_t not in group:
                        group.add(gh_t)
                        new_frontier.append(gh)
                    hg = compose(h, g)
                    hg_t = tuple(hg)
                    if hg_t not in group:
                        group.add(hg_t)
                        new_frontier.append(hg)
            frontier = new_frontier
        return [list(g) for g in group]
    
    try:
        group = generate_group([T, S])
        if time.time() > deadline:
            return []
            
        # Base block candidates
        base_block_candidates = [
            (0, 1, 3, 4, 5, 11),
            (0, 1, 2, 3, 4, 11),
            (0, 1, 2, 3, 5, 11),
            (0, 1, 2, 4, 5, 11),
            (0, 1, 3, 4, 6, 11),
            (0, 2, 3, 4, 5, 11),
        ]
        
        blocks_set = None
        for base_block in base_block_candidates:
            blocks_set = set()
            for perm in group:
                new_block = tuple(sorted(perm[i] for i in base_block))
                blocks_set.add(new_block)
            
            if len(blocks_set) == 132:
                break
            else:
                blocks_set = None
        
        if blocks_set is not None and len(blocks_set) == 132:
            # --- Refinement: Bitmask Verification ---
            # Map 5-subsets to indices 0..791
            all_5subs = list(combinations(range(12), 5))
            n_5subs = len(all_5subs)
            sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
            
            covered_mask = 0
            valid = True
            for block in blocks_set:
                block_mask = 0
                for five in combinations(block, 5):
                    idx = sub_to_idx[five]
                    bit = 1 << idx
                    if covered_mask & bit:
                        valid = False
                        break
                    block_mask |= bit
                if not valid:
                    break
                covered_mask |= block_mask
            
            if valid:
                return [list(b) for b in sorted(blocks_set)]
                
    except Exception:
        pass

    # --- Strategy 2: Fallback Greedy/Backtracking Search (Inspiration) ---
    if time.time() > deadline:
        return []
        
    points = range(12)
    blocks_list = list(combinations(points, 6))
    n_blocks = len(blocks_list)
    
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    block_masks = []
    for b in blocks_list:
        mask = 0
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Greedy with randomization
    import random
    indices = list(range(n_blocks))
    random.shuffle(indices)
    covered = 0
    current_blocks = []
    for i in indices:
        m = block_masks[i]
        if not (covered & m):
            current_blocks.append(i)
            covered |= m
            
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    # Simple Backtracking Improvement (Limited Depth/Time)
    # This is a heuristic search to try to reach 132 if greedy failed.
    # Since 132 is the max, and greedy might get stuck, we try to fill gaps.
    # However, 132 is hard to reach by search without strong heuristics.
    # Given time constraints, we return the best greedy result if algebraic failed.
    
    # If we have time, try a few restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            indices = list(range(n_blocks))
            random.shuffle(indices)
            covered = 0
            current_blocks = []
            for i in indices:
                m = block_masks[i]
                if not (covered & m):
                    current_blocks.append(i)
                    covered |= m
            if len(current_blocks) > best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks[:]
                
    return [list(blocks_list[i]) for i in best_blocks]

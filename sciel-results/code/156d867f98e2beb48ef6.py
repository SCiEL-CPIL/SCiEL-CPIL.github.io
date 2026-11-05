import itertools
import sys
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute: for each block, which t-subsets it covers
    block_covers = [[] for _ in range(n_k_blocks)]
    for i, block in enumerate(k_blocks):
        for t_sub in itertools.combinations(block, t):
            block_covers[i].append(sub_to_idx[t_sub])
    
    # Precompute: for each t-subset, which blocks cover it
    t_covered_by = [[] for _ in range(n_t_subs)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            t_covered_by[j].append(i)
    
    # 3. Greedy Set Cover
    uncovered = [1] * n_t_subs
    selected = set()
    
    while any(uncovered):
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = sum(1 for j in block_covers[i] if uncovered[j])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.add(best_idx)
        for j in block_covers[best_idx]:
            uncovered[j] = 0
    
    # 4. Local Search with dynamic weighting
    random.seed(12345)
    current = list(selected)
    
    # Build coverage count
    def get_counts(blocks):
        counts = [0] * n_t_subs
        for b in blocks:
            for j in block_covers[b]:
                counts[j] += 1
        return counts
    
    # Try to minimize by removing redundant blocks
    def minimize(blocks):
        counts = get_counts(blocks)
        changed = True
        while changed:
            changed = False
            for b in list(blocks):
                # Check if b is redundant
                redundant = True
                for j in block_covers[b]:
                    if counts[j] == 1:
                        redundant = False
                        break
                if redundant:
                    for j in block_covers[b]:
                        counts[j] -= 1
                    blocks = [x for x in blocks if x != b]
                    changed = True
                    break
        return blocks
    
    current = minimize(current)
    
    # Iterative improvement: try to swap blocks to reduce count
    best = list(current)
    best_size = len(best)
    max_iter = 3000
    stall = 0
    
    for it in range(max_iter):
        improved = False
        
        # Try to remove a block if possible
        counts = get_counts(current)
        for b in current:
            can_remove = all(counts[j] > 1 for j in block_covers[b])
            if can_remove:
                new_current = [x for x in current if x != b]
                new_current = minimize(new_current)
                if len(new_current) < best_size:
                    best = list(new_current)
                    best_size = len(best)
                current = new_current
                improved = True
                break
        
        if improved:
            stall = 0
            continue
        
        # Try swap: remove one, add another that covers lost
        counts = get_counts(current)
        swapped = False
        for b in current:
            lost = [j for j in block_covers[b] if counts[j] == 1]
            if not lost:
                continue
            # Find block that covers all lost
            for c in random.sample(range(n_k_blocks), min(100, n_k_blocks)):
                if c == b or c in current:
                    continue
                covers_lost = all(j in block_covers[c] for j in lost)
                if covers_lost:
                    new_current = [x for x in current if x != b]
                    new_current.append(c)
                    new_current = minimize(new_current)
                    if len(new_current) < best_size:
                        best = list(new_current)
                        best_size = len(best)
                    current = new_current
                    swapped = True
                    break
            if swapped:
                break
        
        if swapped:
            stall = 0
        else:
            stall += 1
            if stall > 100:
                # Perturb: add random block, then minimize
                rb = random.randint(0, n_k_blocks - 1)
                if rb not in current:
                    current.append(rb)
                current = minimize(current)
                stall = 0
    
    return [list(k_blocks[i]) for i in best]

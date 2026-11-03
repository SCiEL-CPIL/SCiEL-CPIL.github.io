import itertools
import time
import random

def construct(v, k, t):
    # 1. Precomputation
    # Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    ALL_MASK = (1 << n_t) - 1
    
    # Precompute masks for all k-subsets
    k_subs = list(itertools.combinations(range(v), k))
    masks = []
    for block in k_subs:
        mask = 0
        # For each t-subset within this block, set bit
        for sub in itertools.combinations(block, t):
            mask |= (1 << t_to_idx[sub])
        masks.append(mask)
        
    # 2. Greedy Initialization
    def greedy_init():
        uncovered = ALL_MASK
        blocks = []
        while uncovered:
            best_idx = -1
            best_count = 0
            # Find block covering most uncovered
            for i, m in enumerate(masks):
                count = (m & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                if count == 0: continue # optimization? no, just skip if 0
            
            if best_idx == -1: break # shouldn't happen
            
            blocks.append(best_idx)
            uncovered &= ~masks[best_idx]
        return blocks, uncovered

    # 3. Local Search
    def local_search(initial_blocks):
        blocks = list(initial_blocks)
        covered = 0
        for b in blocks:
            covered |= masks[b]
        uncovered = ALL_MASK ^ covered
        
        # Helper to update covered
        def recalc_covered():
            c = 0
            for b in blocks:
                c |= masks[b]
            return c

        while True:
            improved = False
            
            # Pass 1: Remove redundant
            # Try removing each block
            for i in range(len(blocks)):
                b = blocks[i]
                # Check if b is redundant
                # If covered by others, remove
                # Fast check: if masks[b] & ~covered_without_b == 0
                # But calculating covered_without_b is O(N)
                # Let's just try removing and checking if uncovered increases
                old_uncovered = uncovered
                # Remove b
                new_covered = 0
                for j, bj in enumerate(blocks):
                    if j != i:
                        new_covered |= masks[bj]
                new_uncovered = ALL_MASK ^ new_covered
                if new_uncovered == old_uncovered:
                    # Redundant, remove
                    blocks.pop(i)
                    covered = new_covered
                    improved = True
                    break # restart pass
            
            # Pass 2: Swap
            if not improved:
                # For each block, try swapping with a better one
                # Sort candidates by coverage of current uncovered
                candidates = sorted(range(len(masks)), key=lambda i: (masks[i] & uncovered).bit_count(), reverse=True)
                for i, b in enumerate(blocks):
                    # Try top 20 candidates
                    for c_idx in candidates[:20]:
                        if c_idx == b: continue
                        # Calculate new covered
                        new_covered = 0
                        for j, bj in enumerate(blocks):
                            if j == i:
                                new_covered |= masks[c_idx]
                            else:
                                new_covered |= masks[bj]
                        new_uncovered = ALL_MASK ^ new_covered
                        if new_uncovered.bit_count() < uncovered.bit_count():
                            blocks[i] = c_idx
                            covered = new_covered
                            uncovered = new_uncovered
                            improved = True
                            break
                    if improved:
                        break
            
            # Pass 3: Add block if uncovered
            if not improved and uncovered:
                # Find best block to add
                best_idx = -1
                best_count = 0
                for i, m in enumerate(masks):
                    count = (m & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                if best_idx != -1:
                    blocks.append(best_idx)
                    covered |= masks[best_idx]
                    uncovered = ALL_MASK ^ covered
                    improved = True

            if not improved:
                break

        return blocks

    # 4. Main Loop with Restarts
    best_blocks = []
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial greedy
    init_blocks, _ = greedy_init()
    best_blocks = init_blocks
    best_len = len(init_blocks)
    
    # Run local search on init
    refined = local_search(init_blocks)
    if len(refined) < best_len:
        best_blocks = refined
        best_len = len(refined)
        
    # Try restarts with random perturbations
    while time.time() < deadline:
        # Perturb best solution
        perturbed = list(best_blocks)
        # Randomly remove a few blocks
        for _ in range(3):
            if perturbed:
                perturbed.pop(random.randint(0, len(perturbed)-1))
        
        # Run local search
        refined = local_search(perturbed)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
            
    # Convert indices back to blocks
    return [list(k_subs[b]) for b in best_blocks]

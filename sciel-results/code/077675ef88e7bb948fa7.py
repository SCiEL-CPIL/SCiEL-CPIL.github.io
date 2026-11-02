import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all candidate blocks and the t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_tsubs = []  # for each block, list of t-subset indices it covers
    for b in all_blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # Precompute for each t-subset, which blocks cover it (for deletion)
    # Actually we'll compute this on the fly during deletion

    best_blocks = None
    best_count = float('inf')

    n_restarts = 40

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(1000 + restart * 7)

        # === Phase 1: Greedy construction ===
        covered = [0] * n_t  # coverage count per t-subset
        used_blocks = []  # list of block indices
        used_set = set()

        while True:
            if not time_ok():
                break

            # Find uncovered t-subset
            uncovered_idx = -1
            for i in range(n_t):
                if covered[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx == -1:
                break

            # Find best block to add: maximize newly covered t-subsets
            best_gain = -1
            best_block = -1
            best_tie = -1
            for bi in range(n_blocks):
                if bi in used_set:
                    continue
                gain = 0
                for idx in block_tsubs[bi]:
                    if covered[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    best_tie = 0
                elif gain == best_gain:
                    # tie-breaking: random
                    if random.random() < 0.3:
                        best_block = bi
                        best_tie = 1

            if best_block == -1 or best_gain <= 0:
                break

            # Add the block
            used_blocks.append(best_block)
            used_set.add(best_block)
            for idx in block_tsubs[best_block]:
                covered[idx] += 1

        # === Phase 2: Deletion of redundant blocks ===
        if time_ok():
            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(used_blocks)):
                    if not time_ok():
                        break
                    bi = used_blocks[i]
                    # Check if removing this block leaves all its t-subsets still covered
                    removable = True
                    for idx in block_tsubs[bi]:
                        if covered[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs[bi]:
                            covered[idx] -= 1
                        used_blocks.pop(i)
                        used_set.discard(bi)
                        changed = True
                        break  # restart scan

        # === Phase 3: Local search (swap) ===
        if time_ok():
            # Try to swap blocks to improve (reduce count or maintain count with better structure)
            # We'll try: for each used block, see if we can replace it with an unused block
            # that covers all the t-subsets the old block uniquely covered, while
            # potentially covering additional uncovered t-subsets (to enable later deletion)
            # Actually, a simpler local search: try to find a set of blocks that covers
            # all t-subsets with fewer blocks.
            # 
            # Alternative: iterate and try to replace one block with another that
            # covers the same "unique" t-subsets plus more.
            #
            # Let's do: for each block in used, check if there's an alternative block
            # that covers all t-subsets where current block is the sole cover,
            # and covers at least as many total t-subsets.
            
            improved = True
            max_swap_iters = 20
            for _ in range(max_swap_iters):
                if not time_ok():
                    break
                improved = False
                for i in range(len(used_blocks)):
                    if not time_ok():
                        break
                    bi = used_blocks[i]
                    # Find t-subsets where this block is the sole cover
                    unique_covered = []
                    for idx in block_tsubs[bi]:
                        if covered[idx] == 1:
                            unique_covered.append(idx)
                    
                    if not unique_covered:
                        # This block has no unique coverage - try to remove it
                        for idx in block_tsubs[bi]:
                            covered[idx] -= 1
                        used_blocks.pop(i)
                        used_set.discard(bi)
                        improved = True
                        break
                    
                    # Try to find a replacement block that covers all unique_covered
                    # and has at least as many total t-subsets
                    best_alt = -1
                    best_alt_gain = -1
                    for aj in range(n_blocks):
                        if aj in used_set:
                            continue
                        # Check if aj covers all unique_covered
                        alt_set = set(block_tsubs[aj])
                        can_replace = True
                        for idx in unique_covered:
                            if idx not in alt_set:
                                can_replace = False
                                break
                        if not can_replace:
                            continue
                        # Count how many new t-subsets aj covers (that aren't already covered by other blocks)
                        new_cover = 0
                        for idx in block_tsubs[aj]:
                            if covered[idx] == 1:  # only covered by current block
                                new_cover += 1
                        if new_cover > best_alt_gain:
                            best_alt_gain = new_cover
                            best_alt = aj
                    
                    if best_alt != -1 and best_alt_gain > len(block_tsubs[bi]):
                        # Replace: remove bi, add best_alt
                        for idx in block_tsubs[bi]:
                            covered[idx] -= 1
                        for idx in block_tsubs[best_alt]:
                            covered[idx] += 1
                        used_blocks[i] = best_alt
                        used_set.discard(bi)
                        used_set.add(best_alt)
                        improved = True
                        break
                
                if not improved:
                    break

        # === Phase 4: Another deletion pass after swaps ===
        if time_ok():
            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(used_blocks)):
                    if not time_ok():
                        break
                    bi = used_blocks[i]
                    removable = True
                    for idx in block_tsubs[bi]:
                        if covered[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs[bi]:
                            covered[idx] -= 1
                        used_blocks.pop(i)
                        used_set.discard(bi)
                        changed = True
                        break

        current_count = len(used_blocks)
        if current_count < best_count:
            best_count = current_count
            best_blocks = [all_blocks[bi] for bi in used_blocks]

        # Early stop if we've hit a good solution
        if best_count <= 145:
            break

    if best_blocks is None:
        # Fallback: simple greedy
        covered = [0] * n_t
        used_blocks = []
        used_set = set()
        while True:
            uncovered_idx = -1
            for i in range(n_t):
                if covered[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx == -1:
                break
            best_gain = -1
            best_block = -1
            for bi in range(n_blocks):
                if bi in used_set:
                    continue
                gain = sum(1 for idx in block_tsubs[bi] if covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            if best_block == -1:
                break
            used_blocks.append(best_block)
            used_set.add(best_block)
            for idx in block_tsubs[best_block]:
                covered[idx] += 1
        best_blocks = [all_blocks[bi] for bi in used_blocks]

    # Convert to list of lists
    return [list(b) for b in best_blocks]

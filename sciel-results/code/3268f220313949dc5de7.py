import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    coverage_count = [0] * n_subsets
    blocks = []
    
    def get_indices(block):
        return [subset_to_idx[s] for s in itertools.combinations(block, t)]

    def is_redundant(b):
        for s in itertools.combinations(b, t):
            if coverage_count[subset_to_idx[s]] == 1:
                return False
        return True

    def remove_block(b):
        for s in itertools.combinations(b, t):
            coverage_count[subset_to_idx[s]] -= 1
        blocks.remove(b)

    def add_block(b):
        for s in itertools.combinations(b, t):
            coverage_count[subset_to_idx[s]] += 1
        blocks.append(b)

    def minimize():
        changed = True
        while changed:
            changed = False
            for b in list(blocks):
                if is_redundant(b):
                    remove_block(b)
                    changed = True

    # Greedy Phase
    def greedy_construct():
        nonlocal blocks, coverage_count
        blocks = []
        coverage_count = [0] * n_subsets
        uncovered = set(range(n_subsets))
        
        while uncovered:
            if time.time() > deadline:
                break
            target_idx = random.choice(list(uncovered))
            target = all_t_subsets[target_idx]
            block = list(target)
            
            while len(block) < k:
                best_gain = -1
                best_p = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for p in itertools.combinations(block, t-1):
                        ns = tuple(sorted(p + (x,)))
                        if subset_to_idx[ns] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = x
                if best_p == -1: break
                block.append(best_p)
            
            block.sort()
            add_block(tuple(block))
            for s in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[s])
                
        minimize()

    best_blocks = None
    best_size = float('inf')
    
    greedy_deadline = start_time + 10.0
    while time.time() < greedy_deadline:
        greedy_construct()
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
    blocks = list(best_blocks)
    coverage_count = [0] * n_subsets
    for b in blocks:
        for s in itertools.combinations(b, t):
            coverage_count[subset_to_idx[s]] += 1
    minimize()
    
    # Local Search Phase
    # Maintain set of critical subset indices (count == 1)
    critical_set = set(i for i in range(n_subsets) if coverage_count[i] == 1)
    
    def update_critical():
        critical_set.clear()
        for i in range(n_subsets):
            if coverage_count[i] == 1:
                critical_set.add(i)

    update_critical()
    
    while time.time() < deadline:
        if not critical_set:
            # Stuck? Try random perturbation or break?
            # If no critical subsets, no block is "essential" for any single subset.
            # But we might still be able to remove blocks if they are fully redundant.
            # Minimize should have handled that.
            # So we are at a local minimum where every block covers at least one unique subset?
            # No, if critical_set is empty, every subset has count >= 2.
            # Then every block is redundant?
            # If every subset has count >= 2, then for any block B, every s in B has count >= 2.
            # So B is redundant.
            # So if critical_set is empty, Minimize should remove ALL blocks?
            # Wait. If I remove B, counts drop.
            # If I remove B, and some s in B had count 2, it becomes 1.
            # So B is NOT redundant if removing it would create a critical subset?
            # My `is_redundant` check: `if coverage_count[s] == 1: return False`.
            # If count is 2, it passes this check for that s.
            # If ALL s in B have count >= 2, `is_redundant` returns True.
            # So if critical_set is empty, ALL blocks are redundant.
            # So Minimize should empty the list.
            # But then the covering is invalid!
            # The check `is_redundant` is valid ONLY if the remaining blocks still cover everything.
            # Removing a block with all counts >= 2 is safe.
            # So if critical_set is empty, we can remove any block.
            # After removing one, counts drop, critical_set might become non-empty.
            # So Minimize will keep removing until some subset drops to 1.
            # So if we enter the loop with critical_set empty, Minimize will run and remove blocks.
            # So we just call Minimize.
            minimize()
            update_critical()
            if not critical_set:
                # This implies we removed everything?
                # If we removed everything, covering is invalid.
                # But we only remove if redundant.
                # If we have a valid covering, we can't remove the last block covering a subset.
                # So critical_set will NOT be empty if the covering is valid and minimal?
                # Actually, if critical_set is empty, we can remove a block.
                # After removal, some counts become 1.
                # So critical_set becomes non-empty.
                # So we should not be stuck here.
                continue

        # Pick a random critical subset
        target_idx = random.choice(list(critical_set))
        target = all_t_subsets[target_idx]
        
        # Build a block covering target, maximizing coverage of other critical subsets
        block = list(target)
        while len(block) < k:
            best_gain = -1
            best_p = -1
            for x in range(v):
                if x in block: continue
                gain = 0
                for p in itertools.combinations(block, t-1):
                    ns = tuple(sorted(p + (x,)))
                    idx = subset_to_idx[ns]
                    if idx in critical_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = x
            if best_p == -1: break
            block.append(best_p)
        
        block.sort()
        new_block = tuple(block)
        
        # Snapshot
        old_size = len(blocks)
        old_cov = coverage_count[:]
        old_blocks = blocks[:]
        old_crit = set(critical_set)
        
        add_block(new_block)
        minimize()
        update_critical()
        
        if len(blocks) < old_size:
            # Accepted
            pass
        else:
            blocks = old_blocks
            coverage_count = old_cov
            critical_set = old_crit

    return [list(b) for b in blocks]

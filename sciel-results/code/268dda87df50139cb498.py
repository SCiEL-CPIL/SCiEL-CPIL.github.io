import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Find a covering design C(v, k, t): choose as few k-subsets of {0,...,v-1}
    as possible so that every t-subset is contained in some chosen k-subset.
    
    Strategy: Probabilistic Method with Alteration
    1. Randomly sample k-blocks with probability p (tuned to cover all t-subsets)
    2. Use greedy completion to cover any remaining t-subsets
    3. Apply block removal (alteration) to eliminate redundant blocks
    4. Local search with swaps to improve further
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # For each point, precompute which t-subsets contain it
    # Actually, let's precompute for each k-subset, which t-subsets it covers
    # That's too many. Instead, we'll compute coverage on the fly.
    
    # Helper: get all t-subsets of a block
    def get_t_subs_of_block(block):
        return list(itertools.combinations(block, t))
    
    # Helper: check if a block is valid (all distinct, size k)
    def is_valid_block(block):
        return len(block) == k and len(set(block)) == k
    
    # Greedy completion: given current blocks, add blocks to cover remaining t-subsets
    def greedy_complete(blocks, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        # Compute uncovered t-subsets
        covered = [False] * n_t_subs
        for block in blocks:
            for sub in get_t_subs_of_block(block):
                idx = t_sub_to_idx[sub]
                covered[idx] = True
        
        uncovered = [i for i in range(n_t_subs) if not covered[i]]
        
        while uncovered and time.time() < max_time:
            # Pick the uncovered t-subset with the most "potential" to be covered
            # Use a weighted greedy: pick the t-subset that can be covered by the block
            # that covers the most other uncovered t-subsets
            
            # For efficiency, pick a random uncovered t-subset
            target_idx = random.choice(uncovered)
            target = all_t_subs[target_idx]
            
            # Find the best k-subset containing target
            # We need to add (k-t) more points to target
            remaining_points = [x for x in range(v) if x not in target]
            
            best_block = None
            best_gain = -1
            best_candidates = []
            
            # Try all combinations of (k-t) points from remaining_points
            # This is C(19-4, 7-4) = C(15,3) = 455, which is manageable
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target + extra))
                
                # Compute gain: number of uncovered t-subsets covered by this block
                gain = 0
                for sub in get_t_subs_of_block(block):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [block]
                elif gain == best_gain:
                    best_candidates.append(block)
            
            # Pick randomly from best candidates
            block = random.choice(best_candidates)
            blocks.append(list(block))
            
            # Update covered and uncovered
            for sub in get_t_subs_of_block(block):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = True
                    uncovered.remove(idx)
            
            # Periodic check
            if len(uncovered) % 1000 == 0 and time.time() > max_time:
                break
        
        return blocks
    
    # Block removal: remove redundant blocks
    def remove_redundant_blocks(blocks):
        if not blocks:
            return blocks
        
        # For each t-subset, track which blocks cover it
        t_sub_coverers = {i: [] for i in range(n_t_subs)}
        for bi, block in enumerate(blocks):
            for sub in get_t_subs_of_block(block):
                idx = t_sub_to_idx[sub]
                t_sub_coverers[idx].append(bi)
        
        # Try to remove blocks in random order
        removed = set()
        for bi in random.sample(range(len(blocks)), len(blocks)):
            if bi in removed:
                continue
            
            # Check if block bi is redundant
            # A block is redundant if every t-subset it covers is also covered by another non-removed block
            is_redundant = True
            for sub in get_t_subs_of_block(blocks[bi]):
                idx = t_sub_to_idx[sub]
                # Check if any other non-removed block covers this t-subset
                has_other = False
                for cj in t_sub_coverers[idx]:
                    if cj != bi and cj not in removed:
                        has_other = True
                        break
                if not has_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                removed.add(bi)
        
        return [b for i, b in enumerate(blocks) if i not in removed]
    
    # Local search: try swapping blocks
    def local_search(blocks, max_time=None):
        if max_time is None:
            max_time = deadline - 3.0
        
        current_blocks = list(blocks)
        current_count = len(current_blocks)
        
        # Compute coverage for current blocks
        def compute_coverage(block_list):
            covered = [False] * n_t_subs
            for block in block_list:
                for sub in get_t_subs_of_block(block):
                    idx = t_sub_to_idx[sub]
                    covered[idx] = True
            return covered
        
        covered = compute_coverage(current_blocks)
        uncovered_count = sum(1 for c in covered if not c)
        
        if uncovered_count > 0:
            # First, complete the covering
            current_blocks = greedy_complete(current_blocks, max_time)
            if time.time() > max_time:
                return current_blocks
            covered = compute_coverage(current_blocks)
            uncovered_count = sum(1 for c in covered if not c)
        
        # Now try to reduce the number of blocks
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try removing one block and adding a different one
            for i in range(len(current_blocks)):
                if time.time() > max_time:
                    break
                
                # Temporarily remove block i
                temp_blocks = current_blocks[:i] + current_blocks[i+1:]
                
                # Compute uncovered after removal
                temp_covered = compute_coverage(temp_blocks)
                temp_uncovered = [j for j in range(n_t_subs) if not temp_covered[j]]
                
                if not temp_uncovered:
                    # Block i is redundant!
                    current_blocks = temp_blocks
                    covered = temp_covered
                    improved = True
                    break
                
                # Try to find a single block that covers all temp_uncovered
                # This is expensive, so only try if few uncovered
                if len(temp_uncovered) <= 20:
                    # Find a block that covers all these uncovered t-subsets
                    # The block must contain the union of these t-subsets
                    union_set = set()
                    for idx in temp_uncovered:
                        union_set.update(all_t_subs[idx])
                    
                    if len(union_set) <= k:
                        # Check if there's a k-subset containing union_set
                        remaining = [x for x in range(v) if x not in union_set]
                        need = k - len(union_set)
                        for extra in itertools.combinations(remaining, need):
                            candidate = tuple(sorted(union_set | set(extra)))
                            # Check if this candidate covers all temp_uncovered
                            cand_covered = set()
                            for sub in get_t_subs_of_block(candidate):
                                cand_covered.add(t_sub_to_idx[sub])
                            if all(idx in cand_covered for idx in temp_uncovered):
                                # Replace block i with candidate
                                current_blocks = temp_blocks + [list(candidate)]
                                covered = compute_coverage(current_blocks)
                                improved = True
                                break
                        if improved:
                            break
        
        # Remove redundant blocks again
        current_blocks = remove_redundant_blocks(current_blocks)
        
        return current_blocks
    
    # Main algorithm
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Random initialization with probability p
    # Expected number of t-subsets covered by a random k-block: C(k,t)
    # Number of k-blocks needed to cover all: ~n_t_subs / C(k,t) * log(n_t_subs)
    # C(19,7) = 50388, C(7,4) = 35, n_t_subs = C(19,4) = 3876
    # ~3876/35 * log(3876) ≈ 110 * 8.2 ≈ 900 blocks (too many)
    # Let's use a smaller probability and rely on greedy completion
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Try different strategies
    strategies = [
        ("random_greedy", lambda: random.sample(all_k_blocks, min(100, n_k_blocks))),
        ("probabilistic", lambda: [b for b in all_k_blocks if random.random() < 0.05]),
        ("sparse_random", lambda: [b for b in all_k_blocks if random.random() < 0.02]),
    ]
    
    for strat_name, init_func in strategies:
        if time.time() > deadline:
            break
        
        # Multiple restarts per strategy
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            blocks = init_func()
            if not blocks:
                blocks = [list(random.choice(all_k_blocks))]
            
            # Convert to lists
            blocks = [list(b) for b in blocks]
            
            # Greedy complete
            blocks = greedy_complete(blocks)
            
            # Remove redundant
            blocks = remove_redundant_blocks(blocks)
            
            # Local search
            blocks = local_search(blocks)
            
            # Check validity and count
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
            
            # Early stop if we hit a good number
            if best_count <= 145:
                break
        
        if best_count <= 145:
            break
    
    # Final cleanup: ensure blocks are sorted and unique
    if best_blocks is None:
        # Fallback: pure greedy
        best_blocks = greedy_complete([])
    
    # Sort blocks for consistency
    best_blocks = [sorted(b) for b in best_blocks]
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks

import itertools
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(v, k, t):
    """
    Constructs a covering design using a constraint-based backtracking search 
    with a bottleneck heuristic and lookahead.
    """
    start_time = time.time()
    time_limit = 110
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_subset_to_blocks[idx].append(len(block_t_indices)) # Store block index
        block_t_indices.append(covered_indices)
        
    # Initial counts of available blocks for each t-subset
    avail_count = [0] * num_t_subsets
    for idx in range(num_t_subsets):
        avail_count[idx] = len(t_subset_to_blocks[idx])
        
    # State
    uncovered = set(range(num_t_subsets))
    used_blocks = [False] * num_blocks
    best_size = float('inf')
    best_solution = []
    
    # Lower bound calculation
    def get_lower_bound(uncovered_set, avail_counts):
        lb = 0
        for idx in uncovered_set:
            if avail_counts[idx] > 0:
                lb += 1.0 / avail_counts[idx]
            else:
                return float('inf')
        return lb

    def search(depth):
        nonlocal best_size, best_solution
        nonlocal avail_count, uncovered, used_blocks
        
        if time.time() - start_time > time_limit:
            return
            
        if depth >= best_size:
            return
            
        if not uncovered:
            if depth < best_size:
                best_size = depth
                best_solution = [i for i in range(num_blocks) if used_blocks[i]]
            return
            
        # Calculate lower bound
        lb = get_lower_bound(uncovered, avail_count)
        if depth + lb >= best_size:
            return
            
        # Find bottleneck: t-subset with fewest available blocks
        min_avail = float('inf')
        bottleneck_idx = -1
        
        for idx in uncovered:
            if avail_count[idx] < min_avail:
                min_avail = avail_count[idx]
                bottleneck_idx = idx
            if min_avail == 0:
                return # Dead end
                
        if bottleneck_idx == -1:
            return
            
        # Get candidates for the bottleneck
        candidates = []
        for b_idx in t_subset_to_blocks[bottleneck_idx]:
            if not used_blocks[b_idx]:
                # Score: number of currently uncovered t-subsets this block covers
                score = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in uncovered:
                        score += 1
                candidates.append((score, b_idx))
                
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Limit branching to top few to control search space
        # With lookahead, we can be more selective
        top_k = 5 
        for score, b_idx in candidates[:top_k]:
            # Lookahead: Estimate the new lower bound
            # We simulate adding this block
            
            # Save state
            old_uncovered = set(uncovered)
            old_avail = avail_count[:] # This copy is expensive! 
            # Optimization: Instead of copying the whole array, we can track changes
            # But for simplicity and given small N, let's try to avoid full copy if possible
            # Actually, avail_count changes for all t_subsets in the block.
            
            # Let's just do the changes and revert
            newly_covered = []
            for t_idx in block_t_indices[b_idx]:
                if t_idx in uncovered:
                    uncovered.discard(t_idx)
                    newly_covered.append(t_idx)
                    avail_count[t_idx] -= 1
                else:
                    # Even if not uncovered, the block is used, so avail_count decreases
                    # Wait, avail_count is "number of available blocks".
                    # If a block is used, it's no longer available.
                    # So avail_count decreases for ALL t-subsets covered by the block,
                    # regardless of whether they are currently "uncovered" in our set.
                    # BUT, if a t-subset is already covered (not in uncovered), 
                    # its availability doesn't matter for the "uncovered" set's bottleneck,
                    # EXCEPT that it might become uncovered again? No, we only add blocks.
                    # So, avail_count for covered t-subsets is irrelevant for the search state 
                    # of *uncovered* items, UNLESS we backtrack and they become uncovered again.
                    # So we MUST update avail_count for all t-subsets in the block.
                    avail_count[t_idx] -= 1
            
            used_blocks[b_idx] = True
            
            search(depth + 1)
            
            # Backtrack
            used_blocks[b_idx] = False
            for t_idx in block_t_indices[b_idx]:
                avail_count[t_idx] += 1
                if t_idx in newly_covered:
                    uncovered.add(t_idx)
            # Restore uncovered set properly
            # The loop above added back newly_covered. 
            # But what about t_subsets that were NOT in uncovered before?
            # We didn't remove them from uncovered, so they are still not in it.
            # The `uncovered` set is restored correctly by adding back `newly_covered`.
            
    search(0)
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_solution]

import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # For each k-subset, store the list of t-subset indices it covers
    k_to_t = [[] for _ in range(num_k_subsets)]
    # For each t-subset, store the list of k-subset indices that cover it
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            k_to_t[i].append(idx)
            t_to_k[idx].append(i)

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(uncovered_list, used_set, randomize=True, max_blocks=None):
        """Greedy covering algorithm with efficient gain computation."""
        current_blocks = []
        local_used = set(used_set) if used_set else set()
        uncovered = set(uncovered_list) if isinstance(uncovered_list, set) else set(uncovered_list)
        
        if not uncovered:
            return current_blocks, local_used, uncovered
        
        # Track which k-subsets are available
        available = [i for i in range(num_k_subsets) if i not in local_used]
        
        while uncovered:
            if time.time() > deadline:
                break
            if max_blocks and len(current_blocks) >= max_blocks:
                break
            
            # Compute gain for each available k-subset
            max_gain = 0
            candidates = []
            
            # For efficiency, sample if there are too many candidates
            if len(available) > 3000:
                sample_size = min(3000, len(available))
                sampled = random.sample(available, sample_size)
                for i in sampled:
                    gain = 0
                    for ts_idx in k_to_t[i]:
                        if ts_idx in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        candidates = [i]
                    elif gain == max_gain and gain > 0:
                        candidates.append(i)
            else:
                for i in available:
                    gain = 0
                    for ts_idx in k_to_t[i]:
                        if ts_idx in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        candidates = [i]
                    elif gain == max_gain and gain > 0:
                        candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Pick a candidate
            if randomize and len(candidates) > 1:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0]
            
            current_blocks.append(k_subsets[chosen_idx])
            local_used.add(chosen_idx)
            
            # Remove covered t-subsets
            for ts_idx in k_to_t[chosen_idx]:
                uncovered.discard(ts_idx)
            
            # Update available list (remove chosen)
            if chosen_idx in available:
                available.remove(chosen_idx)
        
        return current_blocks, local_used, uncovered

    def local_search(blocks, max_iterations=2000):
        """Local search to try to reduce the number of blocks by removing redundant ones."""
        if not blocks or len(blocks) < 2:
            return blocks
        
        current_blocks = list(blocks)
        
        # Build coverage map: t_subset_idx -> set of block indices covering it
        coverage = defaultdict(set)
        for i, block in enumerate(current_blocks):
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                coverage[idx].add(i)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            
            # Try removing each block
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                
                # Check if removing block i leaves all t-subsets covered
                can_remove = True
                for ts in itertools.combinations(current_blocks[i], t):
                    idx = t_subset_index[ts]
                    if len(coverage[idx]) <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block i
                    for ts in itertools.combinations(current_blocks[i], t):
                        idx = t_subset_index[ts]
                        coverage[idx].discard(i)
                    
                    current_blocks.pop(i)
                    improved = True
                    break
        
        return current_blocks

    # Run multiple random restarts with greedy + local search
    num_restarts = 50  # We will limit by time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random greedy
        uncovered_all = set(range(num_t_subsets))
        used_set = set()
        
        current_blocks, local_used, remaining_uncovered = greedy_cover(
            uncovered_all, used_set, randomize=True, max_blocks=None
        )
        
        if remaining_uncovered:
            # Try to complete with a second greedy pass
            extra_blocks, _, _ = greedy_cover(remaining_uncovered, local_used, randomize=False)
            current_blocks.extend(extra_blocks)
            local_used.update(extra_blocks)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
        
        # Apply local search if we have time
        if time.time() < deadline - 20 and best_count > 300:
            improved_blocks = local_search(current_blocks, max_iterations=1000)
            if len(improved_blocks) < best_count:
                best_count = len(improved_blocks)
                best_blocks = [list(b) for b in improved_blocks]

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = set(range(num_t_subsets))
        current_blocks, _, _ = greedy_cover(uncovered, set(), randomize=False)
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks

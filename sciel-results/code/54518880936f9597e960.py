"""Covering design C(19,8,5): Optimized greedy + local search with redundancy pruning.
Uses multi-restart strategy with aggressive block removal and swap-based local search.
"""
import time
import random
import itertools
import numpy as np


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # Precompute t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_to_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks_total = len(all_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
    
    # Invert: for each t-subset, list of blocks that cover it
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
    
    def greedy_cover():
        """Fast greedy set cover: pick block covering most uncovered t-subsets."""
        uncovered = set(range(N))
        blocks = []
        
        while uncovered:
            best_b = -1
            max_new = 0
            
            # Try each block, pick the one covering most uncovered subsets
            for bi in range(n_blocks_total):
                new_cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        new_cov += 1
                if new_cov > max_new:
                    max_new = new_cov
                    best_b = bi
                
                # Early termination if we found a block covering all remaining
                if max_new == len(uncovered):
                    break
            
            if best_b == -1:
                break
            
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        
        return blocks
    
    def compute_counts(blocks):
        """Compute how many times each t-subset is covered."""
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt
    
    def is_valid(blocks):
        """Check if all t-subsets are covered."""
        cnt = compute_counts(blocks)
        return np.all(cnt >= 1)
    
    def remove_redundant(blocks, max_passes=50):
        """Repeatedly remove blocks that don't break validity."""
        for _ in range(max_passes):
            if time.time() > deadline - 5:
                break
            cnt = compute_counts(blocks)
            removed = False
            for bi in range(len(blocks) - 1, -1, -1):
                block_idx = blocks[bi]
                # Check if removing this block keeps all t-subsets covered
                can_remove = True
                for ti in block_cover[block_idx]:
                    if cnt[ti] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    del blocks[bi]
                    removed = True
            if not removed:
                break
        return blocks
    
    def local_search(blocks, max_steps=5000):
        """Local search using swap moves to enable further removals."""
        if not is_valid(blocks):
            return blocks
        
        cnt = compute_counts(blocks)
        best_blocks = list(blocks)
        best_size = len(blocks)
        rng = random.Random()
        
        T = 5.0
        cooling = 0.9995
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Try a swap: remove one block, add another
            if len(blocks) == 0:
                break
            
            old_idx = rng.randrange(len(blocks))
            old_block = blocks[old_idx]
            
            # Find t-subsets that would become uncovered if we remove old_block
            weak = []
            for ti in block_cover[old_block]:
                if cnt[ti] == 1:
                    weak.append(ti)
            
            if not weak:
                # Can remove directly
                del blocks[old_idx]
                for ti in block_cover[old_block]:
                    cnt[ti] -= 1
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = list(blocks)
                    no_improve = 0
                else:
                    no_improve += 1
                T *= cooling
                continue
            
            # Find a block that covers all weak t-subsets
            candidates = set(t_to_blocks[weak[0]])
            for ti in weak[1:]:
                candidates &= set(t_to_blocks[ti])
            candidates.discard(old_block)
            
            if not candidates:
                T *= cooling
                continue
            
            new_block = rng.choice(list(candidates))
            
            # Apply swap
            blocks[old_idx] = new_block
            for ti in block_cover[old_block]:
                cnt[ti] -= 1
            for ti in block_cover[new_block]:
                cnt[ti] += 1
            
            # Check if this enables a removal (size might stay same or we might remove later)
            # For now, just accept the swap
            T *= cooling
        
        return best_blocks
    
    # Multi-restart strategy
    best_blocks = None
    best_size = float('inf')
    
    rng = random.Random(123)
    n_restarts = 0
    
    while time.time() < deadline - 10:
        n_restarts += 1
        
        # Greedy initialization
        blocks = greedy_cover()
        
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        # Local search
        blocks = local_search(blocks, max_steps=3000)
        
        # Final redundancy removal
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
        
        # Occasionally do a more intensive local search
        if n_restarts % 3 == 0:
            blocks = local_search(blocks, max_steps=8000)
            blocks = remove_redundant(blocks)
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = list(blocks)
        
        # If we've reached a good solution, do intensive search
        if best_size <= 310:
            blocks = local_search(blocks, max_steps=15000)
            blocks = remove_redundant(blocks)
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = list(blocks)
    
    # Final verification
    if best_blocks is None:
        best_blocks = greedy_cover()
    
    # Verify validity
    if not is_valid(best_blocks):
        best_blocks = greedy_cover()
    
    return [list(all_blocks[bi]) for bi in best_blocks]

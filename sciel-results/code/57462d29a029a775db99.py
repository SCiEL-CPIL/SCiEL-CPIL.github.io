"""Improved covering design C(19,8,5) using randomized greedy with precomputed bitmasks."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each k-subset (block), which t-subsets it covers
    # For v=19, k=8, t=5: C(19,8)=75582 blocks, each covers C(8,5)=56 t-subsets
    # Total t-subsets: C(19,5)=11628
    # We'll use a bitset approach for speed
    
    # Precompute all k-subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    
    # For each k-subset, compute the set of t-subset indices it covers
    # This is expensive but done once
    block_covers = []  # list of frozensets of t-subset indices
    for block in all_ksubsets:
        covered = set()
        for tsub in itertools.combinations(block, t):
            covered.add(tsubset_to_idx[tsub])
        block_covers.append(frozenset(covered))
    
    # Index: for each t-subset index, which blocks cover it
    tsub_to_blocks = defaultdict(list)
    for i, block in enumerate(all_ksubsets):
        for tidx in block_covers[i]:
            tsub_to_blocks[tidx].append(i)
    
    # Greedy algorithm with randomization
    def greedy_randomized(seed, max_blocks=None):
        rng = random.Random(seed)
        uncovered = set(range(num_tsubsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # Weight towards t-subsets with fewer covering options (harder to cover)
            candidates = list(uncovered)
            # Use a soft-min: weight by 1/(number of available blocks + 1)
            weights = []
            for tidx in candidates:
                # Count how many blocks can still cover this (roughly: total blocks covering it)
                w = 1.0 / (len(tsub_to_blocks[tidx]) + 1)
                weights.append(w)
            
            # Sample a t-subset with probability proportional to weight
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_tidx = candidates[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_tidx = candidates[i]
                    break
            
            # Find the best block to add
            # A block is good if it covers many uncovered t-subsets
            best_block = None
            best_gain = -1
            best_block_idx = -1
            
            # Try blocks that cover the chosen t-subset
            for block_idx in tsub_to_blocks[chosen_tidx]:
                gain = len(block_covers[block_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = block_idx
            
            if best_block_idx < 0:
                # Fallback: pick any block
                best_block_idx = rng.randint(0, len(all_ksubsets) - 1)
                best_block = all_ksubsets[best_block_idx]
                best_gain = len(block_covers[best_block_idx] & uncovered)
            else:
                best_block = all_ksubsets[best_block_idx]
            
            blocks.append(best_block)
            uncovered.difference_update(block_covers[best_block_idx])
            
            if max_blocks and len(blocks) >= max_blocks:
                break
            
            # Check time
            if time.time() > deadline:
                break
        
        return blocks if not uncovered else None
    
    # First, run a deterministic greedy to get a baseline
    def greedy_deterministic():
        uncovered = set(range(num_tsubsets))
        blocks = []
        
        while uncovered:
            # Pick the uncovered t-subset with fewest covering blocks
            best_tidx = min(uncovered, key=lambda tidx: len(tsub_to_blocks[tidx]))
            
            # Find the block that covers the most uncovered t-subsets
            best_block_idx = -1
            best_gain = -1
            for block_idx in tsub_to_blocks[best_tidx]:
                gain = len(block_covers[block_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = block_idx
            
            blocks.append(all_ksubsets[best_block_idx])
            uncovered.difference_update(block_covers[best_block_idx])
            
            if time.time() > deadline:
                break
        
        return blocks if not uncovered else None
    
    # Run deterministic greedy first
    best_blocks = greedy_deterministic()
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = []
        uncovered = set(range(num_tsubsets))
        while uncovered and time.time() < deadline:
            tidx = min(uncovered)
            for block_idx in tsub_to_blocks[tidx]:
                gain = len(block_covers[block_idx] & uncovered)
                if gain > 0:
                    best_blocks.append(all_ksubsets[block_idx])
                    uncovered.difference_update(block_covers[block_idx])
                    break
    
    current_best = len(best_blocks)
    
    # Now try randomized restarts to potentially improve
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Try to improve: use a modified greedy that sometimes makes suboptimal choices
        # to explore different parts of the solution space
        blocks = greedy_randomized(seed)
        if blocks is not None and len(blocks) < current_best:
            current_best = len(blocks)
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END

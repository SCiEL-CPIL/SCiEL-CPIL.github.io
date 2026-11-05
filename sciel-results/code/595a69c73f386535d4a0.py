import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute:
    # 1. For each block, which triples it covers (as indices)
    # 2. For each triple, which blocks cover it (as indices)
    block_triples = [set() for _ in range(num_blocks)]
    triple_blocks = [set() for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            block_triples[i].add(idx)
            triple_blocks[idx].add(i)
    
    # Convert to lists for faster iteration if needed, but sets are good for membership
    # For B&B, we need to iterate over candidates for a triple
    
    # 1. Get a good initial solution using Greedy
    def greedy_construct():
        uncovered = [1] * num_triples
        selected = []
        selected_set = set()
        
        while any(uncovered):
            best_block = -1
            best_gain = -1
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = 0
                for tri_idx in block_triples[i]:
                    if uncovered[tri_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                    if gain == num_triples: # Perfect block? Unlikely
                        break
            if best_gain <= 0:
                break
            selected.append(best_block)
            selected_set.add(best_block)
            for tri_idx in block_triples[best_block]:
                uncovered[tri_idx] = 0
        return selected

    initial_sol = greedy_construct()
    best_size = len(initial_sol)
    best_sol = list(initial_sol)
    
    # 2. Branch and Bound
    # State: 
    #   uncovered_mask: bitset or list of uncovered triple indices? 
    #   Using a list of uncovered indices is slow for membership. 
    #   Using a set of uncovered triple indices.
    #   current_count: int
    #   selected_blocks: list (only for reconstruction, not needed for search logic if we track counts)
    
    # To make B&B fast, we need a strong lower bound.
    # LB = ceil(len(uncovered) / max_cover)
    # max_cover is the max number of uncovered triples a single block can cover.
    # We can precompute the max possible cover for any block (it's C(k,t) = 35).
    # But as uncovered set shrinks, max cover decreases.
    # A tighter LB: For each uncovered triple, it must be covered by one of the blocks covering it.
    # This is a hitting set problem on the dual.
    
    # Let's implement a simple B&B.
    # We will use a global variable for best_size and best_sol to allow early termination in recursion.
    
    # Optimization: Use numpy bitsets for uncovered state? 
    # num_triples = 2024. A 2024-bit integer fits in Python's arbitrary precision int.
    # This will be MUCH faster for state manipulation.
    
    triple_bit = [1 << i for i in range(num_triples)]
    
    # Precompute block masks
    block_masks = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for tri_idx in block_triples[i]:
            mask |= triple_bit[tri_idx]
        block_masks[i] = mask
        
    full_mask = (1 << num_triples) - 1
    
    # For B&B, we need to find the "most constrained" uncovered triple.
    # This requires counting how many available blocks cover each uncovered triple.
    # "Available" means not yet selected.
    # In B&B, we don't explicitly track selected blocks in the state for pruning, 
    # but we do need to know which blocks are "used" to avoid re-selecting them?
    # Actually, in set cover, order doesn't matter, but we shouldn't pick the same block twice.
    # If we pick a block, we remove its triples. We don't need to "ban" the block 
    # unless we are doing exact cover. For covering, picking the same block twice is useless.
    # So we should maintain a set of selected block indices.
    
    # However, passing a set of selected blocks in recursion is expensive.
    # Alternative: The search tree is defined by choices of blocks.
    # We can just track `uncovered_mask`.
    # When we pick a block `b`, we do `uncovered_mask &= ~block_masks[b]`.
    # We don't strictly need to know which blocks were picked for the *logic* of coverage, 
    # only for the *count* and *reconstruction*.
    # But we must ensure we don't pick the same block twice. 
    # If we pick block `b`, it covers some triples. If we pick it again, it covers 0 new triples.
    # So if we always pick a block that covers at least 1 uncovered triple, we never pick the same block twice 
    # (because after picking it, it covers 0 uncovered triples).
    # Wait, if we pick block A, then block B, then later we want to pick A again? 
    # A's triples are already covered by A (and maybe B). So A covers 0 *uncovered* triples.
    # So the greedy/B&B step "pick a block covering an uncovered triple" naturally avoids duplicates 
    # IF we only consider blocks that have non-zero intersection with `uncovered_mask`.
    
    def branch_and_bound(uncovered_mask, current_count, selected_blocks, depth):
        nonlocal best_size, best_sol
        
        if time.time() > deadline - 5:
            return
            
        if uncovered_mask == 0:
            if current_count < best_size:
                best_size = current_count
                best_sol = list(selected_blocks)
            return
            
        # Lower Bound
        # Count uncovered triples
        # Using bincount on bits is slow. 
        # We can maintain a count in the state? 
        # Or just estimate: ceil(popcount(uncovered_mask) / 35)
        # popcount is fast in Python 3.10+? int.bit_count()
        num_unc = uncovered_mask.bit_count()
        
        # Max cover per block is at most C(7,3)=35.
        lb = (num_unc + 34) // 35
        
        if current_count + lb >= best_size:
            return
            
        # Find the uncovered triple with the fewest candidate blocks
        # This is the "most constrained" variable.
        # We need to iterate through uncovered triples and count candidates.
        # Candidates for triple t: blocks b such that (block_masks[b] & triple_bit[t]) != 0 
        # AND b is not in selected_blocks? 
        # As established, if b is in selected_blocks, it covers 0 uncovered triples, 
        # so it won't be a candidate for covering an *uncovered* triple.
        # So we just need to find blocks that cover this triple.
        
        # To find the most constrained triple efficiently:
        # We can't easily iterate "uncovered triples" from a bitmask without extracting bits.
        # Extracting bits:
        # temp = uncovered_mask
        # while temp:
        #   lsb = temp & -temp
        #   idx = lsb.bit_length() - 1
        #   ...
        
        # This might be slow if done at every node.
        # Heuristic: Just pick the first uncovered triple? Or the one with min degree?
        # Let's try picking the triple with min degree (fewest blocks covering it).
        
        min_candidates = float('inf')
        best_triple_idx = -1
        best_candidate_blocks = []
        
        temp_mask = uncovered_mask
        # Limit iterations for speed? No, we need to find the best one.
        # But 2024 triples is manageable.
        
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            # Count blocks covering idx
            # triple_blocks[idx] is a set of block indices
            # We need to check if they cover *uncovered* parts? 
            # No, if a block covers idx, and idx is uncovered, the block is a candidate.
            # The number of candidates is just len(triple_blocks[idx]).
            # Wait, some of those blocks might have been selected? 
            # If a block was selected, it covers idx. If idx is still uncovered, 
            # that means the block was NOT selected (contradiction) or the block didn't cover idx?
            # If block b was selected, it covers idx. So idx would be covered.
            # Therefore, if idx is uncovered, NO selected block covers idx.
            # So ALL blocks in triple_blocks[idx] are "available" in the sense that 
            # they haven't been used to cover this specific triple yet.
            # But could a block in triple_blocks[idx] have been selected for *another* triple?
            # Yes. If block b was selected, it covers idx. So idx would be covered.
            # So if idx is uncovered, b was NOT selected.
            # Thus, the set of candidate blocks for an uncovered triple is exactly triple_blocks[idx].
            
            count = len(triple_blocks[idx])
            if count < min_candidates:
                min_candidates = count
                best_triple_idx = idx
                best_candidate_blocks = list(triple_blocks[idx])
                if count == 1:
                    break # Can't get better than 1
        
        if best_triple_idx == -1:
            return # Should not happen if uncovered_mask != 0
            
        # Order candidates by how many *other* uncovered triples they cover (greedy heuristic)
        # This helps find good solutions faster.
        def candidate_score(b_idx):
            # Count intersection of block b_idx with uncovered_mask
            # popcount(block_masks[b_idx] & uncovered_mask)
            return (block_masks[b_idx] & uncovered_mask).bit_count()
            
        best_candidate_blocks.sort(key=candidate_score, reverse=True)
        
        # Branch
        for b_idx in best_candidate_blocks:
            new_mask = uncovered_mask & ~block_masks[b_idx]
            # If no progress, skip (should not happen as b_idx covers best_triple_idx)
            if new_mask == uncovered_mask:
                continue
                
            selected_blocks.append(b_idx)
            branch_and_bound(new_mask, current_count + 1, selected_blocks, depth + 1)
            selected_blocks.pop()
            
            if time.time() > deadline - 5:
                return

    # Run B&B
    # Initial state: full_mask, 0, [], 0
    # We use the greedy solution as the initial upper bound (best_size)
    
    # To make B&B more effective, we can run it multiple times with different random orders?
    # The ordering of candidates is deterministic (sorted by score).
    # We can randomize the tie-breaking or the initial triple selection.
    
    # Let's run B&B once. If it finishes, great. If not, it returns the best found.
    # We should also try to improve the initial greedy solution with a quick local search before B&B?
    # The prompt says "different strategy". B&B is different.
    
    # One issue: B&B might be too slow if the lower bound is weak.
    # Let's add a time check inside the loop.
    
    try:
        branch_and_bound(full_mask, 0, [], 0)
    except RecursionError:
        # If recursion depth is an issue (unlikely for 76 blocks, max depth 76)
        pass

    # If B&B didn't find anything better, fallback to greedy
    if best_sol is None or len(best_sol) > len(initial_sol):
        best_sol = initial_sol

    return [all_blocks[i] for i in best_sol]

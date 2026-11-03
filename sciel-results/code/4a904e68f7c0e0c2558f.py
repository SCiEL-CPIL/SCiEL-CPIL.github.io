"""Improved covering design solver using beam search over precomputed blocks with local improvement."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute which t-subsets it covers
    # block_covers[b_idx] = frozenset of t-subset indices covered by block b_idx
    block_covers = []
    for b_idx, block in enumerate(all_kblocks):
        covered = set()
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            covered.add(idx)
        block_covers.append(frozenset(covered))
    
    # For each t-subset, precompute which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubsets)]
    for b_idx, covers in enumerate(block_covers):
        for ts_idx in covers:
            tsub_to_blocks[ts_idx].append(b_idx)
    
    # Greedy with beam search using numpy for faster gain computation
    def greedy_cover_beam(seed, beam_width=5):
        rng = random.Random(seed)
        uncovered = set(range(num_tsubsets))
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Pick beam_width random uncovered t-subsets as targets
            if len(uncovered) <= beam_width:
                targets = list(uncovered)
            else:
                targets = rng.sample(list(uncovered), min(beam_width, len(uncovered)))
            
            # For each target, find the best block covering it
            candidate_blocks = set()
            for target_idx in targets:
                best_gain = 0
                best_block = -1
                for b_idx in tsub_to_blocks[target_idx]:
                    gain = len(block_covers[b_idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                if best_block != -1:
                    candidate_blocks.add(best_block)
            
            # Pick the best block among candidates
            final_best = -1
            final_gain = 0
            for b_idx in candidate_blocks:
                gain = len(block_covers[b_idx] & uncovered)
                if gain > final_gain:
                    final_gain = gain
                    final_best = b_idx
            
            if final_best == -1:
                break
            
            selected_blocks.append(all_kblocks[final_best])
            uncovered -= block_covers[final_best]
        
        return selected_blocks
    
    # Redundancy removal: remove blocks whose t-subsets are all covered by other blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Convert to frozensets for fast lookup
        block_sets = [frozenset(b) for b in blocks]
        
        # For each t-subset, track which blocks cover it
        tsub_cover_count = defaultdict(int)
        for i, bs in enumerate(block_sets):
            for tsub in itertools.combinations(bs, t):
                idx = tsub_to_idx[tsub]
                tsub_cover_count[idx] += 1
        
        # Remove redundant blocks
        new_blocks = []
        for i, bs in enumerate(block_sets):
            redundant = True
            for tsub in itertools.combinations(bs, t):
                idx = tsub_to_idx[tsub]
                if tsub_cover_count[idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(list(bs))
        
        return new_blocks
    
    # Local improvement: try to remove blocks or replace pairs
    def local_improve(blocks, deadline):
        if not blocks:
            return blocks
        
        blocks = remove_redundant(blocks)
        
        # Try to find blocks that can be removed without losing coverage
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove block i
                candidate_blocks = blocks[:i] + blocks[i+1:]
                
                # Check if all t-subsets are still covered
                covered = set()
                for b in candidate_blocks:
                    bs = frozenset(b)
                    for tsub in itertools.combinations(bs, t):
                        covered.add(tsub)
                
                if len(covered) == num_tsubsets:
                    blocks = candidate_blocks
                    improved = True
                    break
        
        return blocks
    
    # Pairwise block replacement: try to replace two blocks with two others that cover the same t-subsets
    def pairwise_replace(blocks, deadline):
        if len(blocks) < 2:
            return blocks
        
        n = len(blocks)
        blocks = list(blocks)
        improved = True
        
        while improved and time.time() < deadline - 10:
            improved = False
            n = len(blocks)
            
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                
                for j in range(i + 1, n):
                    if time.time() > deadline - 10:
                        break
                    
                    # Try removing blocks i and j, then see if we can add back
                    # two other blocks that cover everything they covered
                    removed_i = blocks[i]
                    removed_j = blocks[j]
                    remaining = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                    
                    # What t-subsets are only covered by i or j?
                    # Actually, let's try a simpler approach: 
                    # Check if we can replace one of them with a different block
                    # that covers all t-subsets the original covered
                    # while also covering some uncovered t-subsets
                    pass
        
        return blocks
    
    # Try multiple strategies with different beam widths and seeds
    best_blocks = None
    best_count = float('inf')
    
    strategies = [
        (1, 42),
        (3, 101),
        (5, 102),
        (10, 103),
        (20, 201),
        (5, 202),
        (10, 301),
        (20, 302),
        (50, 401),
        (3, 501),
        (5, 502),
        (10, 601),
    ]
    
    for beam_width, seed in strategies:
        if time.time() > deadline - 15:
            break
        
        blocks = greedy_cover_beam(seed, beam_width)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
    
    # Try local improvement on the best solution
    if best_blocks is not None and time.time() < deadline - 20:
        improved = local_improve(best_blocks, deadline)
        if improved and len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)
    
    # Try a few more seeds if time remains
    seed = 1000
    while time.time() < deadline - 10:
        blocks = greedy_cover_beam(seed, 5)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        return []
    
    return best_blocks

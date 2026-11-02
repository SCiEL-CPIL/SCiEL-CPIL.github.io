import itertools
import time
import random
import numpy as np

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7

def _precompute():
    global triple_to_idx, idx_to_triple, all_triples
    all_triples = list(itertools.combinations(range(V), T))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    idx_to_triple = all_triples

_precompute()

def _greedy(v, k, t, deadline, rng):
    start_time = time.time()
    uncovered = [True] * len(idx_to_triple)
    blocks = []
    
    # List of uncovered indices for fast selection
    # We can just scan or keep a set. Scanning 2024 items is fast.
    while True:
        if time.time() > deadline:
            break
        # Find an uncovered triple
        uncovered_idx = None
        for i in range(len(uncovered)):
            if uncovered[i]:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        seed_triple = idx_to_triple[uncovered_idx]
        block = list(seed_triple)
        
        # Greedily add points to block
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_rand = 0
            
            # Candidates are all points not in block
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # Triples formed by x and any 2 points in block
                # Note: block has size m. Pairs are C(m, 2)
                for pair in itertools.combinations(block, t - 1):
                    # Triple is (x, pair[0], pair[1])
                    tri = tuple(sorted((x, pair[0], pair[1])))
                    idx = triple_to_idx[tri]
                    if uncovered[idx]:
                        gain += 1
                
                # Tie breaking with randomness
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_rand = rng.random()
                elif gain == best_gain:
                    r = rng.random()
                    if r > best_rand:
                        best_x = x
                        best_rand = r
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Mark covered
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            uncovered[idx] = False
            
    return blocks

def _remove_redundant(blocks, v, t, deadline):
    if not blocks:
        return blocks
    
    # Count coverage for each triple
    # cover_count[i] = number of blocks covering triple i
    cover_count = [0] * len(idx_to_triple)
    block_triples = []
    
    for b in blocks:
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            cover_count[idx] += 1
            tris.append(idx)
        block_triples.append(tris)
    
    changed = True
    active = [True] * len(blocks)
    
    while changed:
        changed = False
        if time.time() > deadline:
            break
        
        for i in range(len(blocks)):
            if not active[i]:
                continue
            
            # Check if block i is redundant
            # It is redundant if for all triples it covers, cover_count[tri] > 1
            # (assuming other active blocks still cover them)
            # Wait, if we remove block i, cover_count decreases.
            # We can only remove block i if it is the *only* block covering some triple? No.
            # We can remove block i if it does *not* uniquely cover any triple.
            # i.e., for all tri in block_triples[i], cover_count[tri] > 1.
            
            is_redundant = True
            for tri_idx in block_triples[i]:
                if cover_count[tri_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                active[i] = False
                changed = True
                # Update cover counts
                for tri_idx in block_triples[i]:
                    cover_count[tri_idx] -= 1
                    
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = random.Random(42)
    
    # Phase 1: Greedy Construction
    blocks = _greedy(v, k, t, deadline, rng)
    
    # Phase 2: Remove Redundant Blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    
    # Phase 3: Local Search (Optional, if time permits)
    # Try to swap points or remove/add blocks to reduce count?
    # Given the complexity, sticking to Greedy + Redundancy Removal is safer for 150s limit.
    # But we can try a few restarts with different seeds if time allows.
    
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try a few more greedy runs with different random seeds if time allows
    seed = 43
    while time.time() < deadline - 10.0:
        rng2 = random.Random(seed)
        candidate_blocks = _greedy(v, k, t, deadline, rng2)
        candidate_blocks = _remove_redundant(candidate_blocks, v, t, deadline)
        
        if len(candidate_blocks) < best_count:
            best_blocks = candidate_blocks
            best_count = len(candidate_blocks)
        
        seed += 1
        # Limit restarts to avoid infinite loop if greedy is slow
        if seed > 100: 
            break

    return best_blocks

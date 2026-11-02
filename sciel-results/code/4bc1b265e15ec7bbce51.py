import itertools
import random
import time
import sys

_PRECOMP_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[key]
        
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = []
    block_triples = []
    blocks_covering_triple = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block)
        
        block_triple_indices = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            block_triple_indices.append(idx)
            blocks_covering_triple[idx].append(block_id)
            
        block_triples.append(block_triple_indices)
        
    _PRECOMP_CACHE[key] = (all_blocks, block_triples, blocks_covering_triple, num_triples)
    return _PRECOMP_CACHE[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_blocks, block_triples, blocks_covering_triple, num_triples = _get_precomputed(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    while time.time() < deadline:
        rng = random.Random()
        
        uncovered = set(range(num_triples))
        selected_blocks = []
        used_blocks = set()
        
        while uncovered:
            chosen_triple = rng.choice(list(uncovered))
            
            candidates = blocks_covering_triple[chosen_triple]
            
            best_block_idx = -1
            best_score = -1
            best_tie_breakers = []
            
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                
                score = 0
                for t_idx in block_triples[b_idx]:
                    if t_idx in uncovered:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_block_idx = b_idx
                    best_tie_breakers = [b_idx]
                elif score == best_score:
                    if best_score > 0:
                        best_tie_breakers.append(b_idx)
            
            if best_block_idx == -1:
                break
                
            if len(best_tie_breakers) > 1:
                best_block_idx = rng.choice(best_tie_breakers)
            
            selected_blocks.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            for t_idx in block_triples[best_block_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    
        # Polish: Remove redundant blocks
        polish_improved = True
        while polish_improved:
            polish_improved = False
            
            coverage = [0] * num_triples
            for b_idx in selected_blocks:
                for t_idx in block_triples[b_idx]:
                    coverage[t_idx] += 1
            
            to_remove = []
            for i, b_idx in enumerate(selected_blocks):
                essential = False
                for t_idx in block_triples[b_idx]:
                    if coverage[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(i)
            
            if to_remove:
                for i in reversed(to_remove):
                    selected_blocks.pop(i)
                polish_improved = True
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            
            if best_size <= 69:
                break

    if best_solution is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_solution]

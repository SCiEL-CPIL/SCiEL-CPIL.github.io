"""Optimized covering C(24,7,3) using Isolated-Point Greedy + Efficient Local Search."""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    """
    blocks = _greedy_cover(v, k, t)
    blocks = _optimize(v, k, t, blocks)
    return blocks

def _greedy_cover(v, k, t):
    """
    Greedy construction prioritizing triples with rare points.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    while uncovered:
        # Calculate point frequencies in uncovered set
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
                
        # Find the uncovered triple with the highest "rarity score"
        # Rarity score = sum of 1 / (freq(p) + 1)
        max_score = -1
        best_triple = None
        for triple in uncovered:
            score = 0.0
            for p in triple:
                score += 1.0 / (point_freq[p] + 1)
            if score > max_score:
                max_score = score
                best_triple = triple
                
        # Extend best_triple to a full block
        block = list(best_triple)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # Calculate new triples formed by adding x
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None:
                break
            block.append(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _optimize(v, k, t, blocks):
    """
    Local search: Compress and Swap.
    """
    blocks = [list(b) for b in blocks]
    
    def get_triples(block):
        return [tuple(triple) for triple in itertools.combinations(block, t)]
        
    # Initialize coverage counts
    triple_counts = defaultdict(int)
    for b in blocks:
        for triple in get_triples(b):
            triple_counts[triple] += 1
            
    def add_block(block):
        for triple in get_triples(block):
            triple_counts[triple] += 1
            
    def remove_block(block):
        for triple in get_triples(block):
            triple_counts[triple] -= 1
            if triple_counts[triple] == 0:
                del triple_counts[triple]
                
    improved = True
    iterations = 0
    while improved and iterations < 50:
        improved = False
        iterations += 1
        
        # Pass 1: Compress (Remove redundant blocks)
        i = 0
        while i < len(blocks):
            b = blocks[i]
            is_redundant = True
            for triple in get_triples(b):
                if triple_counts[triple] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                remove_block(b)
                blocks.pop(i)
                improved = True
            else:
                i += 1
                
        # Pass 2: Swap
        random.seed(42)
        for idx in range(len(blocks)):
            old_block = blocks[idx]
            old_triples = get_triples(old_block)
            
            # Find unique triples (count == 1)
            unique_triples = [tr for tr in old_triples if triple_counts[tr] == 1]
            
            if not unique_triples:
                # Should have been removed in Pass 1
                remove_block(old_block)
                blocks.pop(idx)
                improved = True
                continue
            
            # We must cover all unique_triples with the new block.
            P_req = set()
            for tr in unique_triples:
                P_req.update(tr)
                
            if len(P_req) > k:
                continue
                
            # Try to find a new block B' containing P_req that maximizes "redundancy potential"
            # Redundancy potential = number of triples in B' that have count 1 (and are not in old_block)
            # Actually, if we cover a triple with count 1 from another block, that other block might become redundant.
            
            best_new_block = None
            best_score = -1
            
            fill_count = k - len(P_req)
            available = [p for p in range(v) if p not in P_req]
            
            # Sample candidates
            for _ in range(100):
                if fill_count == 0:
                    new_block = tuple(sorted(P_req))
                else:
                    fill = random.sample(available, fill_count)
                    new_block = tuple(sorted(list(P_req) + fill))
                    
                # Verify it covers all unique_triples
                valid = True
                for tr in unique_triples:
                    if not set(tr).issubset(set(new_block)):
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Score: count of triples in new_block with count 1
                score = 0
                for tr in itertools.combinations(new_block, t):
                    if triple_counts[tr] == 1:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_new_block = new_block
                    
            if best_new_block:
                remove_block(old_block)
                add_block(best_new_block)
                blocks[idx] = list(best_new_block)
                improved = True
                
    return blocks

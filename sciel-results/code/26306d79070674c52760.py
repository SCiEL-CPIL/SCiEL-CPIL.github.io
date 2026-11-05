"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def construct(v, k, t):
    """
    Greedy covering with multiple heuristics and randomization.
    
    Strategy:
    1. Use a fast greedy that picks blocks maximizing coverage of uncovered t-subsets.
    2. Try multiple random restarts with different orderings to find better solutions.
    3. Use a deadline guard to return the best result before timeout.
    
    For C(24,7,3): v=24, k=7, t=3
    Total 3-subsets: C(24,3) = 2024
    Each 7-block covers C(7,3) = 35 triples.
    Lower bound: ceil(2024/35) = 58 (but actual LB is 69).
    Best known: 76.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    total_triples = list(itertools.combinations(range(v), t))
    n_triples = len(total_triples)
    
    # Map each triple to an index
    triple_to_idx = {triple: i for i, triple in enumerate(total_triples)}
    
    def greedy_cover(random_seed=None, max_blocks=None):
        """
        Greedy covering with randomized point selection.
        Returns list of blocks (each a sorted list of k points).
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        # uncovered: set of triple indices still uncovered
        uncovered = set(range(n_triples))
        blocks = []
        
        if max_blocks is None:
            max_blocks = 1000  # safety limit
        
        while uncovered and len(blocks) < max_blocks:
            if time.time() > deadline:
                break
            
            # Pick the first (or a random) uncovered triple to seed the block
            if len(uncovered) > 0:
                # Use the lexicographically smallest uncovered triple for determinism
                # Or randomize for diversity
                seed_triple = min(uncovered)
                seed_points = list(total_triples[seed_triple])
            else:
                break
            
            block = set(seed_points)
            
            # Greedily add points that cover the most uncovered triples
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_gain = -1
                best_points = []
                
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered triples would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(triple, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    break
                
                # Randomly choose among best points for diversity
                chosen = rng.choice(best_points)
                block.add(chosen)
            
            if len(block) == k:
                block_sorted = sorted(block)
                blocks.append(block_sorted)
                
                # Update uncovered
                for triple in itertools.combinations(block_sorted, t):
                    idx = triple_to_idx.get(triple, -1)
                    if idx >= 0:
                        uncovered.discard(idx)
        
        return blocks, len(uncovered) > 0
    
    # Try multiple strategies
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Deterministic greedy (seed=0)
    blocks, incomplete = greedy_cover(random_seed=0)
    if not incomplete:
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Strategy 2: Multiple random restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        blocks, incomplete = greedy_cover(random_seed=i + 100)
        if not incomplete:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    # If no complete covering found, use the one with fewest blocks
    if best_blocks is None:
        # Find the run with fewest blocks even if incomplete
        best_blocks = []
        best_count = float('inf')
        for i in range(min(n_restarts, 10)):
            if time.time() > deadline:
                break
            blocks, incomplete = greedy_cover(random_seed=i + 1000)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    # Final check: ensure all triples are covered
    # If not, add more blocks greedily
    if best_blocks is None:
        best_blocks = []
    
    # Verify and complete if needed
    if time.time() < deadline - 5:
        covered = set()
        for block in best_blocks:
            for triple in itertools.combinations(block, t):
                covered.add(triple)
        
        if len(covered) < n_triples:
            # Add more blocks to cover remaining triples
            uncovered_triples = [tr for tr in total_triples if tr not in covered]
            for triple in uncovered_triples:
                if time.time() > deadline:
                    break
                # Create a block containing this triple
                block = set(triple)
                while len(block) < k:
                    best_gain = -1
                    best_x = None
                    for x in range(v):
                        if x in block:
                            continue
                        gain = sum(1 for rest in itertools.combinations(block, t-1)
                                   if tuple(sorted(rest + (x,))) not in covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is not None:
                        block.add(best_x)
                    else:
                        break
                if len(block) == k:
                    block_sorted = sorted(block)
                    # Check if this block is new
                    if block_sorted not in best_blocks:
                        best_blocks.append(block_sorted)
                        for tr in itertools.combinations(block_sorted, t):
                            covered.add(tr)
    
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks


# EVOLVE-BLOCK-END

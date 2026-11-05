import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage bitsets for each block
    # Each block covers C(k, t) triples.
    # We represent the set of triples covered by a block as a Python integer bitset.
    # This allows fast intersection and union operations.
    block_bitsets = [0] * num_blocks
    block_triple_indices = [[] for _ in range(num_blocks)]
    
    for i, b in enumerate(all_blocks):
        bs = 0
        indices = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bs |= (1 << idx)
            indices.append(idx)
        block_bitsets[i] = bs
        block_triple_indices[i] = indices

    def greedy_construct(random_seed, max_blocks_limit=150):
        """
        Constructs a covering using a weighted greedy algorithm.
        Randomness is introduced by shuffling the order of blocks with equal gain,
        or by slightly perturbing gains.
        """
        rng = random.Random(random_seed)
        uncovered_bitset = (1 << num_triples) - 1
        selected_indices = []
        
        # To speed up, we can maintain a list of candidate blocks.
        # However, scanning all 346k blocks every step is ~346k * 100 steps = 34M ops.
        # In Python, this might be slow if done naively.
        # Let's optimize the gain calculation.
        
        # Optimization: 
        # Instead of scanning all blocks, we can note that only blocks that cover 
        # at least one uncovered triple are useful.
        # But maintaining this list dynamically is complex.
        
        # Alternative: 
        # Use a heuristic that prefers blocks with higher "efficiency".
        # Let's stick to the scan but optimize the inner loop.
        
        # Actually, a better strategy for speed:
        # 1. Shuffle all blocks initially.
        # 2. Iterate through shuffled blocks.
        # 3. If a block covers any uncovered triple, add it? 
        #    This is "random greedy" which is often worse than "best greedy".
        
        # Let's try a hybrid: 
        # First few steps: Best Greedy (scan all).
        # Later steps: Sampled Greedy (scan subset) to save time?
        # Or just rely on the fact that we have ~100s.
        # 346k blocks * 35 triples = 12M checks per step.
        # 70 steps = 840M checks. Too slow in pure Python.
        
        # We need a faster gain calculation.
        # (block_bitsets[i] & uncovered_bitset).bit_count() is fast in C.
        # Let's benchmark mentally: 346k iterations of bit_count on 2024-bit ints.
        # Should be doable in < 0.1s per step? Maybe 0.05s.
        # 70 steps * 0.05s = 3.5s. Acceptable.
        
        while uncovered_bitset and len(selected_indices) < max_blocks_limit:
            if time.time() > deadline - 20:
                break
            
            best_gain = -1
            best_idx = -1
            # We can randomize the tie-breaking by iterating in random order?
            # That would require shuffling indices every time, which is O(N log N).
            # Instead, we can just pick the first max gain found, and if we want diversity,
            # we can add noise to the gain?
            
            # For now, deterministic best greedy, but we run multiple seeds.
            # To introduce diversity, we can randomly skip some blocks?
            # No, let's just run multiple identical greedy constructions? 
            # That yields the same result.
            
            # To get diversity, we need to perturb the objective.
            # Let's add a small random noise to the gain?
            # Or, we can pre-shuffle the block list and iterate in that order, 
            # picking the first block that has max gain? No, that's not max gain.
            
            # Strategy: 
            # 1. Calculate gain for all blocks.
            # 2. Find max gain.
            # 3. Collect all blocks with max gain (or close to max).
            # 4. Pick one randomly from that set.
            
            # This collection step might be expensive if many ties.
            # But ties are rare in early stages.
            
            max_gain_val = -1
            candidates = []
            
            # Optimization: Early exit if gain is 0?
            # If max possible gain is small, we are stuck.
            
            for i in range(num_blocks):
                gain = (block_bitsets[i] & uncovered_bitset).bit_count()
                if gain > max_gain_val:
                    max_gain_val = gain
                    candidates = [i]
                elif gain == max_gain_val:
                    candidates.append(i)
                
                if gain == 0:
                    # If we found 0 gain, and we are looking for max, 
                    # we can break early ONLY IF we know no block can have gain > 0?
                    # No, we are scanning.
                    pass
            
            if max_gain_val <= 0:
                break
                
            # Randomly select from candidates
            selected_idx = rng.choice(candidates)
            selected_indices.append(selected_idx)
            uncovered_bitset &= ~block_bitsets[selected_idx]
            
        return selected_indices

    def prune_solution(block_indices):
        """
        Removes redundant blocks from the solution.
        A block is redundant if every triple it covers is also covered by another block in the set.
        """
        if not block_indices:
            return []
        
        # Convert to set for O(1) lookup? 
        # We need to check coverage counts.
        # covered_counts[tri] = number of blocks in block_indices covering tri.
        
        covered_counts = [0] * num_triples
        for b_idx in block_indices:
            for tri_idx in block_triple_indices[b_idx]:
                covered_counts[tri_idx] += 1
        
        # Sort blocks by "essentiality"? 
        # Blocks covering more unique triples are harder to remove.
        # Let's just iterate and try to remove.
        
        current_set = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_set)):
                b_idx = current_set[i]
                # Check if b_idx is redundant
                redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if covered_counts[tri_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b_idx
                    current_set.pop(i)
                    for tri_idx in block_triple_indices[b_idx]:
                        covered_counts[tri_idx] -= 1
                    changed = True
                    if time.time() > deadline - 10:
                        return current_set
                    # Restart loop because indices changed
                    break
        
        return current_set

    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy constructions with different random seeds
    # The randomness comes from tie-breaking.
    # To increase diversity, we can also try different initial "biases".
    # But let's stick to tie-breaking randomness first.
    
    num_starts = 20
    for seed in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        raw_solution = greedy_construct(seed)
        
        if not raw_solution:
            continue
            
        # Prune
        pruned_solution = prune_solution(raw_solution)
        
        # Verify validity (should be valid by construction, but check just in case)
        # And check size
        if len(pruned_solution) < best_size:
            # Verify coverage
            covered = [0] * num_triples
            for b_idx in pruned_solution:
                for tri_idx in block_triple_indices[b_idx]:
                    covered[tri_idx] += 1
            
            if all(c >= 1 for c in covered):
                best_solution = pruned_solution
                best_size = len(pruned_solution)
                
                # If we hit the lower bound (69), we can stop
                if best_size <= 69:
                    break

    if best_solution is None:
        # Fallback: Return a large valid covering
        # Use the first greedy result without pruning? Or just generate one.
        # If greedy failed, something is wrong. 
        # Let's just return a trivial covering (all blocks? No, too many).
        # Return the first 100 blocks? Might not cover.
        # Let's do a simple greedy without time limit check for fallback.
        uncovered_bitset = (1 << num_triples) - 1
        fallback = []
        while uncovered_bitset:
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                gain = (block_bitsets[i] & uncovered_bitset).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain <= 0:
                break
            fallback.append(best_idx)
            uncovered_bitset &= ~block_bitsets[best_idx]
        
        if not fallback:
             # Absolute fallback: return all blocks (valid but huge score)
             return [all_blocks[i] for i in range(num_blocks)]
             
        best_solution = fallback

    # Convert indices to blocks
    result_blocks = [all_blocks[i] for i in best_solution]
    
    return result_blocks

# To ensure the function is defined even if imports fail or something
# (Standard library and numpy/scipy are available)

if __name__ == "__main__":
    # Test with small parameters
    # v=7, k=3, t=2 -> C(7,3,2) = 7 blocks (Fano plane)
    # blocks = construct(7, 3, 2)
    # print(len(blocks))
    pass

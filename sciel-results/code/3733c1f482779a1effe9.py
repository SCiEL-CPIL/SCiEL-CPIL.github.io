"""Covering design C(24,7,3) using VNS on block permutations."""
import itertools
import random
import time
import sys

# Precompute all 3-subsets for v=24
# But we need to be general for the interface, though the task is specific.
# Let's write a general solution that works well for v=24, k=7, t=3.

def construct(v, k, t):
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    # For v=24, k=7, t=3
    # Total triples: C(24,3) = 2024
    # We represent the set of covered triples as a Python integer bitmask.
    # Each bit i in the mask corresponds to the i-th triple in a fixed ordering.
    
    # Map each triple to a unique index 0..N-1
    triples = list(itertools.combinations(range(v), t))
    N_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    full_mask = (1 << N_triples) - 1
    
    # Helper to get mask for a block
    def get_block_mask(block):
        mask = 0
        for comb in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(sorted(comb))]
            mask |= (1 << idx)
        return mask

    # Step 1: Generate a pool of candidate blocks using Randomized Greedy
    # We want a diverse pool of good blocks.
    pool_size = 2000
    candidates = []
    
    # We will generate blocks by starting from random uncovered triples
    # and greedily extending them.
    
    # To ensure diversity, we randomize the choice of the next point in the greedy step.
    
    for _ in range(pool_size):
        if time.time() - start_time > timeout - 20: # Leave time for VNS
            break
            
        # Pick a random triple to start the block
        # Ideally, we should pick from triples that are "hard" to cover, but for pool generation,
        # random is okay if we generate many.
        start_triple = random.choice(triples)
        block = list(start_triple)
        block_set = set(block)
        
        # Greedily extend to size k
        while len(block) < k:
            best_x = -1
            best_gain = -1
            best_rand = 0
            
            # To add randomness, we can consider the top few gainers and pick randomly,
            # or add noise to the gain.
            # Let's just pick the best, but if ties, pick random.
            # Or better: pick from top 5 with probability proportional to gain?
            # Simpler: add small random noise to gain.
            
            block_set = set(block)
            # Precompute combinations of block of size t-1
            # For t=3, this is pairs.
            pairs = list(itertools.combinations(block, t - 1))
            
            gains = []
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in pairs:
                    tri = (x, p[0], p[1])
                    tri_sorted = tuple(sorted(tri))
                    # We don't have a "uncovered" set here because this is just pool generation.
                    # We want blocks that cover *many* triples, regardless of what else is covered.
                    # So gain is just the number of triples covered by adding x.
                    # Since we don't track global coverage here, we just count how many
                    # triples of the form {x} U C(block, t-1) exist.
                    # This is always C(len(block), t-1) if we assume all are valid?
                    # No, we just want to maximize the number of new triples.
                    # Since we don't know what's covered, we can just maximize the number
                    # of triples formed. This is constant for all x?
                    # Wait, the triples formed are specific.
                    # If we just want a "good" block, we want one that covers many distinct triples.
                    # Any k-subset covers C(k,t) triples.
                    # So all blocks are equal in total coverage capacity.
                    # The difference is *which* triples they cover.
                    # So for pool generation, we should just generate random k-subsets?
                    # No, that's too sparse.
                    # Randomized Greedy usually works by fixing a partial block and extending it
                    # to cover specific uncovered items.
                    # But here we don't have a global "uncovered" state for pool gen.
                    
                    # Alternative: Generate blocks by picking a random center triple,
                    # then picking random other points?
                    # Or: Pick a random triple. Then pick a random 4-subset of the remaining points?
                    # This yields uniform random k-subsets containing that triple.
                    # This ensures every triple is the "seed" for many blocks.
                    
                    pass # Logic revised below
                
            # Revised Pool Generation Strategy:
            # 1. Pick a random triple T.
            # 2. Pick k-t random points from the remaining v-t points.
            # 3. This creates a block containing T.
            # This ensures that for any triple, there are many candidate blocks containing it.
            # It's a simple and effective way to build a diverse pool.
            
            break # Exit the while loop logic above, use the simpler method below.
        
        # Simpler Pool Generation:
        # Pick a random triple.
        # Pick k-t random distinct points from the rest.
        # This is O(1) per block.
        
    # Let's restart the pool generation loop with the simpler strategy
    candidates = []
    for _ in range(pool_size):
        if time.time() - start_time > timeout - 20:
            break
        # Pick random triple
        triple = random.choice(triples)
        # Pick k-t random points from remaining
        remaining = [p for p in range(v) if p not in triple]
        extra = random.sample(remaining, k - t)
        block = sorted(list(triple) + extra)
        candidates.append(block)
        
    # Remove duplicates from candidates
    unique_candidates = list({tuple(b) for b in candidates})
    
    # Precompute masks for unique candidates
    candidate_masks = [get_block_mask(b) for b in unique_candidates]
    
    # If pool is too small, we might need more. But 2000 should be fine.
    # If we have fewer unique candidates, generate more?
    # For v=24, k=7, C(24,7) is large, so 2000 samples will likely be mostly unique.
    
    # Step 2: VNS on permutation of candidate indices
    n_cand = len(unique_candidates)
    
    # Initial permutation: Sort by mask "popcount" descending? 
    # Or just random.
    # Let's start with a random permutation.
    perm = list(range(n_cand))
    random.shuffle(perm)
    
    # Objective function: Find the smallest prefix length m such that OR of masks[perm[0:m]] == full_mask
    # If no such m exists (which shouldn't happen if pool is good enough to cover all),
    # return a large penalty.
    
    def evaluate(perm):
        current_mask = 0
        for i in range(len(perm)):
            current_mask |= candidate_masks[perm[i]]
            if current_mask == full_mask:
                return i + 1
        return float('inf')
        
    best_perm = perm[:]
    best_score = evaluate(best_perm)
    
    # VNS
    # Neighborhoods:
    # 1. Swap i, j
    # 2. Reverse i..j
    # 3. Move i to j
    
    # We will iterate through neighborhoods, performing local search (first improvement)
    # then shaking (random perturbation) if no improvement.
    
    # To keep it simple and fast, we will do a fixed number of VNS iterations.
    
    # Optimization: 
    # Instead of full VNS which can be slow, we can use a simpler "Greedy + Local Search" on the SET.
    # But the directive asks for VNS on permutations.
    # Let's implement a lightweight VNS.
    
    # We can optimize evaluation by caching? No, perm changes.
    # Evaluation is O(n_cand * cost_of_OR). cost_of_OR is O(1) for big ints.
    # So evaluate is O(n_cand).
    # n_cand ~ 2000. 2000 ops is fast.
    # We can do many iterations.
    
    max_vns_iters = 10000
    vns_iter = 0
    
    while vns_iter < max_vns_iters and time.time() - start_time < timeout - 5:
        vns_iter += 1
        
        improved = False
        
        # Neighborhood 1: Swap
        # Try swapping each pair? O(n^2) * O(n) eval = O(n^3). Too slow.
        # Instead, try random swaps or limited local search.
        # Let's do a "First Improvement" local search on swaps of adjacent or nearby elements?
        # Or just random swaps and accept if better.
        
        # Let's try: Pick 2 random indices, swap, evaluate. If better, accept.
        # This is a simple random walk / local search.
        # To make it VNS-like, we can vary the "distance" of the swap.
        
        # Neighborhood k: Swap elements at distance k?
        # Let's just use random swaps and reversals.
        
        # Try a few random moves
        moves_made = 0
        for _ in range(10): # Try 10 random moves per VNS iter
            if time.time() - start_time > timeout - 5:
                break
                
            # Random move type
            move_type = random.choice(['swap', 'reverse', 'move'])
            
            new_perm = perm[:]
            
            if move_type == 'swap':
                i, j = random.sample(range(n_cand), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            elif move_type == 'reverse':
                i, j = sorted(random.sample(range(n_cand), 2))
                # Reverse segment i..j
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
            else: # move
                i, j = random.sample(range(n_cand), 2)
                item = new_perm.pop(i)
                new_perm.insert(j, item)
                
            score = evaluate(new_perm)
            if score < best_score:
                best_score = score
                best_perm = new_perm
                perm = new_perm
                improved = True
                break # Accept and restart neighborhood search? Or just continue?
                # In VNS, we usually do local search until local optimum.
                # Here, we just accept first improvement to keep it fast.
            
        if not improved:
            # Shake: Randomly perturb the current best_perm to escape local optimum
            # Swap 3 random elements
            for _ in range(3):
                i, j = random.sample(range(n_cand), 2)
                perm[i], perm[j] = perm[j], perm[i]
            perm = perm[:]
            
    # Extract the best blocks
    if best_score == float('inf'):
        # Fallback: Just return the first n_cand blocks? Or greedy from scratch?
        # This shouldn't happen if pool is good.
        # Let's do a simple greedy on the pool to ensure validity.
        blocks = []
        covered_mask = 0
        # Sort candidates by popcount descending?
        # Just iterate
        for i in range(n_cand):
            if covered_mask == full_mask:
                break
            blocks.append(unique_candidates[perm[i]])
            covered_mask |= candidate_masks[perm[i]]
        return blocks
        
    return [unique_candidates[i] for i in best_perm[:best_score]]

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")

import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115  # Safety margin

    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(triples)}
    num_triples = len(triples)
    
    # Pre-compute triple indices for all blocks
    # This is the heavy part, but done once.
    # Number of blocks: C(24, 7) = 346,104
    # For each, C(7, 3) = 35 triples.
    # Total ~12M lookups. Should be < 2-3 seconds.
    block_triple_indices = {}
    for b in itertools.combinations(range(v), k):
        block_triple_indices[b] = [triple_to_idx[tup] for tup in itertools.combinations(b, t)]

    def greedy_cover(randomize_tie_break=False):
        uncovered = [True] * num_triples
        blocks_used = []
        
        while any(uncovered):
            if time.time() > deadline:
                return None
                
            # Pick an uncovered triple
            uncovered_idx = -1
            for i in range(num_triples):
                if uncovered[i]:
                    uncovered_idx = i
                    break
            
            if uncovered_idx == -1:
                break
                
            target_triple = triples[uncovered_idx]
            
            best_block = None
            best_gain = -1
            best_gain_blocks = []
            
            remaining_points = [p for p in range(v) if p not in target_triple]
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(target_triple) + list(combo)))
                
                # Get pre-computed indices
                idxs = block_triple_indices[block]
                
                gain = 0
                for idx in idxs:
                    if uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_gain_blocks = [block]
                elif gain == best_gain:
                    best_gain_blocks.append(block)
                    
            if best_gain <= 0:
                break
                
            if randomize_tie_break and best_gain_blocks:
                best_block = random.choice(best_gain_blocks)
            else:
                best_block = best_gain_blocks[0]
                
            blocks_used.append(list(best_block))
            
            # Update uncovered
            for idx in block_triple_indices[best_block]:
                uncovered[idx] = False
                
        return blocks_used

    best_solution = None
    best_len = float('inf')
    
    # Deterministic pass
    sol = greedy_cover(False)
    if sol:
        best_solution = sol
        best_len = len(sol)
        
    # Random restarts
    # We have plenty of time.
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        random.seed()
        sol = greedy_cover(True)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
            
    return best_solution

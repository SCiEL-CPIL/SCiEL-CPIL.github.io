import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triple indices for each block
    # Using numpy for fast bitset operations later
    block_triple_indices = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # For ILS, we need to know which blocks cover which triples.
    # We will represent the "coverage" of the current solution as a count array.
    # We will represent "uncovered triples" as a boolean array.
    
    # Precompute for each block, the list of triple indices it covers
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices.append(tris)
    
    # Convert to numpy arrays for potential speedup in counting?
    # Actually, lists are fine for iteration, but numpy is better for set ops.
    # Let's stick to lists for block_triple_indices as they are small (35 triples per block).
    
    def get_uncovered_count(counts):
        # counts is a numpy array of length num_triples
        return np.count_nonzero(counts == 0)
    
    def strong_greedy(seed):
        rng = random.Random(seed)
        # counts[i] = number of selected blocks covering triple i
        counts = np.zeros(num_triples, dtype=np.int32)
        selected = []
        selected_set = set()
        
        # Precompute gain for all blocks initially?
        # Gain of block b = number of uncovered triples in b
        # Initially all are uncovered, so gain is constant (35).
        # We need to update gains as we add blocks.
        # This is expensive. 
        # Alternative: 
        # 1. Pick a block that covers the most uncovered triples.
        # 2. Update counts.
        # 3. Repeat.
        
        # To speed up, we can maintain a list of uncovered triples.
        # But checking "which block covers the most uncovered" is O(N_blocks * 35).
        # N_blocks = 346k. 346k * 35 = 12M ops per step.
        # If we need ~70 steps, that's 840M ops. Might be too slow in Python.
        
        # Better: 
        # Use a priority queue? No, gains change.
        # Use randomization: Pick a random triple, pick a random block covering it.
        # This is a "random greedy" or "heuristic greedy".
        # Let's try a "randomized greedy" that picks the block with max gain among a sample?
        # Or just pick the block with max gain exactly, but optimized.
        
        # Optimization: 
        # Maintain `uncovered_triples` as a list.
        # When we add a block, we remove its triples from `uncovered_triples`.
        # To find the best block, we still need to scan all blocks.
        
        # Let's try a different strong start:
        # Start with a random valid covering (if we had one) or a large random set.
        # Then minimize.
        
        # Actually, let's just do a fast greedy:
        # 1. Pick random triple from uncovered.
        # 2. Pick random block covering it.
        # 3. Add block.
        # This is fast O(1) per step (amortized).
        # It might produce a larger initial solution, but ILS can fix it.
        
        uncovered_set = set(range(num_triples))
        
        while uncovered_set and time.time() < deadline - 30:
            # Pick random uncovered triple
            tri_idx = next(iter(uncovered_set))
            # Pick random block covering it
            candidates = triple_to_blocks[tri_idx]
            # Filter out already selected? 
            # If we pick a selected block, gain is 0 for that triple (already covered).
            # But we want to cover NEW triples.
            # So we should pick a block that covers at least one OTHER uncovered triple?
            # Or just pick any block covering tri_idx.
            # If it's already selected, it doesn't help.
            # So filter candidates to those not in selected_set.
            
            valid_candidates = [b for b in candidates if b not in selected_set]
            if not valid_candidates:
                # Should not happen if uncovered is not empty
                break
                
            chosen = rng.choice(valid_candidates)
            selected.append(chosen)
            selected_set.add(chosen)
            
            # Update counts and uncovered_set
            for tri_idx_in_block in block_triple_indices[chosen]:
                counts[tri_idx_in_block] += 1
                if counts[tri_idx_in_block] == 1:
                    uncovered_set.discard(tri_idx_in_block)
                    
        return selected, counts

    def local_search_minimize(selected_blocks, counts, time_limit):
        sol = list(selected_blocks)
        sol_set = set(sol)
        # counts is already numpy array
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle order for randomness?
            rng = random.Random(int(time.time() * 1000) % 10000)
            order = list(range(len(sol)))
            rng.shuffle(order)
            
            for i in order:
                if i >= len(sol): continue
                idx = sol[i]
                # Check if redundant
                redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
        return sol, counts

    def ills(selected_blocks, counts, time_limit, seed):
        rng = random.Random(seed)
        sol = list(selected_blocks)
        sol_set = set(sol)
        # counts is numpy array, passed by reference? No, copy it.
        cur_counts = counts.copy()
        
        best_sol = list(sol)
        best_counts = counts.copy()
        best_len = len(sol)
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < time_limit and T > 0.01:
            # Move: Swap or Replace
            
            # Try to find a block to remove
            if not sol:
                break
                
            to_remove = rng.choice(sol)
            
            # Check impact of removing to_remove
            # Uncovered triples if removed:
            # Those where count == 1
            affected_tris = []
            for tri_idx in block_triple_indices[to_remove]:
                if cur_counts[tri_idx] == 1:
                    affected_tris.append(tri_idx)
            
            if not affected_tris:
                # Redundant block, remove it
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                for tri_idx in block_triple_indices[to_remove]:
                    cur_counts[tri_idx] -= 1
                
                if len(sol) < best_len:
                    best_len = len(sol)
                    best_sol = list(sol)
                    best_counts = cur_counts.copy()
            else:
                # We need to replace to_remove with a block that covers all affected_tris
                # AND is not in sol.
                
                # Find candidates
                # Intersection of triple_to_blocks for all affected_tris
                # This can be expensive.
                # Let's pick one affected triple and find its blocks.
                # Then check if they cover the rest.
                
                target_tri = affected_tris[0]
                candidates = triple_to_blocks[target_tri]
                
                found = False
                for cand in candidates:
                    if cand in sol_set:
                        continue
                    
                    # Check if cand covers all affected_tris
                    covers_all = True
                    cand_tris = set(block_triple_indices[cand])
                    for at in affected_tris:
                        if at not in cand_tris:
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Perform swap
                        sol.remove(to_remove)
                        sol_set.discard(to_remove)
                        for tri_idx in block_triple_indices[to_remove]:
                            cur_counts[tri_idx] -= 1
                            
                        sol.append(cand)
                        sol_set.add(cand)
                        for tri_idx in block_triple_indices[cand]:
                            cur_counts[tri_idx] += 1
                        
                        found = True
                        break
                        
                if not found:
                    # Try a "kick": Remove a random block (even if it breaks coverage)
                    # and add a random block?
                    # This is a "restart" or "perturbation".
                    # Let's just do a random kick: remove 1 random, add 1 random.
                    # This might increase uncovered count, but SA can accept it.
                    
                    to_remove_kick = rng.choice(sol)
                    to_add_kick = rng.choice(range(num_blocks))
                    
                    if to_add_kick in sol_set:
                        continue
                        
                    # Calculate delta
                    # Remove to_remove_kick
                    for tri_idx in block_triple_indices[to_remove_kick]:
                        cur_counts[tri_idx] -= 1
                    sol.remove(to_remove_kick)
                    sol_set.discard(to_remove_kick)
                    
                    # Add to_add_kick
                    for tri_idx in block_triple_indices[to_add_kick]:
                        cur_counts[tri_idx] += 1
                    sol.append(to_add_kick)
                    sol_set.add(to_add_kick)
                    
                    # Check if better
                    # Energy = len(sol) + penalty * uncovered
                    # Since len is same, just check uncovered.
                    # Actually, we want to minimize len.
                    # If we can't reduce len, we stay.
                    # This kick is only useful if it leads to a better swap later.
                    
                    # For simplicity, let's revert if it doesn't immediately help?
                    # No, SA needs to explore.
                    # But we don't have a proper energy function here.
                    # Let's just keep it if it doesn't increase uncovered count?
                    # Or just accept it with probability based on T?
                    
                    # Let's calculate uncovered count change.
                    # This is expensive.
                    # Let's just accept the swap if it covers the affected triples.
                    # If not, we might be stuck.
                    # Let's try a different move: 
                    # "1-1 exchange" where we remove 1 and add 1.
                    # We already tried to find a perfect replacement.
                    # If not found, try to find a replacement that covers MOST affected?
                    
                    # For now, let's just revert the kick and try another remove candidate.
                    sol.remove(to_add_kick)
                    sol_set.discard(to_add_kick)
                    for tri_idx in block_triple_indices[to_add_kick]:
                        cur_counts[tri_idx] -= 1
                    
                    sol.append(to_remove_kick)
                    sol_set.add(to_remove_kick)
                    for tri_idx in block_triple_indices[to_remove_kick]:
                        cur_counts[tri_idx] += 1
                    
            T *= cooling
            
        return best_sol, best_counts

    best_sol = None
    best_counts = None
    best_size = num_blocks
    
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        print(f"Restart {i}", file=sys.stderr)
        
        # Strong Greedy
        g_sol, g_counts = strong_greedy(1000 + i)
        
        # Local Search
        ls_sol, ls_counts = local_search_minimize(g_sol, g_counts, deadline - 40)
        
        if len(ls_sol) < best_size:
            # Verify
            # We trust the counts if they are correct.
            # But let's double check if len is very small.
            if len(ls_sol) < 80:
                 # Quick verification
                 cover_set = set()
                 for idx in ls_sol:
                     for tri_idx in block_triple_indices[idx]:
                         cover_set.add(tri_idx)
                 if len(cover_set) == num_triples:
                     best_size = len(ls_sol)
                     best_sol = ls_sol
                     best_counts = ls_counts
            else:
                 best_size = len(ls_sol)
                 best_sol = ls_sol
                 best_counts = ls_counts

        # ILS
        if time.time() < deadline - 10:
            ills_sol, ills_counts = ills(ls_sol, ls_counts, deadline - 5, 2000 + i)
            if len(ills_sol) < best_size:
                # Verify
                if len(ills_sol) < 80:
                     cover_set = set()
                     for idx in ills_sol:
                         for tri_idx in block_triple_indices[idx]:
                             cover_set.add(tri_idx)
                     if len(cover_set) == num_triples:
                         best_size = len(ills_sol)
                         best_sol = ills_sol
                         best_counts = ills_counts
                else:
                     best_size = len(ills_sol)
                     best_sol = ills_sol
                     best_counts = ills_counts

    if best_sol is None:
        # Fallback
        g_sol, g_counts = strong_greedy(1)
        ls_sol, ls_counts = local_search_minimize(g_sol, g_counts, deadline - 5)
        best_sol = ls_sol
        
    return [all_blocks[i] for i in best_sol]

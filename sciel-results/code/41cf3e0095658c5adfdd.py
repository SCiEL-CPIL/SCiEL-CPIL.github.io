import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precomputation
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    block_triple_sets = [set() for _ in range(num_blocks)] # For fast subset checks
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris
        block_triple_sets[i] = set(tris)

    best_sol = None
    best_energy = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # 1. Greedy Construction
        sol = greedy_construct(rng, num_blocks, block_triple_indices, num_triples, deadline - 5)
        
        # 2. Local Search (SA)
        time_left_for_sa = deadline - 5 - time.time()
        if time_left_for_sa < 1:
            break
            
        sol = local_search(sol, num_blocks, block_triple_indices, block_triple_sets, triple_to_blocks, num_triples, time_left_for_sa, rng)
        
        # Evaluate
        if sol:
            # Check validity and count
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                for t_idx in block_triple_indices[b_idx]:
                    covered[t_idx] = True
            
            uncovered_count = np.sum(~covered)
            
            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)
                    
                # If we hit the lower bound or a very good number, we might stop?
                # Lower bound is 69. If we get 70, that's great.
                if energy <= 71: # Heuristic: if we get close to LB, stop searching for better?
                    # Actually, keep going if time permits, but 71 is excellent.
                    pass

    if best_sol is None:
        # Fallback: return a valid solution from the last run or a greedy one
        # This shouldn't happen if greedy works
        return []
        
    # Convert indices back to point lists
    return [list(all_blocks[i]) for i in best_sol]

def greedy_construct(rng, num_blocks, block_triple_indices, num_triples, deadline):
    uncovered = [1] * num_triples
    selected = []
    selected_set = set()
    
    while any(uncovered) and time.time() < deadline:
        best_gain = -1
        candidates = []
        
        # Optimization: Only check blocks that haven't been selected
        # In early stages, most blocks are candidates.
        for i in range(num_blocks):
            if i in selected_set:
                continue
            
            gain = 0
            for t_idx in block_triple_indices[i]:
                if uncovered[t_idx]:
                    gain += 1
                    # Early exit if we can't beat best_gain?
                    # Max possible gain is C(k,t) = 35.
                    if gain > best_gain:
                        # We can't break here because we need to count all to know if it's equal?
                        # No, if gain > best_gain, it's the new best. But we don't know if it's 35 yet.
                        # If gain == 35, we can break.
                        if gain == 35:
                            break
            if gain > best_gain:
                best_gain = gain
                candidates = [i]
            elif gain == best_gain and gain > 0:
                candidates.append(i)
        
        if best_gain <= 0:
            break
            
        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_set.add(chosen)
        
        for t_idx in block_triple_indices[chosen]:
            uncovered[t_idx] = 0
            
    return selected

def local_search(initial_sol, num_blocks, block_triple_indices, block_triple_sets, triple_to_blocks, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)
    
    # Coverage counts
    counts = [0] * num_triples
    for idx in sol:
        for t_idx in block_triple_indices[idx]:
            counts[t_idx] += 1
            
    def get_uncovered():
        return sum(1 for c in counts if c == 0)
        
    current_uncovered = get_uncovered()
    current_energy = len(sol) + 100 * current_uncovered
    best_sol = list(sol)
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < time_limit and T > 0.01:
        move_type = rng.random()
        
        if move_type < 0.6 and sol:
            # Removal
            to_remove = rng.choice(sol)
            is_redundant = True
            delta_uncovered = 0
            for t_idx in block_triple_indices[to_remove]:
                if counts[t_idx] == 1:
                    is_redundant = False
                    delta_uncovered += 1
                    break # Just need to know if it's redundant? No, need count for energy if accepted.
                    # Actually, if not redundant, we need exact delta_uncovered.
            
            if is_redundant:
                # Always accept
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                for t_idx in block_triple_indices[to_remove]:
                    counts[t_idx] -= 1
                current_uncovered = get_uncovered() # Should be same
                current_energy = len(sol) + 100 * current_uncovered
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                # Recalculate delta_uncovered properly if we accept
                # (We broke early above, so recalc)
                delta_uncovered = 0
                for t_idx in block_triple_indices[to_remove]:
                    if counts[t_idx] == 1:
                        delta_uncovered += 1
                
                delta_e = 1 + 100 * delta_uncovered
                if rng.random() < pow(2.718, -delta_e / T):
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    for t_idx in block_triple_indices[to_remove]:
                        counts[t_idx] -= 1
                    current_uncovered += delta_uncovered
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.9 and sol:
            # Swap
            to_remove = rng.choice(sol)
            
            # Find triples that will be uncovered
            uncovered_tris = []
            for t_idx in block_triple_indices[to_remove]:
                if counts[t_idx] == 1:
                    uncovered_tris.append(t_idx)
            
            if not uncovered_tris:
                # This is a removal, handled above. Skip or treat as removal.
                continue
            
            # Find a block that covers all uncovered_tris
            candidates = []
            for t_idx in uncovered_tris:
                for b_idx in triple_to_blocks[t_idx]:
                    if b_idx in sol_set:
                        continue
                    # Check if b_idx covers ALL uncovered_tris
                    if block_triple_sets[b_idx].issuperset(uncovered_tris):
                        candidates.append(b_idx)
                        break # One candidate is enough for this specific t_idx? 
                        # No, we need a block that covers ALL. 
                        # If we break here, we might miss a better candidate for other t_idx?
                        # Actually, we iterate t_idx in uncovered_tris. 
                        # If we find a candidate for the first t_idx, we must check if it covers the rest.
                        # The issubset check does that.
                        # So if it passes, it's a valid candidate for the whole set.
                        # We can break the inner loop.
                        # But should we look for more candidates? 
                        # Picking randomly from valid candidates is good.
                        # Let's collect a few.
            
            if candidates:
                to_add = rng.choice(candidates)
                
                # Perform swap
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                for t_idx in block_triple_indices[to_remove]:
                    counts[t_idx] -= 1
                
                sol.append(to_add)
                sol_set.add(to_add)
                for t_idx in block_triple_indices[to_add]:
                    counts[t_idx] += 1
                    
                # Energy change:
                # Size is same.
                # Uncovered change: 
                # We removed to_remove (lost coverage of uncovered_tris).
                # We added to_add (gained coverage of uncovered_tris, and maybe others).
                # Since to_add covers all uncovered_tris, those are now covered.
                # Did to_add cover any other previously uncovered triples? 
                # No, because we are in a state where we just created uncovered_tris by removing to_remove.
                # Before removal, all were covered. 
                # After removal, uncovered_tris are uncovered. 
                # Adding to_add covers them. 
                # Does to_add cover any *other* triples that were covered by to_remove? 
                # If so, counts for those increase.
                # Does to_add cover any triples that were *only* covered by to_remove? 
                # Yes, that's uncovered_tris.
                # So uncovered_count becomes 0 (relative to this move).
                # Wait, what if the current state was already invalid (had other uncovered triples)?
                # current_uncovered tracks the global count.
                # Let's just recalculate energy or update delta.
                
                # Simplest: Recalculate uncovered count? Expensive.
                # Delta calculation:
                # Old state: Uncovered set U_old.
                # Remove to_remove: U_new = U_old U {t | count(t)==1 in old state and t in to_remove}
                # Add to_add: U_final = U_new \ {t in to_add}
                # We know U_new contains uncovered_tris (which were count 1).
                # to_add covers all uncovered_tris.
                # So U_final = U_old \ {t in to_add \ to_remove} ... this is getting complex.
                
                # Let's just compute new uncovered count if we accept?
                # Or estimate.
                # Since we are in SA, we can afford to be a bit slower if it's rare.
                # But swaps are frequent.
                
                # Let's assume the move is neutral or improving.
                # If it was a valid state (U_old=0), then U_final=0. Energy change = 0.
                # If it was invalid, it might improve.
                
                # For simplicity and robustness, let's just accept neutral swaps (energy change 0) 
                # and only accept improving ones? 
                # Actually, SA needs to accept worse moves.
                # Let's compute the exact delta_uncovered.
                
                # This is too slow to compute exactly in the inner loop.
                # Approximation:
                # If current_uncovered == 0:
                #   delta_e = 0
                # Else:
                #   delta_e = 0 (assume neutral)
                
                # Let's just accept the swap if it doesn't increase energy significantly.
                # Since we don't know delta_e exactly without recounting, 
                # and recounting is O(N_triples) = 2024, which is fast.
                
                new_uncovered = get_uncovered()
                new_energy = len(sol) + 100 * new_uncovered
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < pow(2.718, -delta_e / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject: Undo swap
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    for t_idx in block_triple_indices[to_add]:
                        counts[t_idx] -= 1
                        
                    sol.append(to_remove)
                    sol_set.add(to_remove)
                    for t_idx in block_triple_indices[to_remove]:
                        counts[t_idx] += 1

        T *= cooling
        
        if time.time() > time_limit:
            break
            
    return best_sol

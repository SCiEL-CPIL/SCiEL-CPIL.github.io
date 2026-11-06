import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs.
    # There are 15 ways to partition 6 items into 3 pairs.
    # Total configs: C(16,6) * 15
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    config_triples = [] # List of lists of triple indices
    
    # Optimization: Precompute the 15 pairings for a generic set of 6 indices 0..5
    # and map them to the actual points.
    # This saves time in the inner loop.
    generic_pairings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]

    for points in combinations(range(n), 6):
        p = points
        for gp in generic_pairings:
            # Map generic indices to actual points
            pair1 = (p[gp[0][0]], p[gp[0][1]])
            pair2 = (p[gp[1][0]], p[gp[1][1]])
            pair3 = (p[gp[2][0]], p[gp[2][1]])
            
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            # We store the list of triples for quick access in removal logic
            k222_list.append(trans_indices)
            config_triples.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    def time_left():
        return deadline - time.time()

    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    def is_addable(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def local_search_targeted(active, counts, current_size):
        """
        Local search that:
        1. Greedily adds any valid triple.
        2. If stuck, identifies saturated K(2,2,2)s (count 7).
        3. Removes a triple from a saturated K(2,2,2) to break the constraint.
        4. Repeats.
        """
        improved = True
        while improved and time_left() > 2:
            improved = False
            
            # Phase 1: Greedy Add
            # We iterate through all inactive triples. 
            # To avoid bias, we can shuffle the inactive list or just iterate 0..N-1.
            # Iterating 0..N-1 is fast. Shuffling helps escape order-based biases.
            # Let's use a static order for speed, but if we get stuck, we rely on the removal phase.
            
            added_any = False
            for i in range(num_triples):
                if not active[i] and is_addable(i, counts):
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    added_any = True
                    improved = True
                    # We could break here to restart the scan, but continuing allows 
                    # multiple adds in one pass if they don't conflict. 
                    # However, adding one might block another. 
                    # Since we check `is_addable` dynamically, it's safe to continue.
            
            if added_any:
                continue # Restart loop to check for more adds after the batch

            # Phase 2: If no adds, we are in a local optimum.
            # Find saturated configurations (count == 7)
            # We need to pick one and remove a triple from it.
            
            # To make the kick effective, we can pick a random saturated config.
            # Or, we can pick a triple that is in many saturated configs.
            # Let's try: Pick a random saturated config, then pick a random triple from it to remove.
            
            # Collect indices of saturated configs
            # This scan is O(num_k222s). num_k222s is ~ 8000. This is fast.
            saturated_configs = [k for k in range(num_k222s) if counts[k] == 7]
            
            if not saturated_configs:
                break # Should not happen if no adds were possible and we started with a valid state,
                      # unless we are at the absolute maximum.
            
            # Randomly select a saturated config
            # To diversify, we can also look at the "density" of the triples in it.
            k_idx = random.choice(saturated_configs)
            
            # Get the triples in this config
            triples_in_config = config_triples[k_idx]
            
            # We want to remove one of these triples.
            # Which one? 
            # Strategy A: Random.
            # Strategy B: The one that appears in the most other saturated configs (most "critical").
            # Strategy C: The one with the highest degree (appears in most K222s overall).
            
            # Let's use a hybrid: 
            # Calculate a score for each triple in the config.
            # Score = number of saturated configs it belongs to.
            # Remove the one with the highest score (most likely to unlock other adds).
            
            best_remove_idx = -1
            best_score = -1
            
            for t_idx in triples_in_config:
                if not active[t_idx]:
                    continue # Should be active since count is 7
                
                # Count how many saturated configs this triple is part of
                # This is O(degree of t_idx). Degree is small (each triple is in few K222s).
                # Actually, we can precompute or just iterate its k222s.
                score = 0
                for k2 in triple_to_k222s[t_idx]:
                    if counts[k2] == 7:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_remove_idx = t_idx
            
            if best_remove_idx != -1:
                # Remove it
                active[best_remove_idx] = False
                current_size -= 1
                for k_idx2 in triple_to_k222s[best_remove_idx]:
                    counts[k_idx2] -= 1
                improved = True
                # Loop will restart and try to add things again

        return active, counts, current_size

    def simulated_annealing(active, counts, current_size, max_iters=500):
        """
        Simulated annealing to escape deep local minima if targeted search stalls.
        """
        best_size_sa = current_size
        best_active_sa = active[:]
        best_counts_sa = counts[:]
        
        temp = 1.0
        cooling = 0.99
        iters = 0
        
        while iters < max_iters and time_left() > 5:
            iters += 1
            temp *= cooling
            
            # Single swap
            active_edges = [i for i in range(num_triples) if active[i]]
            inactive_edges = [i for i in range(num_triples) if not active[i]]
            
            if not active_edges or not inactive_edges:
                continue
                
            t_remove = random.choice(active_edges)
            t_add = random.choice(inactive_edges)
            
            # Try swap
            for k_idx in triple_to_k222s[t_remove]:
                counts[k_idx] -= 1
            
            valid = is_addable(t_add, counts)
            
            if valid:
                active[t_remove] = False
                active[t_add] = True
                # Size unchanged
                
                for k_idx in triple_to_k222s[t_add]:
                    counts[k_idx] += 1
                
                if current_size > best_size_sa:
                    best_size_sa = current_size
                    best_active_sa = active[:]
                    best_counts_sa = counts[:]
            else:
                # Reject: undo
                for k_idx in triple_to_k222s[t_remove]:
                    counts[k_idx] += 1
            
            # Occasionally do a double swap to jump over barriers
            if random.random() < 0.1 and len(active_edges) >= 2 and len(inactive_edges) >= 2:
                t_r1, t_r2 = random.sample(active_edges, 2)
                t_a1, t_a2 = random.sample(inactive_edges, 2)
                
                for k_idx in triple_to_k222s[t_r1]:
                    counts[k_idx] -= 1
                for k_idx in triple_to_k222s[t_r2]:
                    counts[k_idx] -= 1
                
                valid1 = is_addable(t_a1, counts)
                if valid1:
                    for k_idx in triple_to_k222s[t_a1]:
                        counts[k_idx] += 1
                    valid2 = is_addable(t_a2, counts)
                    for k_idx in triple_to_k222s[t_a1]:
                        counts[k_idx] -= 1
                else:
                    valid2 = False
                
                if valid1 and valid2:
                    active[t_r1] = False
                    active[t_r2] = False
                    active[t_a1] = True
                    active[t_a2] = True
                    # Size unchanged
                    
                    for k_idx in triple_to_k222s[t_a1]:
                        counts[k_idx] += 1
                    for k_idx in triple_to_k222s[t_a2]:
                        counts[k_idx] += 1
                        
                    if current_size > best_size_sa:
                        best_size_sa = current_size
                        best_active_sa = active[:]
                        best_counts_sa = counts[:]
                else:
                    for k_idx in triple_to_k222s[t_r1]:
                        counts[k_idx] += 1
                    for k_idx in triple_to_k222s[t_r2]:
                        counts[k_idx] += 1

        return best_active_sa, best_counts_sa, best_size_sa

    best_size = 0
    best_active = None
    best_counts = None
    
    # ILS Loop
    # 1. Greedy Init
    active, counts, current_size = greedy_init()
    
    # 2. Targeted Local Search
    active, counts, current_size = local_search_targeted(active, counts, current_size)
    
    if current_size > best_size:
        best_size = current_size
        best_active = active[:]
        best_counts = counts[:]
        
    # 3. Iterated Local Search
    # We alternate between Targeted Local Search and Simulated Annealing kicks
    while time_left() > 10:
        # Perturb
        # Instead of random removal, we can use a larger random kick to escape
        # the basin of the targeted search.
        k = random.randint(10, 20)
        active_edges = [i for i in range(num_triples) if active[i]]
        if len(active_edges) < k:
            k = len(active_edges)
            
        for _ in range(k):
            if not active_edges: break
            t_idx = random.choice(active_edges)
            active_edges.remove(t_idx)
            active[t_idx] = False
            current_size -= 1
            for k_idx in triple_to_k222s[t_idx]:
                counts[k_idx] -= 1
                
        # Local Search on perturbed solution
        active, counts, current_size = local_search_targeted(active, counts, current_size)
        
        # If we are stuck (size didn't improve much or no adds possible), try SA
        if time_left() > 15:
            # Run a short SA phase to mix things up
            active, counts, current_size = simulated_annealing(active, counts, current_size, max_iters=200)
            # Then run targeted search again to clean up
            active, counts, current_size = local_search_targeted(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Final polish
    if best_active:
        active = best_active[:]
        counts = best_counts[:]
        current_size = best_size
        # Run local search one last time
        active, counts, current_size = local_search_targeted(active, counts, current_size)
        
        # Additional SA pass on the best solution
        if time_left() > 10:
            active, counts, current_size = simulated_annealing(active, counts, current_size, max_iters=300)
            active, counts, current_size = local_search_targeted(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Construct result from best_active
    if best_active is None:
        # Fallback
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")

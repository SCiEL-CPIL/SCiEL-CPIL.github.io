"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses an efficient local search with precomputed conflict counts.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is determined by three disjoint pairs
    k222_configs = []  # Each is a list of 8 triple indices
    
    def get_pairings(points):
        """Generate all ways to partition 6 points into 3 unordered pairs."""
        points = list(points)
        pairings = []
        
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        
        rec(points, [])
        return pairings
    
    seen_configs = set()
    
    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)

    # Local search with efficient violation tracking
    random.seed(42)
    start_time = time.time()
    time_limit = 110  # Leave some time for the function to return
    
    best_solution = None
    best_count = -1
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Greedy initialization: start with all False, add triples if they don't cause violation
        solution = [False] * num_triples
        # counts[c] is the number of True triples in config c
        counts = [0] * num_configs
        
        # Add triples in random order
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            # Check if adding this triple would cause any config to reach 8
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            
            if safe:
                solution[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
        
        # Now try to improve by flipping False triples to True if possible, 
        # and flipping True to False if it allows more additions? 
        # Actually, standard local search: 
        # 1. While there are violations, remove a triple from a violated config.
        # 2. Then try to add more triples.
        
        # First, ensure no violations (should be none from greedy, but let's be safe)
        # Greedy ensures no violation.
        
        # Now, try to add more triples by looking for False triples that can be set to True
        # without creating a violation.
        # But we already added all "safe" ones in random order. 
        # However, the order matters. A triple that was unsafe earlier might be safe now? 
        # No, adding triples only increases counts. So if it was unsafe, it remains unsafe.
        # So greedy gives a maximal set with respect to the order.
        
        # To improve, we can use local search:
        # Pick a random True triple, remove it (decrement counts).
        # Then try to add it back or add other False triples.
        # Better: Pick a random False triple. If it can be added, add it.
        # If not, maybe we can swap?
        
        # Let's do a simple hill climbing:
        # Current state is valid (no violations).
        # We want to maximize size.
        # Try flipping a True triple to False? That reduces size. Only do it if it allows adding 2 or more new triples.
        
        # Strategy:
        # 1. Try to add any False triple that is "safe" (doesn't complete a config to 8).
        #    Since we did this greedily, there might be none left?
        #    Wait, greedy added if safe at that moment. Since counts only increase, 
        #    any triple that was unsafe is still unsafe. 
        #    So the greedy set is maximal (no single triple can be added).
        
        # So we need to perform swaps or removals to reach a different local optimum.
        
        # Let's perform random restarts with different greedy orders, 
        # and also try to "shake" the solution.
        
        # Shake: Pick a random True triple, remove it. Then re-run greedy addition for the remaining False triples?
        # That's expensive.
        
        # Alternative: WalkSAT-like.
        # If we are stuck (maximal), we can try to remove a triple that is part of many "tight" configs (count=7).
        # Then we might be able to add different triples.
        
        # Let's just rely on multiple random restarts of the greedy algorithm with different shuffles.
        # This is often very effective for such combinatorial problems.
        
        current_count = sum(solution)
        if current_count > best_count:
            best_count = current_count
            best_solution = solution[:]
            
        # Try to improve current solution by local moves
        # Move: Remove a True triple, then try to add any False triple that becomes safe.
        # If we can add more than we removed, keep the change.
        
        # For efficiency, we'll just do a few iterations of "remove one, try to refill"
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            if not any(solution):
                break
                
            # Pick a random True triple to remove
            true_indices = [i for i in range(num_triples) if solution[i]]
            if not true_indices:
                break
            remove_idx = random.choice(true_indices)
            
            # Remove it
            solution[remove_idx] = False
            for c_idx in triple_to_configs[remove_idx]:
                counts[c_idx] -= 1
            
            # Try to add any False triple that is now safe
            added_any = False
            false_indices = [i for i in range(num_triples) if not solution[i]]
            random.shuffle(false_indices)
            
            for add_idx in false_indices:
                safe = True
                for c_idx in triple_to_configs[add_idx]:
                    if counts[c_idx] == 7:
                        safe = False
                        break
                if safe:
                    solution[add_idx] = True
                    for c_idx in triple_to_configs[add_idx]:
                        counts[c_idx] += 1
                    added_any = True
                    break # Only add one at a time? Or add all safe ones?
                    # Adding all safe ones is better.
            
            # Let's add ALL safe ones
            for add_idx in false_indices:
                if solution[add_idx]: continue
                safe = True
                for c_idx in triple_to_configs[add_idx]:
                    if counts[c_idx] == 7:
                        safe = False
                        break
                if safe:
                    solution[add_idx] = True
                    for c_idx in triple_to_configs[add_idx]:
                        counts[c_idx] += 1
            
            current_count = sum(solution)
            if current_count > best_count:
                best_count = current_count
                best_solution = solution[:]
                
            # If we didn't add anything, we are stuck in a local optimum for this removal.
            # The loop continues to try another removal.
    
    if best_solution is None:
        # Fallback: empty set (valid)
        best_solution = [False] * num_triples
        best_count = 0

    # Convert solution back to list of triples
    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    
    return result
# EVOLVE-BLOCK-END

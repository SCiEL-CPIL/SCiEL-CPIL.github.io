import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    # Optimization: Only iterate over unique sets of 6 points
    # There are C(16,6) = 8008 such sets.
    # For each set, there are 15 ways to partition into 3 pairs.
    # Total configs = 8008 * 15 = 120120.
    
    # We can optimize the pair generation by pre-computing partitions of 6 items
    # But for simplicity and speed, we generate on the fly.
    
    # To speed up, we can note that the structure of K(2,2,2) is highly symmetric.
    # However, 120k configs is manageable.
    
    # Let's optimize the inner loop.
    # Instead of generating all 15 partitions explicitly with lists, 
    # we can use a faster method or just accept the overhead if it's small.
    
    # Let's precompute the 15 pairings of indices 0..5
    indices_6 = [0, 1, 2, 3, 4, 5]
    base_pairings = []
    seen_pairs = set()
    # Generate all perfect matchings of 6 elements
    # There are 15 such matchings.
    def gen_matchings(remaining):
        if not remaining:
            yield []
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            rest = remaining[1:i] + remaining[i+1:]
            for sub in gen_matchings(rest):
                yield [(first, second)] + sub

    for m in gen_matchings(indices_6):
        norm = tuple(sorted(tuple(sorted(p)) for p in m))
        if norm not in seen_pairs:
            seen_pairs.add(norm)
            base_pairings.append(norm)
            
    # Now map these to actual points
    for pts in itertools.combinations(points, 6):
        # pts is a tuple of 6 distinct points
        # Map 0..5 to pts[0]..pts[5]
        # Actually, we just need to substitute
        for bp in base_pairings:
            # Substitute indices with actual points
            actual_pairs = []
            for p in bp:
                actual_pairs.append((pts[p[0]], pts[p[1]]))
            
            # Normalize for uniqueness (though base_pairings are already unique for the set 0..5,
            # different pts sets will produce different actual points)
            # Wait, the config ID should be based on the actual points.
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in actual_pairs))
            
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    # Structural Heuristic:
    # In AG(4,2), a triple {a,b,c} is a "line" if a^b^c == 0.
    # Lines are the basic building blocks of affine geometry.
    # Hypothesis: A system composed largely of lines might be more regular and dense.
    # Let's assign a score to each triple.
    # Score 1: a^b^c == 0 (Line)
    # Score 0: otherwise
    # We will prefer lines.
    
    triple_scores = np.zeros(num_triples, dtype=np.int8)
    for i, t in enumerate(all_triples):
        if t[0] ^ t[1] ^ t[2] == 0:
            triple_scores[i] = 1
        else:
            triple_scores[i] = 0
            
    # Pre-sort triples by score (descending), then by index (ascending)
    # This gives us a deterministic priority order.
    # However, we want to be dynamic. 
    # Strategy: 
    # 1. Start with empty set.
    # 2. At each step, find the highest scoring valid triple.
    # 3. If none, backtrack.
    
    # To make this efficient, we can use a max-heap or just scan.
    # Scanning 560 items is fast.
    
    # We will implement a recursive backtracking search with a depth limit
    # and a "best first" strategy.
    
    # Actually, pure backtracking might be too slow if it explores too much.
    # Let's use a "Greedy with Lookahead" approach.
    
    # Alternative: 
    # Use the pre-sorted order as a base, but allow "swaps" if it leads to a better structural outcome.
    
    # Let's stick to a deterministic greedy that uses the structural score.
    # If it gets stuck, it tries to remove the triple that was added last 
    # and try the next best option at that step.
    
    # This is essentially depth-first search with pruning, but guided by the score.
    
    # Given the time limit, we can run this DFS for a limited depth/time.
    
    best_set = []
    best_count = 0
    
    # We'll use a stack for manual backtracking to avoid recursion depth issues.
    # State: (current_set, next_candidate_index, count)
    # But we need to remember which choices were made to backtrack.
    
    # Simpler approach: 
    # Run a deterministic greedy. 
    # If it gets stuck, restart with a slightly different priority (e.g., invert score for non-lines, or break ties differently).
    # This is deterministic but explores different "regions" of the solution space based on structural preferences.
    
    def run_greedy(priority_order):
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # priority_order is a list of triple indices sorted by our heuristic
        # We iterate through this list. If a triple is valid, we add it.
        # This is a simple greedy. It doesn't backtrack.
        # To improve, we can do a "second pass" where we try to swap.
        
        for t_idx in priority_order:
            if not present[t_idx]:
                # Check if valid
                valid = True
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        valid = False
                        break
                if valid:
                    present[t_idx] = True
                    current_set.add(t_idx)
                    for cid in triple_to_configs[t_idx]:
                        config_counts[cid] += 1
                    
        return len(current_set), list(current_set)
        
    # Generate several deterministic priority orders based on structural features.
    
    # Order 1: Lines first, then non-lines. Ties by index.
    order1 = sorted(range(num_triples), key=lambda i: (-triple_scores[i], i))
    
    # Order 2: Non-lines first, then lines.
    order2 = sorted(range(num_triples), key=lambda i: (triple_scores[i], i))
    
    # Order 3: Randomized but seeded deterministically to create variation.
    # We can create a few seeded random orders.
    
    orders = [order1, order2]
    
    # Add a few random orders with fixed seeds
    import random
    for seed in [1, 2, 3, 4, 5]:
        r = random.Random(seed)
        ord_i = list(range(num_triples))
        r.shuffle(ord_i)
        orders.append(ord_i)
        
    # Run all these greedy passes.
    for ord_i in orders:
        if time.time() > deadline - 5:
            break
        count, s = run_greedy(ord_i)
        if count > best_count:
            best_count = count
            best_set = s
            
    # Now, try to improve the best set found using local search (deterministic).
    # We will try to add any missing triple. If none, try to swap one out for one in.
    # We prioritize swaps that increase the number of lines in the system.
    
    if best_set:
        # Convert best_set to indices
        current_set = set(best_set)
        present[:] = False
        config_counts[:] = 0
        for t_idx in current_set:
            present[t_idx] = True
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] += 1
                
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # 1. Try to add any missing triple
            # Sort missing triples by score
            missing = [i for i in range(num_triples) if not present[i]]
            missing.sort(key=lambda i: -triple_scores[i])
            
            for t_idx in missing:
                valid = True
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        valid = False
                        break
                if valid:
                    # Add it
                    present[t_idx] = True
                    current_set.add(t_idx)
                    for cid in triple_to_configs[t_idx]:
                        config_counts[cid] += 1
                    improved = True
                    # Restart the "add" phase to see if more can be added
                    break
            
            if improved:
                continue
                
            # 2. If no single addition works, try a swap
            # We want to remove a triple with low score and add one with high score.
            # Or simply try to find a swap that allows a new addition.
            
            # Let's try to find a triple that is "almost" addable (blocked by 1 config)
            # and see if we can remove a triple from that config to unblock it.
            
            candidates = []
            for t_idx in missing:
                blockers = 0
                blocking_config = -1
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        blockers += 1
                        blocking_config = cid
                if blockers == 1:
                    candidates.append((t_idx, blocking_config))
            
            # Sort candidates by score of the triple to add
            candidates.sort(key=lambda x: -triple_scores[x[0]])
            
            for t_add, cid_block in candidates:
                # Find a triple in cid_block to remove
                # Prefer removing low-score triples
                present_in_cid = [t for t in triples_in_config[cid_block] if present[t]]
                present_in_cid.sort(key=lambda t: triple_scores[t]) # Lowest score first
                
                for t_remove in present_in_cid:
                    # Check if removing t_remove allows adding t_add
                    # It unblocks cid_block.
                    # Does it block anything else? No, we are removing.
                    # Does adding t_add block anything else?
                    # We need to check if t_add is valid after removing t_remove.
                    
                    # Simulate removal
                    for cid in triple_to_configs[t_remove]:
                        config_counts[cid] -= 1
                        
                    # Check validity of t_add
                    valid = True
                    for cid in triple_to_configs[t_add]:
                        if config_counts[cid] == 7:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        present[t_remove] = False
                        current_set.remove(t_remove)
                        
                        present[t_add] = True
                        current_set.add(t_add)
                        
                        for cid in triple_to_configs[t_add]:
                            config_counts[cid] += 1
                            
                        improved = True
                        break
                    else:
                        # Restore
                        for cid in triple_to_configs[t_remove]:
                            config_counts[cid] += 1
                            
                if improved:
                    break
                    
    # Final result
    result = [all_triples[i] for i in current_set] if best_set else []
    
    return result

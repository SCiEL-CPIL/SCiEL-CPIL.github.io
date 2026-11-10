import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Strategy:
    # 1. Construct a dense linear triple system using the group Z_4 x Z_4.
    #    Points are (x, y) with x, y in {0, 1, 2, 3}.
    #    We define triples based on lines in the affine geometry or specific group differences.
    #    A known dense packing on 16 points can be derived from the affine plane AG(2, 4).
    #    However, AG(2,4) has 4-sets (lines of size 4). We need 3-sets.
    #    Alternatively, we can use a greedy approach guided by symmetry, or simply 
    #    use a known good configuration and refine it.
    
    # Let's try a specific construction:
    # Map points 0..15 to Z_4 x Z_4.
    # point_id = 4*x + y
    
    # We will generate candidate triples.
    # A linear triple system requires no pair repeated.
    
    # Let's use a randomized local search starting from a high-density initial state.
    # Initial state: Greedy packing with a bias towards "structured" triples.
    
    def point_to_id(x, y):
        return 4 * x + y

    def id_to_point(p):
        return (p // 4, p % 4)

    # Generate all possible triples
    all_triples = list(combinations(points, 3))
    
    # Pre-compute pairs for each triple for fast lookup
    triple_pairs = {}
    for t in all_triples:
        pairs = frozenset((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))
        # Actually, just store the set of pairs
        triple_pairs[t] = {
            frozenset((t[0], t[1])),
            frozenset((t[1], t[2])),
            frozenset((t[0], t[2]))
        }

    def calc_alpha(triples):
        if not triples:
            return n
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        max_ind = 0
        # Optimization: If we have a very dense system, alpha is small.
        # We can stop early if we find a large independent set? No, we need max.
        # But we can prune subsets?
        # For n=16, 65536 subsets is fast.
        
        # To speed up, we can check only subsets of size >= current best?
        # Generating combinations of size k is costly.
        # Bitmask iteration is better.
        
        for mask in range(1 << n):
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                cnt = bin(mask).count('1')
                if cnt > max_ind:
                    max_ind = cnt
                    # Early exit? No, there might be a larger one.
                    # But if max_ind == n, we can stop.
                    if max_ind == n:
                        return n
        return max_ind

    def is_valid_addition(triples_set, used_pairs, new_triple):
        # triples_set: set of triples (as tuples) for O(1) existence check if needed
        # used_pairs: set of frozensets of pairs
        for p in triple_pairs[new_triple]:
            if p in used_pairs:
                return False
        return True

    def try_add(triples, used_pairs, triple):
        if is_valid_addition(triples, used_pairs, triple):
            triples.append(triple)
            for p in triple_pairs[triple]:
                used_pairs.add(p)
            return True
        return False

    def remove_triple(triples, used_pairs, triple):
        triples.remove(triple)
        for p in triple_pairs[triple]:
            used_pairs.remove(p)

    # Phase 1: Construct a high-quality initial solution
    # We use a greedy algorithm but with multiple random restarts and a "structured" bias.
    # To introduce structure, we can prioritize triples that form arithmetic progressions or lines in Z_4xZ_4.
    
    best_triples = []
    best_alpha = n
    
    # Generate "structured" triples:
    # Lines in AG(2,4) have 4 points. We can split them into two triples? No, that creates repeated pairs within the line?
    # If line is {a,b,c,d}, triples {a,b,c} and {a,d,b}? Pair (a,b) repeated.
    # So we can't just split lines.
    
    # Instead, let's rely on a robust randomized local search.
    # The key to beating the seed (alpha=11) is to get alpha down to ~6 or 7.
    # A random greedy packing usually achieves alpha ~ 7-8.
    # To improve, we need to optimize the structure.
    
    # Let's try a specific heuristic:
    # 1. Start with an empty system.
    # 2. Add triples greedily, but choose the next triple to maximize the number of "future" options?
    #    Or minimize the "clique number" of the complement?
    
    # Alternative: Use a known dense packing.
    # Max number of triples in a linear 3-uniform hypergraph on 16 vertices.
    # Upper bound: floor(16/3 * 15/2 / 3) = floor(120/3) = 40.
    # But STS(16) doesn't exist.
    # A 2-(16,3,1) packing has at most 16*15/(3*2) = 40 blocks? No, each block covers 3 pairs. Total pairs 120.
    # So max 40 blocks.
    # If we can get close to 40, alpha will be small.
    
    # Let's run a simulated annealing or hill climbing.
    # State: set of triples.
    # Move: Remove one triple, add one different triple.
    # Objective: Minimize alpha.
    
    # Since alpha calculation is expensive (65536 * T), we can't do too many iterations.
    # But 65536 * 40 = 2.6M ops. In Python, ~0.1-0.2s.
    # We can do ~500 iterations in 100s.
    
    # Initial Solution:
    # Greedy with random restarts.
    
    def get_initial_solution():
        triples = []
        used_pairs = set()
        remaining = all_triples[:]
        random.shuffle(remaining)
        
        # To make it denser, we can try to fill "gaps" intelligently.
        # But simple random greedy is a good baseline.
        for t in remaining:
            if is_valid_addition(triples, used_pairs, t):
                triples.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        return triples

    # Local Search
    def local_search(initial_triples, time_limit):
        triples = initial_triples[:]
        used_pairs = set()
        for t in triples:
            for p in triple_pairs[t]:
                used_pairs.add(p)
        
        current_alpha = calc_alpha(triples)
        best_alpha_local = current_alpha
        best_triples_local = triples[:]
        
        # Pre-compute all triples that can be added given current used_pairs?
        # This is dynamic.
        
        # Move: Swap
        # Pick a triple T1 to remove.
        # Pick a triple T2 to add (must be valid after removing T1).
        
        # Optimization:
        # When we remove T1, 3 pairs become available.
        # T2 must use at least one of these pairs? No, T2 must be valid in the new state.
        # T2 is valid if none of its pairs are in (used_pairs - pairs(T1)).
        
        # To speed up, we can pre-compute for each triple T2, which triples T1 it "conflicts" with?
        # T2 conflicts with T1 if they share a pair.
        # If T2 conflicts with T1, we cannot add T2 after removing T1 unless T2 uses only pairs freed by T1?
        # No, if T2 and T1 share a pair, T2 uses that pair. If we remove T1, that pair is free.
        # But T2 might use other pairs that are still used by other triples.
        
        # Let's just iterate.
        
        # To make this faster, we can limit the number of swap attempts.
        
        start_ls = time.time()
        stuck_count = 0
        
        while time.time() < start_ls + time_limit:
            if not triples:
                break
                
            # Pick a random triple to remove
            idx = random.randint(0, len(triples) - 1)
            t_remove = triples[idx]
            
            # Temporarily remove
            triples.pop(idx)
            freed_pairs = []
            for p in triple_pairs[t_remove]:
                used_pairs.remove(p)
                freed_pairs.append(p)
            
            # Find candidates for addition
            # A candidate T_add must be valid: none of its pairs in used_pairs.
            # We can iterate through all_triples? 560 is small.
            # But we can filter: T_add must use at least one freed pair?
            # Not necessarily. It could use only pairs that were already free?
            # No, if it used only already free pairs, it would have been added before.
            # So T_add MUST use at least one freed pair.
            
            candidates = []
            for t_add in all_triples:
                if t_add == t_remove:
                    continue
                # Check if t_add uses any freed pair
                uses_freed = False
                for p in triple_pairs[t_add]:
                    if p in freed_pairs:
                        uses_freed = True
                        break
                if not uses_freed:
                    continue
                
                # Check if t_add is valid in current state
                valid = True
                for p in triple_pairs[t_add]:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    candidates.append(t_add)
            
            if not candidates:
                # Revert
                triples.append(t_remove)
                for p in freed_pairs:
                    used_pairs.add(p)
                continue
            
            # Pick a random candidate
            t_add = random.choice(candidates)
            
            # Apply addition
            triples.append(t_add)
            for p in triple_pairs[t_add]:
                used_pairs.add(p)
            
            # Calculate new alpha
            new_alpha = calc_alpha(triples)
            
            if new_alpha < best_alpha_local:
                best_alpha_local = new_alpha
                best_triples_local = triples[:]
                stuck_count = 0
            else:
                stuck_count += 1
                # With some probability, accept worse solution to escape local minima?
                # Simple hill climbing is often enough for this size.
                # But let's try a random restart of the local search if stuck too long.
                if stuck_count > 50:
                    # Revert to best
                    triples = best_triples_local[:]
                    used_pairs = set()
                    for t in triples:
                        for p in triple_pairs[t]:
                            used_pairs.add(p)
                    stuck_count = 0
                    
            if time.time() > deadline:
                break
                
        return best_triples_local, best_alpha_local

    # Main Loop
    # We will run several independent local searches starting from different greedy initializations.
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts if time.time() < deadline else 0
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Get initial solution
        init_triples = get_initial_solution()
        
        # Run local search
        ls_triples, ls_alpha = local_search(init_triples, time_per_restart)
        
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples[:]
            
            # If we reach alpha 5 or lower, it's very good.
            # Theoretical min for n=16?
            # If alpha=5, score = 11/5 = 2.2.
            # If alpha=6, score = 11/6 = 1.83.
            # Current best is 1.375 -> alpha = 11/1.375 = 8.
            # So we need to get alpha to 7 or lower to beat 1.375.
            # alpha=7 -> score = 11/7 = 1.57.
            # alpha=6 -> score = 1.83.
            
    # Final check: ensure best_triples is valid
    if not best_triples:
        # Fallback to a simple valid system if something went wrong
        # E.g., 5 disjoint triples
        best_triples = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]]
        # Alpha for this: 2 from each of 5 triples = 10, plus point 15 = 11.
        # Score 1.0.
        
    return best_triples

import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    # 1. Precompute GCD table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    # 2. Identify all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == gab and gt[b][c] == gab:
                    bad_triples.append((a, b, c))

    # Map element -> list of bad triple indices it participates in
    elem_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    # 3. Structural Change: Integer Programming / Branch & Bound or Greedy with Stronger Heuristics
    # The previous approach used a greedy hitting set which is myopic.
    # Let's try a different strategy:
    # Start with a "core" of elements that are mutually compatible (no bad triples among them)
    # and then greedily add elements that don't conflict.
    # Or better: Randomized Local Search with a better initial solution.
    # Let's try Simulated Annealing on the set membership, but with a faster evaluation.
    
    # To make SA fast, we need to quickly check if adding/removing an element creates/destroys bad triples.
    # Precompute for each element x, the set of pairs (a,b) such that (a,b,x) is a bad triple.
    # This is essentially the neighborhood in the hypergraph.
    
    # Let's define a function to check validity of a set quickly.
    # But even better, let's define a function to find all bad triples *within* a given set.
    # If the set is small (e.g., < 60), checking all triples in it is O(|S|^3) which might be slow if done many times.
    # However, we can precompute a 3D boolean array bad[a][b][c] = True if (a,b,c) is bad.
    # With N=128, this is 128^3 ~ 2M booleans. 2MB. Very feasible.
    
    bad_3d = [[[False] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for a, b, c in bad_triples:
        bad_3d[a][b][c] = True
        bad_3d[a][c][b] = True
        bad_3d[b][a][c] = True
        bad_3d[b][c][a] = True
        bad_3d[c][a][b] = True
        bad_3d[c][b][a] = True

    def count_bad_triples_in_set(S):
        """Count number of bad triples fully contained in S."""
        count = 0
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if bad_3d[a][b][c]:
                        count += 1
        return count

    def try_add_element(S, x):
        """Check if adding x to S creates any new bad triples."""
        # A new bad triple must include x and two elements from S.
        for a in S:
            for b in S:
                if a == b:
                    continue
                if bad_3d[x][a][b]:
                    return False
        return True

    def try_remove_element(S, x):
        """Always valid to remove."""
        return True

    # Strategy:
    # 1. Generate a random initial set.
    # 2. Use Simulated Annealing to maximize |S| subject to 0 bad triples.
    #    State: bitmask or set of elements.
    #    Move: Add a random element, Remove a random element, or Swap.
    #    Cost: Number of bad triples. We want Cost = 0.
    #    Objective: Maximize |S| among states with Cost = 0.
    #    Or: Minimize (BadTriples * LargeNumber - |S|).
    
    # Let's use a penalty approach.
    # Score(S) = |S| * 10000 - count_bad_triples_in_set(S)
    # We want to maximize this.
    
    # To speed up count_bad_triples_in_set, we can maintain the count incrementally.
    # But for simplicity and robustness, let's first see if a simple Greedy + Polish works better than the previous hitting set.
    
    # Actually, the previous program scored 1.75. The inspiration 1 scored 2.0.
    # Inspiration 1 mentioned "greedy approach with multiple restarts".
    # Let's try a more aggressive Greedy with randomization.
    
    def greedy_construct(seed):
        random.seed(seed)
        S = set()
        # Order elements by some heuristic?
        # Elements that are in fewer bad triples are "safer" to include.
        # Let's compute static degree (number of bad triples an element is in).
        static_degree = [0] * (N + 1)
        for a, b, c in bad_triples:
            static_degree[a] += 1
            static_degree[b] += 1
            static_degree[c] += 1
        
        # Shuffle elements, but bias towards low degree?
        # Or just random order.
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        for x in elements:
            if try_add_element(S, x):
                S.add(x)
        
        # Polish: Try to add any remaining element
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S:
                    if try_add_element(S, x):
                        S.add(x)
                        changed = True
        
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:  # Leave 10s for local search on best
            break
        S = greedy_construct(i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # print(f"Restart {i}: size {len(S)}")

    # Local Search on the best set found
    # We will perform swaps to try to increase size while maintaining validity.
    # Move: Remove x, Add y. If valid and size increases (or stays same but allows future gains), accept.
    # Since we always keep size same in a swap, we need a "grow" move: Add y (if valid).
    # If no grow move, try swap: Remove x, Add y1, Add y2 (2-for-1).
    
    def local_search(S, max_iters=200):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try to add a single element
            for x in range(1, N + 1):
                if x not in S:
                    if try_add_element(S, x):
                        S.add(x)
                        improved = True
                        break
                        
            # 2. Try 1-for-1 swap (remove x, add y)
            # This doesn't increase size, but might allow a subsequent add.
            # Or we can try 1-for-2 (remove x, add y1, y2).
            
            if not improved:
                # Try 1-for-2 exchange
                removed = None
                added = []
                # Pick a random element to remove
                # To be deterministic-ish, iterate
                for x in list(S):
                    temp_S = S - {x}
                    # Try to add as many elements as possible from the complement
                    candidates = [y for y in range(1, N + 1) if y not in temp_S]
                    random.shuffle(candidates)
                    
                    # Greedily add up to 2 elements
                    added_count = 0
                    new_S = set(temp_S)
                    for y in candidates:
                        if added_count >= 2:
                            break
                        if try_add_element(new_S, y):
                            new_S.add(y)
                            added_count += 1
                            
                    if len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                    elif len(new_S) == len(S):
                        # Keep size same, maybe accept with probability? 
                        # Or just store as candidate if it unlocks future adds.
                        # For now, let's only accept if size increases or if it's a "swap" that seems beneficial.
                        # Let's stick to size increase for simplicity and speed.
                        pass
                
        return S

    # Run local search on the best greedy result
    best_set = local_search(best_set)
    
    # Also run local search on a few other good greedy results
    # Store top 5 greedy results
    # (Simplified: just re-run greedy a few more times and polish)
    
    for i in range(num_restarts, num_restarts + 20):
        if time.time() > deadline - 5:
            break
        S = greedy_construct(i)
        S = local_search(S, max_iters=50)
        if len(S) > len(best_set):
            best_set = S

    # Final polish
    final_set = local_search(best_set, max_iters=100)
    if len(final_set) > len(best_set):
        best_set = final_set

    if not best_set:
        # Fallback
        best_set = {1, 2, 4, 8, 16, 32, 64, 128}

    return sorted(best_set)

# EVOLVE-BLOCK-END

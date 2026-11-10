"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    num_idx = {num: i for i, num in enumerate(numbers)}
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples and build conflict structure
    # We represent the set of selected numbers as a Python int bitmask for fast operations.
    # Bit i represents number i+1.
    # conflict_mask[x] is a bitmask of numbers y such that x and y are part of a bad triple?
    # No, that's not enough. We need to know if adding x creates a bad triple with ANY pair in S.
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    # Let's precompute for each pair (a,b), the set of c such that (a,b,c) is a bad triple.
    # pair_conflicts[(a,b)] = bitmask of c.
    # Then, to check if x can be added to S:
    # For every pair (u, v) in S, check if x is in pair_conflicts[(u,v)].
    # This is O(|S|^2) per check, which is slow for large S.
    
    # Alternative representation:
    # For each x, let's precompute a list of "forbidden pairs".
    # forbidden_pairs[x] = list of (u, v) such that {x, u, v} is a bad triple.
    # To check if x can be added to S:
    # Iterate through forbidden_pairs[x]. If both u and v are in S, it's invalid.
    # This is O(k) where k is the number of forbidden pairs for x.
    
    forbidden_pairs = {i: [] for i in range(1, n + 1)}
    
    # Precompute bad triples
    # We can optimize this by grouping by LCM value?
    # If lcm(a,b) = L, then c must also have lcm(a,c)=L and lcm(b,c)=L.
    # Let's just iterate all triples for simplicity first, n=128 is small enough for precomputation.
    # 128^3 / 6 ~ 350k checks. Very fast.
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    forbidden_pairs[a].append((b, c))
                    forbidden_pairs[b].append((a, c))
                    forbidden_pairs[c].append((a, b))
                    
    # Convert forbidden_pairs to bitmasks for faster "both in S" check?
    # If we have S as a bitmask, checking "u in S and v in S" is (S >> (u-1)) & 1 and (S >> (v-1)) & 1.
    # This is fast.
    
    # Initial Greedy Solution to get a lower bound
    # Order by number of forbidden pairs (ascending)
    conflict_counts = {i: len(forbidden_pairs[i]) for i in range(1, n + 1)}
    greedy_order = sorted(numbers, key=lambda x: conflict_counts[x])
    
    current_set_mask = 0
    current_set = set()
    
    def can_add(x, mask):
        for u, v in forbidden_pairs[x]:
            if ((mask >> (u-1)) & 1) and ((mask >> (v-1)) & 1):
                return False
        return True
        
    for x in greedy_order:
        if can_add(x, current_set_mask):
            current_set_mask |= (1 << (x-1))
            current_set.add(x)
            
    best_size = len(current_set)
    best_mask = current_set_mask
    
    if time.time() > deadline - 10:
        return sorted([i+1 for i in range(n) if (best_mask >> i) & 1])

    # Backtracking Search with Pruning
    # State: current_mask, current_size, available_numbers
    # We want to maximize size.
    
    # To make backtracking effective, we need good variable ordering.
    # Static ordering based on conflict count is a start.
    # Dynamic ordering (MRV) is better but expensive to maintain.
    
    # Let's use a recursive function that tries to add more elements.
    # We will iterate through candidates in a specific order.
    
    # Optimization: Precompute for each x, the list of forbidden pairs as bitmasks?
    # No, we need to check pairs.
    
    # Let's try a simple backtracking that explores adding elements in a fixed order,
    # but uses pruning: if current_size + count(available) <= best_size, prune.
    
    # This is essentially finding a maximal independent set.
    
    # A better approach: Branch and Bound on the hypergraph.
    
    def backtrack(current_mask, current_size, candidates):
        nonlocal best_size, best_mask, deadline, start_time
        
        if time.time() > deadline - 2:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        if not candidates:
            return
            
        # Pruning: Even if we add all remaining candidates, can we beat best_size?
        if current_size + len(candidates) <= best_size:
            return
            
        # Pick a candidate to branch on.
        # Heuristic: Pick the candidate with the most conflicts with other candidates?
        # Or just pick the first one?
        # Let's pick the one with the smallest index for determinism, 
        # but we can also try the one with highest "potential".
        
        # For speed, let's just iterate through candidates in the pre-sorted order (greedy_order).
        # This might be slow if the branching factor is high.
        
        # Let's try a different strategy: 
        # Instead of branching on "add x" or "skip x", 
        # we can use a greedy fill at the end of each branch?
        
        # Actually, a standard Maximum Independent Set algorithm on a graph is NP-hard.
        # Here we have a hypergraph.
        
        # Let's use a simple recursive search that tries to add elements one by one.
        # We will maintain a list of "available" elements.
        
        # To speed up, we can use bit operations.
        # Let's represent candidates as a bitmask too.
        
        # Re-implementing with bitmasks for candidates might be complex.
        # Let's stick to sets/lists but optimize the inner loop.
        
        # Pick the first candidate in the list (which is sorted by conflict count)
        x = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include x
        if can_add(x, current_mask):
            new_mask = current_mask | (1 << (x-1))
            # Filter rest: remove any y that conflicts with x
            # A y conflicts with x if there exists z in current_set such that {x,y,z} is bad?
            # No, y conflicts with x if adding y to {current_set, x} is invalid.
            # This is not just a pairwise conflict.
            # So we cannot simply remove y from rest if x is added.
            # We must check can_add(y, new_mask) when we recurse.
            
            backtrack(new_mask, current_size + 1, rest)
            
        # Option 2: Exclude x
        backtrack(current_mask, current_size, rest)

    # The above backtracking is O(2^N) in worst case. N=128 is too big.
    # We need a much smarter approach.
    
    # Let's go back to Local Search but with a stronger move: 
    # "Swap" moves: Remove one, Add one.
    # And "Double Swap": Remove two, Add two.
    
    # The previous program did "Remove 1, Greedy Fill".
    # Let's try "Simulated Annealing" on the space of valid sets.
    # State: Valid set S.
    # Energy: -|S| (we want to minimize energy, i.e., maximize size).
    # Moves:
    # 1. Add x if valid. (Always accept, size increases)
    # 2. Remove x. (Size decreases. Accept with prob exp(delta/T))
    # 3. Swap x (in S) with y (not in S). If (S\{x}) U {y} is valid.
    #    Delta = 0. Accept with prob 1? Or random?
    
    # Since we want to maximize size, we should prioritize Add moves.
    # If no Add moves are possible, we are at a local maximum.
    # We must then perform Swaps or Removals to escape.
    
    # Let's implement a robust Local Search:
    # 1. Start with greedy solution.
    # 2. While time remains:
    #    a. Try all possible Add moves. If any exist, take the best one (or random).
    #    b. If no Add moves, try Swap moves.
    #       For each x in S, for each y not in S:
    #          Check if (S - {x}) U {y} is valid.
    #          If valid, this is a neutral move (size same).
    #          We want to find a sequence of neutral moves that leads to an Add move.
    #          This is hard to plan.
    #    c. Try Random Removals. Remove k random elements, then Greedy Fill.
    
    # The previous program did (c) effectively.
    # Let's try to improve the Greedy Fill by using a better ordering.
    # Instead of static conflict count, use dynamic conflict count relative to current set.
    
    def greedy_fill(current_mask, current_set):
        # Order candidates by how many conflicts they have WITH THE CURRENT SET
        # Actually, we just want to add as many as possible.
        # A good heuristic: Add the element that "blocks" the fewest other elements?
        # Or the element that has the fewest conflicts in the remaining pool?
        
        # Let's try: Sort all not-in-set elements by (number of conflicts with current_set, conflict_count_global)
        # But computing conflicts with current_set is expensive.
        
        # Let's just use the global conflict count order, but skip invalid ones.
        # To break ties or improve, we can randomize the order slightly.
        
        added = 0
        for x in greedy_order:
            if x in current_set:
                continue
            if can_add(x, current_mask):
                current_mask |= (1 << (x-1))
                current_set.add(x)
                added += 1
        return current_mask, current_set, added

    # Let's run a loop of "Perturb and Fill"
    # Perturb: Remove k random elements.
    # Fill: Greedy add.
    
    # To make it better, let's try different k values and different random seeds.
    
    # Also, let's try a "Hill Climbing" on the number of elements we can add.
    
    # Let's try to improve the initial greedy by trying multiple random orderings.
    
    best_mask = current_set_mask
    best_size = len(current_set)
    
    # Multiple random restarts of greedy
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        random.shuffle(greedy_order)
        mask = 0
        s = set()
        for x in greedy_order:
            if can_add(x, mask):
                mask |= (1 << (x-1))
                s.add(x)
        if len(s) > best_size:
            best_size = len(s)
            best_mask = mask
    
    # Now, local search from the best found
    current_mask = best_mask
    current_set = set([i+1 for i in range(n) if (current_mask >> i) & 1])
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try removing each element and refilling
        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            
            trial_mask = current_mask & ~(1 << (x-1))
            trial_set = set(current_set)
            trial_set.remove(x)
            
            # Greedy fill
            for y in greedy_order:
                if y in trial_set:
                    continue
                if can_add(y, trial_mask):
                    trial_mask |= (1 << (y-1))
                    trial_set.add(y)
            
            if len(trial_set) > len(current_set):
                current_mask = trial_mask
                current_set = trial_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_mask = current_mask
                improved = True
                
    # One more phase: Random swaps
    if time.time() < deadline - 2:
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            if not current_set:
                break
            # Pick random x in S, random y not in S
            x = random.choice(list(current_set))
            # Find y not in S
            not_in_s = [i for i in range(1, n+1) if i not in current_set]
            if not not_in_s:
                break
            y = random.choice(not_in_s)
            
            # Check if swap is valid
            trial_mask = (current_mask & ~(1 << (x-1))) | (1 << (y-1))
            
            # We need to check if trial_set is valid.
            # Since we only swapped one element, we only need to check if y creates a bad triple with pairs in (S - {x}).
            # And that x wasn't part of a bad triple that is now broken (which is fine, we remove constraints).
            # So we just check if y is valid in (S - {x}).
            
            valid = True
            for u, v in forbidden_pairs[y]:
                if ((trial_mask >> (u-1)) & 1) and ((trial_mask >> (v-1)) & 1):
                    # Check if u or v is x? 
                    # If u=x or v=x, they are not in trial_mask, so the check passes.
                    # So this correctly checks validity against the new set.
                    valid = False
                    break
            
            if valid:
                # Size is same, so this doesn't improve size directly.
                # But it might allow future improvements.
                # However, since we are maximizing size, and we are already at a local max for additions,
                # a neutral swap might lead to a state where an addition is possible.
                # Let's perform the swap and then try to add.
                current_mask = trial_mask
                current_set.remove(x)
                current_set.add(y)
                
                # Try to add more
                for z in greedy_order:
                    if z in current_set:
                        continue
                    if can_add(z, current_mask):
                        current_mask |= (1 << (z-1))
                        current_set.add(z)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_mask = current_mask
                        break # Restart add loop? Or just continue?
                # If we added something, we improved.
                # If not, we just moved.
                
    return sorted([i+1 for i in range(n) if (best_mask >> i) & 1])

# EVOLVE-BLOCK-END

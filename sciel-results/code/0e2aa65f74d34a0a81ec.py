import time
from math import gcd
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]

    # 2. Identify all "bad triples"
    # We need to know, for any pair (a, b), which c's make it bad?
    # Or just list all bad triples.
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    T = len(bad_triples)
    
    # Precompute conflict counts: how many bad triples each element is involved in
    conflict_count = [0] * (N + 1)
    for a, b, c in bad_triples:
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
        
    # Helper to check if adding x to set S creates a bad triple
    # S is a set of ints.
    # A bad triple is (x, y, z) where y, z in S.
    # We can precompute for each x, the list of pairs (y, z) in bad_triples containing x.
    # pairs_for_x[x] = list of (y, z) such that {x, y, z} is a bad triple.
    pairs_for_x = [[] for _ in range(N + 1)]
    for a, b, c in bad_triples:
        pairs_for_x[a].append((b, c))
        pairs_for_x[b].append((a, c))
        pairs_for_x[c].append((a, b))
        
    def is_valid_addition(x, S):
        """Check if x can be added to S without creating a bad triple."""
        # For each pair (y, z) in pairs_for_x[x], check if both y and z are in S.
        for y, z in pairs_for_x[x]:
            if y in S and z in S:
                return False
        return True
        
    # 3. Greedy Forward Construction
    # Sort elements by conflict_count (ascending). Low conflict first.
    # Ties broken by value (deterministic) or random.
    # Let's try sorting by (conflict_count, value)
    sorted_elements = sorted(elements, key=lambda x: (conflict_count[x], x))
    
    S = set()
    for x in sorted_elements:
        if is_valid_addition(x, S):
            S.add(x)
            
    current_size = len(S)
    best_S = set(S)
    best_size = current_size
    
    # 4. Local Search: Conflict-Driven Swaps
    # We want to replace elements in S with elements not in S.
    # We prioritize removing elements in S that have HIGH conflict_count 
    # and adding elements not in S that have LOW conflict_count.
    
    # To make this efficient, we maintain lists.
    # in_S = list of elements in S
    # not_in_S = list of elements not in S
    
    # We will iterate through potential swaps.
    # To avoid O(N^2) per step, we can just do a few passes.
    
    # Precompute: for each x, the "cost" of x is its conflict_count.
    # We want to minimize the sum of costs in S? No, we want to maximize size.
    # But high conflict elements block many additions.
    # So, swapping a high-conflict element out for a low-conflict one might allow 
    # *other* low-conflict elements to be added later.
    
    # Strategy:
    # 1. Identify "expensive" elements in S (high conflict).
    # 2. Identify "cheap" elements outside S (low conflict).
    # 3. Try swapping expensive_out with cheap_in.
    # 4. If valid, accept.
    
    # To be more effective, we can also try to add cheap elements directly if they fit.
    
    # Let's run a loop of improvements.
    # We'll use a simple randomized restart or just a single long greedy pass.
    # Given the small N=128, we can afford a decent amount of search.
    
    # Let's implement a "Hill Climbing" on the set size.
    # Move: Add x (if valid).
    # Move: Remove x, Add y (if valid and size increases? No, size stays same. 
    #       But maybe removing x allows adding z later? 
    #       Actually, if we just want to maximize size, we should only accept 
    #       moves that increase size, OR moves that keep size same but 
    #       "prepare" for future increases.
    #       Standard local search for Max IS: 
    #       1. Try to add any valid node.
    #       2. If none, try to swap u in S with v not in S. 
    #          Accept if the new set allows adding MORE nodes later? 
    #          That's lookahead.
    #       Simpler: Accept swap if it doesn't decrease size (i.e., size stays same) 
    #          and the "potential" of the new set is higher?
    #          Potential = number of nodes not in S that are "almost" valid?
    
    # Let's stick to a simpler, robust heuristic:
    # 1. Try to add any node not in S. If successful, update S.
    # 2. If no node can be added, try to swap a node u in S with v not in S.
    #    Condition for swap: 
    #    - The set (S - {u} + {v}) must be valid.
    #    - We accept the swap if it allows us to add *at least one more* node 
    #      that was previously blocked? 
    #    - Or, simpler: Accept swap if conflict_count[v] < conflict_count[u].
    #      This reduces the "average conflict" of the set, potentially freeing up space.
    
    # Let's combine:
    # Phase A: Greedy Add.
    # Phase B: Swap to reduce conflict profile.
    
    # We already did Greedy Add in step 3.
    # Now Phase B:
    
    # To make swaps effective, we need to know if a swap *enables* further growth.
    # We can estimate "freedom" as the number of nodes not in S that are valid 
    # with respect to S.
    
    def count_valid_additions(S_set):
        count = 0
        for x in elements:
            if x not in S_set:
                if is_valid_addition(x, S_set):
                    count += 1
        return count
        
    # Current freedom
    current_freedom = count_valid_additions(S)
    
    # If freedom is 0, we are stuck. We must swap.
    # If freedom > 0, we should add.
    
    # Let's do a loop:
    # While time remains:
    #   1. Try to add a valid node. (Pick one with lowest conflict)
    #   2. If no valid node, try swaps.
    #      - Pick u in S with highest conflict.
    #      - Pick v not in S with lowest conflict.
    #      - If S - {u} + {v} is valid:
    #          - Check if new freedom > current freedom OR (new freedom == current freedom and conflict_sum decreases)
    #          - If yes, accept.
    
    # To speed up "is_valid_addition", we use the precomputed pairs.
    
    # Let's run this loop.
    
    # To avoid getting stuck in a local optimum where freedom is 0 but a swap 
    # doesn't immediately increase freedom (it might take 2 swaps), we can 
    # allow swaps that maintain freedom if they reduce the "sum of conflicts" in S.
    
    sum_conflicts = sum(conflict_count[x] for x in S)
    
    while time.time() < deadline - 5.0:
        # 1. Try to add
        added = False
        # Find valid additions
        valid_adds = []
        for x in elements:
            if x not in S:
                if is_valid_addition(x, S):
                    valid_adds.append(x)
                    
        if valid_adds:
            # Pick the one with lowest conflict
            valid_adds.sort(key=lambda x: conflict_count[x])
            x_add = valid_adds[0]
            S.add(x_add)
            sum_conflicts += conflict_count[x_add]
            added = True
            # Update best
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                
        if not added:
            # 2. Try swaps
            # We need to find u in S, v not in S such that S' = S - {u} + {v} is valid.
            # And S' is "better".
            
            # To be efficient, we don't check all pairs.
            # We prioritize high-conflict u and low-conflict v.
            
            # Get sorted lists
            in_S_sorted = sorted([x for x in S], key=lambda x: -conflict_count[x])
            not_in_S_sorted = sorted([x for x in elements if x not in S], key=lambda x: conflict_count[x])
            
            swap_made = False
            # Try a limited number of swaps per iteration to avoid O(N^2)
            # N=128, so N^2 is 16k. That's fast.
            # But is_valid_addition is O(degree).
            
            # Let's try all pairs? 16k checks * ~10 ops = 160k ops. Fast.
            # We will find the "best" swap based on a heuristic score.
            
            best_swap = None
            best_swap_score = -1
            current_freedom = count_valid_additions(S)
            
            # We calculate freedom for the original S once.
            # For each swap, we calculate new freedom.
            # This is expensive: O(N^2 * N) = 2M ops. Might be slow in Python.
            # 128^3 = 2M. 2M * 50ns = 0.1s. Acceptable for a few iterations.
            # But we only do this if no adds are possible.
            
            # Optimization: Only check swaps where u is "high" and v is "low".
            # Let's check top 20 u's and bottom 20 v's.
            candidates_u = in_S_sorted[:20]
            candidates_v = not_in_S_sorted[:20]
            
            for u in candidates_u:
                for v in candidates_v:
                    if u == v: continue
                    # Check if S - {u} + {v} is valid
                    # This is equivalent to checking if v is valid in S - {u}.
                    # is_valid_addition(v, S - {u})
                    
                    # We can optimize this check.
                    # v is invalid in S iff there exists (y, z) in pairs_for_x[v] 
                    # such that y, z in S.
                    # In S - {u}, u is removed. So if the only blocking pair 
                    # involving u is removed, v might become valid.
                    
                    # Simple check:
                    temp_S = S.copy()
                    temp_S.remove(u)
                    temp_S.add(v)
                    
                    # Check validity of temp_S? 
                    # No, just check if v is valid in temp_S (which is S - {u}).
                    # But we also need to ensure that removing u didn't break validity 
                    # of other elements? 
                    # Removing an element NEVER creates a bad triple. 
                    # So if S was valid, S - {u} is valid.
                    # We just need to check if v can be added to S - {u}.
                    
                    if is_valid_addition(v, temp_S):
                        # Valid swap.
                        # Score: Does it increase freedom?
                        # Freedom = count of x not in temp_S that are valid in temp_S.
                        # This is expensive to calculate fully.
                        
                        # Heuristic Score:
                        # 1. If len(S) increases? No, size is same.
                        # 2. If sum_conflicts decreases?
                        new_sum = sum_conflicts - conflict_count[u] + conflict_count[v]
                        if new_sum < sum_conflicts:
                            # Better conflict profile.
                            # Let's accept this swap?
                            # Risk: It might reduce freedom.
                            # Let's calculate freedom for a few promising swaps.
                            
                            # To save time, we just accept the first swap that 
                            # significantly reduces conflict, OR we do a quick 
                            # freedom check.
                            
                            # Let's do a quick freedom check for this specific swap.
                            # Freedom in temp_S:
                            # Count x not in temp_S such that is_valid_addition(x, temp_S)
                            # This is O(N * deg).
                            
                            # Since we are in the "stuck" phase, we can afford this.
                            new_freedom = 0
                            for x in elements:
                                if x not in temp_S:
                                    if is_valid_addition(x, temp_S):
                                        new_freedom += 1
                                        
                            # Accept if new_freedom > current_freedom 
                            # OR (new_freedom == current_freedom and new_sum < sum_conflicts)
                            if new_freedom > current_freedom or (new_freedom == current_freedom and new_sum < sum_conflicts):
                                # Accept
                                S = temp_S
                                sum_conflicts = new_sum
                                current_freedom = new_freedom
                                swap_made = True
                                break
                if swap_made:
                    break
            
            if not swap_made:
                # No good swap found. 
                # We might be in a local optimum.
                # To escape, we can do a random "kick":
                # Remove a random element from S and add a random element not in S, 
                # even if it reduces freedom/conflict, to explore.
                # But this is risky.
                
                # Alternative: Restart with a different greedy order?
                # Given the time limit, let's just break if we've been stuck for a while?
                # Or just stop. The greedy add + swap is usually strong for N=128.
                
                # Let's try a few random swaps regardless of score to escape local minima.
                if random.random() < 0.1:
                    # Random swap
                    u = random.choice(list(S))
                    v = random.choice([x for x in elements if x not in S])
                    temp_S = S.copy()
                    temp_S.remove(u)
                    temp_S.add(v)
                    if is_valid_addition(v, temp_S):
                        S = temp_S
                        sum_conflicts = sum(conflict_count[x] for x in S)
                        # Don't update best yet, see if it grows
                else:
                    # Stuck, break to save time?
                    # Or just continue looping to check time.
                    pass

    # Final verification and return
    # Ensure best_S is valid
    # (The logic ensures S is always valid, and best_S is a copy of a valid S)
    
    # Return the largest valid set found
    return list(best_S)

# EVOLVE-BLOCK-END

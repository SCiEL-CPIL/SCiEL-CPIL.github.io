import time
import math
import random
import sys
import heapq

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
        
    best_set = set()
    best_size = 0
    
    def greedy_removal(random_seed):
        """
        Greedy removal using a priority queue to efficiently find the element
        with the maximum number of active triples (conflicts).
        """
        random.seed(random_seed)
        S = set(range(1, N + 1))
        counts = [3] * T  # Initially all triples are fully in S
        
        # active_count[x] = number of triples t containing x such that counts[t] == 3
        active_count = [0] * (N + 1)
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        # Heap stores (-active_count, element) for max-heap behavior
        # We need to handle stale entries in the heap
        heap = []
        for x in range(1, N + 1):
            heapq.heappush(heap, (-active_count[x], x))
            
        active_triples_remaining = T
        
        while active_triples_remaining > 0:
            if time.time() > deadline - 5.0:
                return S
                
            # Pop valid max conflict element
            x_to_remove = None
            while heap:
                neg_ac, x = heapq.heappop(heap)
                ac = -neg_ac
                if x in S and active_count[x] == ac:
                    x_to_remove = x
                    break
                # else stale entry, discard
            
            if x_to_remove is None:
                break # No more valid elements with conflicts? Should imply no active triples
            
            S.remove(x_to_remove)
            
            # Update counts and active_count for triples containing x_to_remove
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    # This triple was active, now it's not
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                    active_triples_remaining -= 1
                    
                    # Push updated active_counts for a, b, c
                    # We push them even if they are 0 to maintain heap consistency if needed,
                    # but we check validity on pop.
                    heapq.heappush(heap, (-active_count[a], a))
                    heapq.heappush(heap, (-active_count[b], b))
                    heapq.heappush(heap, (-active_count[c], c))
                elif counts[t_idx] == 2:
                    # Was already inactive (2 in S), now 1 in S
                    counts[t_idx] = 1
                # else counts[t_idx] == 1, now 0. No active count change.
                
        return S

    def local_search(S, random_seed):
        """
        1. Try to add back removed elements (single addition).
        2. Try 2-opt swaps: remove x from S, add y,z from removed.
        """
        random.seed(random_seed)
        S = set(S)
        
        # Rebuild counts for current S
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
                
        removed = [x for x in range(1, N + 1) if x not in S]
        
        # Phase 1: Simple additions
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            # Shuffle removed to randomize order
            random.shuffle(removed)
            for x in removed:
                if x in S:
                    continue
                # Check if x can be added
                can_add = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                    # Remove from removed list? We iterate over a copy or check membership.
                    # Since we check `if x in S`, it's safe.
        
        # Phase 2: 2-opt swaps
        # Try to remove one element x from S and add two elements y, z from removed.
        # This increases size by 1.
        
        # Precompute potential pairs? Or just iterate.
        # For each x in S:
        #   Temporarily remove x.
        #   Try to add all possible elements from removed.
        #   If we can add >= 2, accept the best combination (or just first found).
        #   If not, restore x.
        
        # To be efficient, we only try this for a limited number of iterations or if stuck.
        
        # Let's do a few passes of this.
        for _ in range(5): # Limit passes for time
            if time.time() > deadline - 5.0:
                break
            
            # List of current S elements to try removing
            S_list = list(S)
            random.shuffle(S_list)
            
            for x in S_list:
                if time.time() > deadline - 5.0:
                    break
                    
                # Temporarily remove x
                S.remove(x)
                # Update counts
                for t_idx in elem_triples[x]:
                    counts[t_idx] -= 1
                
                # Try to add elements from removed
                added = []
                # We need to find 2 elements that can be added simultaneously?
                # Or just greedily add as many as possible.
                # Greedy addition:
                # Iterate through removed, if can add, add it.
                # But adding one might block another?
                # Yes. So we should try to find a pair that works.
                
                # Simpler: Just greedily add. If we add 2, great.
                # But greedy might add 1 that blocks the second.
                # Given small N, we can try to be smarter.
                
                # Let's just try to add all possible.
                # We need to track which ones we added to revert if we don't get enough.
                
                potential_adds = []
                for y in removed:
                    if y in S:
                        continue
                    # Check if y can be added given current counts (with x removed)
                    can_add = True
                    for t_idx in elem_triples[y]:
                        if counts[t_idx] >= 2:
                            can_add = False
                            break
                    if can_add:
                        potential_adds.append(y)
                        
                # If we have at least 2 potential adds, can we add both?
                # They might conflict with each other.
                # Let's try to pick 2 non-conflicting ones.
                
                found_pair = False
                if len(potential_adds) >= 2:
                    # Try all pairs? O(K^2). K is small usually.
                    for i in range(len(potential_adds)):
                        y1 = potential_adds[i]
                        # Temporarily add y1
                        S.add(y1)
                        for t_idx in elem_triples[y1]:
                            counts[t_idx] += 1
                            
                        # Now try to add y2
                        for j in range(len(potential_adds)):
                            if i == j:
                                continue
                            y2 = potential_adds[j]
                            if y2 in S:
                                continue
                            can_add_y2 = True
                            for t_idx in elem_triples[y2]:
                                if counts[t_idx] >= 2:
                                    can_add_y2 = False
                                    break
                            if can_add_y2:
                                # Success! Add y2
                                S.add(y2)
                                for t_idx in elem_triples[y2]:
                                    counts[t_idx] += 1
                                found_pair = True
                                break
                        if found_pair:
                            break
                        
                        # Revert y1 if we didn't find a pair with it
                        if not found_pair:
                            S.remove(y1)
                            for t_idx in elem_triples[y1]:
                                counts[t_idx] -= 1
                    # If found_pair, we added y1 and y2. x is still removed.
                    # Net gain: +2 - 1 = +1.
                    # We accept this move.
                    # Note: x is removed from S permanently in this iteration.
                    # We should break out of the x loop to restart?
                    # Or continue?
                    # If we accept, the state of S and counts has changed.
                    # The 'removed' list is stale.
                    # Let's break and restart the phase.
                    if found_pair:
                        break
                else:
                    # No pair found, revert x
                    pass # x was already removed, we need to add it back?
                    # Wait, if we didn't find a pair, we should revert the removal of x.
                    S.add(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                
                if found_pair:
                    # State changed significantly. Break out to refresh lists.
                    break
                    
        return S

    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        S_greedy = greedy_removal(i)
        S_final = local_search(S_greedy, i + 1000)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)
            
    # Final validity check
    if not best_set:
        best_set = set(range(90, 181))
        
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        # Fallback repair
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c)
                
    return list(best_set)

# EVOLVE-BLOCK-END

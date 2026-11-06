import time
import random
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin (90s limit)
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    aps3 = []
    aps4 = []
    
    elem_to_aps3 = {i: [] for i in range(1, n+1)}
    elem_to_aps4 = {i: [] for i in range(1, n+1)}
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps3.append((a, b, c))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps4.append((a, b, c, e))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1
                
    total_aps3 = len(aps3)
    total_aps4 = len(aps4)
    
    # Helper to count 3-APs in a set A
    def count_3aps(A):
        count = 0
        for a, b, c in aps3:
            if a in A and b in A and c in A:
                count += 1
        return count
        
    # Helper to check if adding x to A creates a 4-AP
    def is_valid_add(A, x):
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(y in A for y in others):
                return False
        return True
        
    # Helper to count 4-APs in A
    def count_4aps(A):
        count = 0
        for a, b, c, d in aps4:
            if a in A and b in A and c in A and d in A:
                count += 1
        return count

    # Strategy: 
    # 1. Start with a high-quality initial set using a "density-aware" greedy.
    #    We prioritize elements that form many 3-APs but few 4-APs conflicts.
    # 2. Use a local search with deterministic moves:
    #    - "Swap" moves: Replace one element with another to increase 3-AP count.
    #    - "Insert" moves: Add elements if they don't break 4-AP freedom.
    #    - "Remove" moves: Remove elements that are "bad" (low contribution, high conflict).
    
    # To make local search fast, we maintain:
    # - A: current set
    # - score: current 3-AP count
    # - conflict_count[x]: number of 4-APs that would be formed if x was added to A (or are currently "threatened")
    #   Actually, simpler: just check validity on the fly for n=256, it's fast enough.
    #   The bottleneck is counting 3-APs. We need incremental updates.
    
    def get_delta_3aps_add(A, v):
        """Returns the number of new 3-APs formed by adding v to A."""
        gained = 0
        for ap_idx in elem_to_aps3[v]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != v]
            if all(y in A for y in others):
                gained += 1
        return gained

    def get_delta_3aps_remove(A, u):
        """Returns the number of 3-APs lost by removing u from A."""
        lost = 0
        for ap_idx in elem_to_aps3[u]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != u]
            if all(y in A for y in others):
                lost += 1
        return lost

    def get_delta_3aps_swap(A, u, v):
        """Returns the change in 3-AP count when swapping u (in A) for v (not in A)."""
        # Lost: 3-APs containing u where other 2 are in A
        lost = 0
        for ap_idx in elem_to_aps3[u]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != u]
            if all(y in A for y in others):
                lost += 1
                
        # Gained: 3-APs containing v where other 2 are in A\{u}
        gained = 0
        for ap_idx in elem_to_aps3[v]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != v]
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1
                
        return gained - lost

    def get_delta_4aps_swap(A, u, v):
        """Returns the change in 4-AP count when swapping u (in A) for v (not in A).
           We require the result to be 0 (or <= current, which is 0)."""
        lost = 0
        for ap_idx in elem_to_aps4[u]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != u]
            if all(y in A for y in others):
                lost += 1
                
        gained = 0
        for ap_idx in elem_to_aps4[v]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != v]
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1
                
        return gained - lost

    # Initial Construction: 
    # Use a greedy that maximizes immediate 3-AP gain, breaking ties by 4-AP penalty.
    # This is a "weighted greedy".
    
    def run_weighted_greedy():
        A = set()
        remaining = list(range(1, n + 1))
        
        while remaining:
            best_candidate = None
            best_gain = -1
            best_penalty = float('inf')
            
            # Optimization: Only check candidates that are valid.
            # n=256, so 256 checks per iteration. 256 iterations. 65k checks. Fast.
            for x in remaining:
                if not is_valid_add(A, x):
                    continue
                
                gain = get_delta_3aps_add(A, x)
                
                # Penalty: How many 4-APs does x "touch" with elements in A?
                # This is a heuristic for future flexibility.
                # If x is in many 4-APs with A, it might block future additions.
                # Simpler penalty: number of 4-APs x is part of in total? No, static.
                # Let's use: number of 4-APs containing x where at least one other element is in A.
                penalty = 0
                for ap_idx in elem_to_aps4[x]:
                    a, b, c, d = aps4[ap_idx]
                    others = [y for y in (a, b, c, d) if y != x]
                    if any(y in A for y in others):
                        penalty += 1
                        
                # We want high gain, low penalty.
                # Score = gain * 100 - penalty
                current_score = gain * 100 - penalty
                
                if current_score > best_gain:
                    best_gain = current_score
                    best_candidate = x
                    best_penalty = penalty
            
            if best_candidate is not None:
                A.add(best_candidate)
                remaining.remove(best_candidate)
            else:
                # No valid candidate? Should not happen if we start empty, 
                # unless we get stuck. But 1..256 always has valid singletons.
                # If stuck, break.
                break
                
        return A

    # Local Search: 
    # We perform a series of "best-improvement" swaps.
    # To make it efficient, we iterate through all possible swaps (u in A, v not in A)
    # and pick the one with the highest positive delta_3 that keeps 4-AP count at 0.
    
    def local_search(A, max_iterations=500):
        current_score = count_3aps(A)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            best_delta = 0
            best_u = None
            best_v = None
            best_delta4 = 0
            
            A_list = list(A)
            not_in_A = [x for x in range(1, n + 1) if x not in A]
            
            # Iterate all swaps
            for u in A_list:
                for v in not_in_A:
                    delta4 = get_delta_4aps_swap(A, u, v)
                    # If delta4 > 0, it creates a 4-AP. Invalid.
                    # If delta4 < 0, it removes a 4-AP (impossible if current is 0).
                    # So we must have delta4 == 0.
                    if delta4 != 0:
                        continue
                    
                    delta3 = get_delta_3aps_swap(A, u, v)
                    
                    if delta3 > best_delta:
                        best_delta = delta3
                        best_u = u
                        best_v = v
                        best_delta4 = delta4
            
            if best_delta > 0:
                A.remove(best_u)
                A.add(best_v)
                current_score += best_delta
            else:
                # No improving swap found. 
                # Try random perturbation to escape local optima?
                # Or try insertions?
                # Let's try a random valid insertion if possible.
                inserted = False
                for x in not_in_A:
                    if is_valid_add(A, x):
                        gain = get_delta_3aps_add(A, x)
                        if gain > 0:
                            A.add(x)
                            current_score += gain
                            inserted = True
                            break
                
                if not inserted:
                    # Try a random swap even if neutral or slightly negative?
                    # No, stick to best improvement. If stuck, break.
                    break
        
        return A, current_score

    # Run multiple initializations and local searches
    best_set = set()
    best_score = 0
    
    # 1. Weighted Greedy
    A_init = run_weighted_greedy()
    A_opt, score_opt = local_search(A_init, max_iterations=100)
    if score_opt > best_score:
        best_score = score_opt
        best_set = set(A_opt)
        
    # 2. Random Greedies with Local Search
    for i in range(5):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(42 + i)
        A_rand = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(A_rand, x):
                A_rand.add(x)
        
        A_opt, score_opt = local_search(A_rand, max_iterations=100)
        if score_opt > best_score:
            best_score = score_opt
            best_set = set(A_opt)

    # 3. Final Polish: Try to add any remaining elements that don't break 4-APs
    # and increase 3-AP count.
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for x in range(1, n + 1):
            if x not in best_set and is_valid_add(best_set, x):
                gain = get_delta_3aps_add(best_set, x)
                if gain > 0:
                    best_set.add(x)
                    best_score += gain
                    changed = True

    # Final Verification
    if count_4aps(best_set) != 0:
        # Fallback
        best_set = run_weighted_greedy()
        best_score = count_3aps(best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-END

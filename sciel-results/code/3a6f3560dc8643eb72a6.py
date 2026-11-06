import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Evaluator kills at 90s, task says 60s. 
    # Use 58s to be safe.
    time_limit = 58.0
    
    # 1. Precomputation of 4-AP conflicts and 3-AP contributions
    
    # conflict_mask[v] is a bitmask (int) where bit u is 1 if {u, v} forms part of a 4-AP.
    # Actually, we need to know if adding v to S creates a 4-AP.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # If we add v, we check if there exist d such that {v, v+d, v+2d, v+3d} subset S U {v} etc.
    # This is complex to precompute as a simple pairwise mask because 4-APs involve 4 elements.
    # However, we can precompute for each pair (u, v) if they are "conflicting" in a way that 
    # prevents both from being in S? No, 3 elements can be in S, just not 4 in AP.
    
    # Let's stick to the bitmask validity check but optimize the search.
    # The previous program used a slow Python loop for SA.
    # We will use a Greedy + Vectorized 2-Swap approach.
    
    # Bit values
    bit_val = [1 << i for i in range(n)]
    
    # Precompute 3-AP contributions for each element u.
    # ap3_contrib[u] is a list of masks M such that (S & M) == M implies u is in a 3-AP.
    # Actually, we want to count distinct 3-APs.
    # A 3-AP is {a, a+d, a+2d}.
    # Each 3-AP involves 3 elements.
    # If we maintain a count of 3-APs, adding an element x adds the number of 3-APs containing x 
    # that are fully contained in S U {x}.
    
    # Let's precompute for each x, the list of "partner pairs" (p1, p2) such that {x, p1, p2} is a 3-AP.
    # If we add x to S, we gain count of pairs (p1, p2) in S such that {x, p1, p2} is a 3-AP.
    # But we must ensure the 3-AP is not already counted.
    # It's easier to just recalculate the total 3-AP count from scratch or use a delta formula.
    
    # Delta formula for adding x to S:
    # gain = number of pairs {a, b} in S such that {a, b, x} forms a 3-AP.
    # We can precompute for each x a list of masks M_x such that (S & M_x) == M_x implies x completes a 3-AP with S.
    # Note: A 3-AP {a, b, c} will be counted 3 times if we sum over all x in S?
    # No, if we count "distinct 3-APs", each AP is a set of 3 numbers.
    # If we iterate x in S, and count pairs in S\{x}, each AP is counted exactly once for its "newest" element? 
    # No, the set is static.
    # Let's define: For each x, let Pairs(x) be the set of pairs {u, v} such that {x, u, v} is a 3-AP.
    # If we have a set S, the number of 3-APs in S is NOT simply sum_{x in S} |Pairs(x) intersect S|.
    # That sum counts each AP 3 times (once for each member).
    # So Total 3-APs = (1/3) * sum_{x in S} |Pairs(x) intersect S|.
    
    # Let's precompute pairs_for_x[x] as a list of masks (u_bit | v_bit).
    pairs_for_x = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # x is 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            u, v = x + d, x + 2 * d
            if v <= n:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, n - x) + 1):
            u, v = x - d, x + d
            if u >= 1:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        # x is 3rd: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            u, v = x - 2 * d, x - d
            if u >= 1:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        
    # Precompute 4-AP constraints for validity.
    # valid_add(x, S_mask) checks if x can be added to S without creating a 4-AP.
    # This requires checking all 4-APs containing x.
    # For each 4-AP containing x, if the other 3 elements are in S, then x cannot be added.
    # Let's precompute forbidden_masks_4[x] as a list of masks M such that if (S & M) == M, x is forbidden.
    forbidden_4 = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        masks = []
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if x + 3 * d <= n:
                m = bit_val[x + d - 1] | bit_val[x + 2 * d - 1] | bit_val[x + 3 * d - 1]
                masks.append(m)
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, n // 3 + 1):
            if x - d >= 1 and x + 2 * d <= n:
                m = bit_val[x - d - 1] | bit_val[x + d - 1] | bit_val[x + 2 * d - 1]
                masks.append(m)
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, n // 3 + 1):
            if x - 2 * d >= 1 and x + d <= n:
                m = bit_val[x - 2 * d - 1] | bit_val[x - d - 1] | bit_val[x + d - 1]
                masks.append(m)
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, n // 3 + 1):
            if x - 3 * d >= 1:
                m = bit_val[x - 3 * d - 1] | bit_val[x - 2 * d - 1] | bit_val[x - d - 1]
                masks.append(m)
        forbidden_4[x] = masks

    def is_valid_addition(S_mask, x):
        for m in forbidden_4[x]:
            if (S_mask & m) == m:
                return False
        return True

    def count_3aps(S_mask):
        # (1/3) * sum_{x in S} count_pairs_in_S(x)
        total = 0
        S_set = {i + 1 for i in range(n) if (S_mask >> i) & 1}
        for x in S_set:
            for m in pairs_for_x[x]:
                if (S_mask & m) == m:
                    total += 1
        return total // 3

    # 2. Greedy Initialization
    # Start with empty set, add elements one by one if valid.
    # To maximize 3-APs, we should prioritize elements that have high potential for 3-APs.
    # Or just random order? Random order is often good for diversity.
    # Let's try a few random greedy starts and keep the best.
    
    best_S_mask = 0
    best_count = 0
    
    # Run multiple greedy restarts
    num_restarts = 10
    for r in range(num_restarts):
        S_mask = 0
        candidates = list(range(1, n + 1))
        np.random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S_mask, x):
                S_mask |= bit_val[x - 1]
                
        cnt = count_3aps(S_mask)
        if cnt > best_count:
            best_count = cnt
            best_S_mask = S_mask
            
    # 3. Local Search: 2-Swap (Replace one element with another)
    # We iterate through all pairs (u in S, v not in S).
    # If replacing u with v is valid and increases 3-AP count, we do it.
    # We repeat until no improvement is found or time limit.
    
    S_mask = best_S_mask
    current_count = best_count
    S_set = {i + 1 for i in range(n) if (S_mask >> i) & 1}
    N_set = set(range(1, n + 1)) - S_set
    
    # To speed up, we can vectorize the validity check for all v in N_set given u in S.
    # But Python sets and loops are okay for n=256.
    # Number of pairs is |S| * |N| ~ 100 * 150 = 15000.
    # Each check is O(1) average if we use bit masks.
    # 15000 checks is very fast.
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Sort S_set and N_set? Not necessary.
        # We need to be careful: modifying S_set while iterating is bad.
        # We collect moves first.
        
        moves = []
        S_list = list(S_set)
        N_list = list(N_set)
        
        for u in S_list:
            # Temporarily remove u
            S_mask_without_u = S_mask & ~bit_val[u - 1]
            
            # Calculate loss of 3-APs from removing u
            # Loss = (1/3) * count_pairs_in_S(u) ? No.
            # If we remove u, we lose all 3-APs containing u.
            # Number of 3-APs containing u in S is (1/3) * |{pairs in S that complete 3-AP with u}|?
            # No, total 3-APs = (1/3) * Sum_{x in S} C_x, where C_x is count of pairs in S completing AP with x.
            # If we remove u, new count = (1/3) * (Sum_{x in S\{u}} C_x_new).
            # C_x_new might change if x was paired with u?
            # Yes, if {x, y, u} was an AP, then y and x no longer form a pair for u?
            # Actually, C_x counts pairs {a,b} in S such that {x,a,b} is AP.
            # If u is removed, for any x != u, C_x might decrease if {x, u, y} was an AP.
            # This makes delta calculation complex.
            
            # Simpler: Just calculate new count from scratch?
            # count_3aps is O(|S|^2) roughly? No, O(|S| * avg_pairs).
            # |S| ~ 100, avg_pairs ~ 20. 2000 ops.
            # 15000 swaps * 2000 ops = 30M ops. Might be too slow in Python.
            
            # Let's optimize delta calculation.
            # Delta = Count(S - {u} + {v}) - Count(S).
            # Count(S) = (1/3) * Sum_{x in S} C_x(S).
            # Let C_x(S) be number of pairs in S completing 3-AP with x.
            # When we remove u:
            # 1. Term for u vanishes.
            # 2. For each x in S\{u}, C_x(S\{u}) = C_x(S) - (1 if u is in a pair completing AP with x else 0).
            #    Note: u is in a pair completing AP with x if {u, y, x} is AP for some y in S.
            #    This is equivalent to: is there y in S such that {x, u, y} is AP?
            #    This is just checking if the "partner" of u relative to x is in S.
            #    For a fixed x and u, there is at most one y such that {x, u, y} is AP.
            #    So C_x(S\{u}) = C_x(S) - I(u, x, S).
            
            # This is getting complicated to vectorize easily without precomputing more structures.
            # Given n=256, let's try the naive "recalculate count" approach but optimized.
            # If it's too slow, we'll see.
            
            for v in N_list:
                # Check validity of v in S\{u}
                valid = True
                for m in forbidden_4[v]:
                    if (S_mask_without_u & m) == m:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try swap
                new_S_mask = S_mask_without_u | bit_val[v - 1]
                new_count = count_3aps(new_S_mask)
                
                if new_count > current_count:
                    moves.append((u, v, new_count))
                    # We can break early if we find a good move?
                    # Or collect all and pick best?
                    # Picking best is better for convergence.
        
        if moves:
            # Pick the move with max new_count
            best_move = max(moves, key=lambda x: x[2])
            u, v, new_count = best_move
            
            S_mask = (S_mask & ~bit_val[u - 1]) | bit_val[v - 1]
            S_set.remove(u)
            S_set.add(v)
            N_set.remove(v)
            N_set.add(u)
            
            current_count = new_count
            improved = True
            best_S_mask = S_mask
            best_count = current_count

    # Final result
    res_list = sorted([i + 1 for i in range(n) if (best_S_mask >> i) & 1])
    if not res_list:
        return [1]
    return res_list

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])

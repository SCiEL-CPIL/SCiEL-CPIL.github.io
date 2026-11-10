import time
import random
import sys
from functools import lru_cache

N = 256
DEADLINE = 55.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Precompute all 3-APs and 4-APs with bitmasks
    three_aps = []
    aps_by_elem = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            mask_bc = (1 << (b-1)) | (1 << (c-1))
            aps_by_elem[a].append(mask_bc)
            mask_ac = (1 << (a-1)) | (1 << (c-1))
            aps_by_elem[b].append(mask_ac)
            mask_ab = (1 << (a-1)) | (1 << (b-1))
            aps_by_elem[c].append(mask_ab)
            
    four_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask_bce = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[a].append(mask_bce)
            mask_ace = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[b].append(mask_ace)
            mask_abe = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            four_aps_by_v[c].append(mask_abe)
            mask_abc = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            four_aps_by_v[e].append(mask_abc)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def get_set_bits(S):
        bits = []
        temp = S
        i = 0
        while temp:
            if temp & 1:
                bits.append(i + 1)
            temp >>= 1
            i += 1
        return bits

    def count_3aps(S):
        count = 0
        for a, b, c in three_aps:
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            if (S & mask) == mask:
                count += 1
        return count

    # Weight of each vertex = number of 3-APs it participates in
    vertex_weight = [0] * (N + 1)
    for a, b, c in three_aps:
        vertex_weight[a] += 1
        vertex_weight[b] += 1
        vertex_weight[c] += 1

    # Build conflict graph: two vertices conflict if they can co-occur in a 4-AP
    # Actually, we need a hypergraph: 4-APs are the constraints.
    # For branch-and-bound, we model it as: we want to select a subset S such that
    # no 4-AP is fully contained in S. This is an independent set problem on a 4-uniform hypergraph.
    
    # For B&B, we process vertices one by one. At each step, we decide include/exclude.
    # Upper bound: current_score + sum of weights of remaining vertices that could still be added
    # (without violating 4-AP constraints). A simple upper bound is just the sum of all remaining weights.
    
    # Better: use a greedy upper bound on how many more 3-APs we can get.
    
    # Strategy: Branch and bound with ordering by weight (descending).
    # We maintain:
    #   - current set S (bitmask)
    #   - current 3-AP count
    #   - index of next vertex to consider
    #   - upper bound = current_count + optimistic estimate of additional 3-APs
    
    # For the upper bound, we can use: for each remaining vertex v, the max additional 3-APs
    # it could contribute is its weight, but that's too optimistic since 3-APs require 3 vertices.
    # A tighter bound: for each remaining 3-AP, check if all 3 vertices are still "available"
    # (not excluded and not causing a 4-AP conflict).
    
    # Simpler approach: use the sum of vertex weights of remaining vertices as an upper bound
    # on the number of 3-APs (since each 3-AP involves 3 vertices, this is 3x the actual count,
    # but it's still a valid upper bound for pruning).
    
    # Even better: precompute for each vertex which 3-APs it's in, and track which 3-APs
    # are still "possible" (all 3 vertices not yet excluded).
    
    # Let's use a different strategy: 
    # 1. First, get a good initial solution via greedy
    # 2. Then use branch-and-bound to improve, starting from the greedy solution
    
    # Greedy construction (same as before)
    def greedy_construct():
        S = 0
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            valid = True
            for mask in four_aps_by_v[x]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= bit
        return S

    S = greedy_construct()
    current_score = count_3aps(S)
    best_S = S
    best_score = current_score

    # Branch and bound improvement
    # We'll try to improve by considering swaps: remove a low-weight vertex, add a high-weight one
    # But use B&B to decide which swaps to make
    
    # Alternative B&B: process vertices in order of decreasing weight.
    # At each vertex, decide include or exclude.
    # Upper bound: current 3-AP count + (number of 3-APs that could still be completed)
    
    # For efficiency, let's use a modified approach:
    # Start with the greedy solution. Then try local improvements using B&B-style pruning.
    
    # Let's implement a proper B&B:
    # Order vertices by weight descending
    ordered_vertices = list(range(1, N + 1))
    ordered_vertices.sort(key=lambda x: -vertex_weight[x])
    
    # Precompute: for each vertex, which 3-AP indices it belongs to
    vertex_to_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        vertex_to_3aps[a].append(idx)
        vertex_to_3aps[b].append(idx)
        vertex_to_3aps[c].append(idx)
    
    # B&B state:
    # - decision array: for each vertex in ordered_vertices, 0=undecided, 1=include, 2=exclude
    # - current set bitmask
    # - current 3-AP count
    # - index of next vertex to decide
    
    # Upper bound calculation:
    # For each 3-AP, if all 3 vertices are either included or undecided, it's still possible.
    # The upper bound is current_count + (number of possible 3-APs that are not yet counted)
    
    # This is expensive to compute exactly. Let's use a simpler bound:
    # UB = current_count + sum of weights of undecided vertices / 3
    # (since each 3-AP has 3 vertices, this is an upper bound)
    
    # Even simpler: UB = current_count + (total_weight_of_undecided_vertices)
    # This is a valid upper bound (each 3-AP contributes 3 to the weight sum)
    
    # Let's implement B&B with this bound.
    
    sys.setrecursionlimit(10000)
    
    best_score_bb = best_score
    best_S_bb = best_S
    nodes_explored = 0
    
    def is_valid_addition(S, v):
        """Check if adding v to S doesn't create a 4-AP"""
        v_bit = 1 << (v - 1)
        for mask in four_aps_by_v[v]:
            if (S & mask) == mask:
                return False
        return True
    
    def compute_3ap_count_incremental(S, v):
        """Compute how many new 3-APs are formed by adding v to S"""
        count = 0
        for mask in aps_by_elem[v]:
            if (S & mask) == mask:
                count += 1
        return count
    
    # B&B recursion
    def branch_and_bound(idx, S, current_count, undecided_weight_sum):
        nonlocal best_score_bb, best_S_bb, nodes_explored
        
        if time_left() < 3.0:
            return
        
        nodes_explored += 1
        if nodes_explored % 10000 == 0 and time_left() < 5.0:
            return
        
        # Upper bound
        ub = current_count + undecided_weight_sum
        if ub <= best_score_bb:
            return
        
        if idx == len(ordered_vertices):
            if current_count > best_score_bb:
                best_score_bb = current_count
                best_S_bb = S
            return
        
        v = ordered_vertices[idx]
        v_bit = 1 << (v - 1)
        w = vertex_weight[v]
        
        # Try including v
        if is_valid_addition(S, v):
            new_count = current_count + compute_3ap_count_incremental(S, v)
            new_undecided = undecided_weight_sum - w
            branch_and_bound(idx + 1, S | v_bit, new_count, new_undecided)
        
        # Try excluding v
        new_undecided = undecided_weight_sum - w
        branch_and_bound(idx + 1, S, current_count, new_undecided)
    
    # Start B&B from empty set
    total_weight = sum(vertex_weight[1:])
    branch_and_bound(0, 0, 0, total_weight)
    
    if best_score_bb > best_score:
        best_score = best_score_bb
        best_S = best_S_bb
    
    # Also try starting B&B from the greedy solution
    # Mark greedy vertices as included, others as undecided
    # This is more complex, so let's skip for now
    
    # Convert best_S bitmask to sorted list
    result = get_set_bits(best_S)
    return sorted(result)

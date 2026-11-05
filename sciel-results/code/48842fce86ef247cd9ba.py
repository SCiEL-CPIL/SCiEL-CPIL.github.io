"""
Strategy: Optimized Maximum Weight Independent Set (MWIS) solver with Local Search Polish.
1. Uses a recursive branch-and-bound algorithm with bitmask representation for speed.
2. Uses greedy coloring for upper bounds.
3. Applies a local search phase (1-for-1, 1-for-2, 2-for-1 swaps) to refine the solution found by B&B.
"""

import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    N = 512
    # Map values 1..511 to indices 0..510
    # Index i corresponds to value i + 1
    # Weight of index i is 1 / (512 - (i+1)) = 1 / (511 - i)
    
    # Precompute weights
    weights = [0.0] * (N - 1)
    for i in range(N - 1):
        weights[i] = 1.0 / (N - 1 - i)
        
    # Precompute coprime masks
    # mask[i] is an integer with bits set for all j such that gcd(i+1, j+1) == 1
    coprime_masks = [0] * (N - 1)
    for i in range(N - 1):
        val_i = i + 1
        mask = 0
        for j in range(N - 1):
            if i != j and gcd(val_i, j + 1) == 1:
                mask |= (1 << j)
        coprime_masks[i] = mask
        
    # Order nodes by weight descending (index 510 has weight 1.0, index 0 has weight 1/511)
    # We want to try high weight nodes first.
    # Indices: 510 (val 511), 509 (val 510), ..., 0 (val 1)
    # So order is [510, 509, ..., 0]
    order = list(range(N - 1, -1, -1))
    
    best_weight = 0.0
    best_clique_indices = []
    
    def greedy_color_bound(P_mask):
        """
        Returns an upper bound on the maximum weight of a clique in P_mask.
        Uses greedy coloring on the subgraph induced by P_mask.
        """
        if P_mask == 0:
            return 0.0
        
        # Extract nodes in P_mask
        # We iterate through the global order and pick those in P_mask
        # This ensures we process high-weight nodes first for the greedy coloring assignment?
        # Standard greedy coloring: assign smallest color.
        # To get a tight bound (low sum of max weights per color), we should assign colors
        # such that high weight nodes get distinct colors if possible? 
        # Actually, the bound is sum of max weight in each color class.
        # If we put two high weight nodes in the same color class, the max is the higher one.
        # If we put them in different classes, the sum is w1 + w2.
        # We want the bound to be LOW to prune. So we want to MINIMIZE the sum of max weights.
        # This means we should try to put high-weight nodes in the SAME color class if they are compatible (not adjacent in coprime graph, i.e., gcd > 1).
        # Wait. In the coprime graph, an edge means "compatible" (gcd=1).
        # We are coloring the COPRIME graph.
        # Adjacent nodes in coprime graph must have DIFFERENT colors.
        # So if u and v are coprime, they MUST have different colors.
        # So we cannot put coprime nodes in the same color class.
        # We CAN put non-coprime nodes (gcd > 1) in the same color class.
        # To minimize the sum of max weights, we should group high-weight nodes that are NOT coprime together.
        
        # Greedy strategy:
        # Process nodes in order of decreasing weight.
        # For each node u, assign the smallest color c such that no neighbor (coprime node) of u in P has color c.
        # This tends to assign high-weight nodes to early colors.
        # If a high-weight node u is coprime to another high-weight node v, they get different colors.
        # If they are not coprime, they might get the same color (if v was assigned earlier and didn't block u? No, if v is earlier, u checks v. If v is coprime to u, v's color is forbidden. If not coprime, v's color is NOT forbidden, so u can take v's color).
        # So this greedy method naturally groups non-coprime high-weight nodes, which minimizes the sum of max weights. This is good for a tight upper bound.
        
        active_nodes = []
        # Efficient extraction of set bits in order of weight?
        # Since order is sorted by weight descending, we can just iterate order and check bit.
        for i in order:
            if (P_mask >> i) & 1:
                active_nodes.append(i)
                
        if not active_nodes:
            return 0.0
            
        colors = {} # node index -> color int
        color_max_weight = {} # color int -> max weight
        
        for u in active_nodes:
            # Find forbidden colors
            forbidden = set()
            # Iterate neighbors of u in P
            # neighbors are set bits in coprime_masks[u] & P_mask
            neighbors_mask = coprime_masks[u] & P_mask
            # Iterate set bits in neighbors_mask
            # This is O(degree)
            m = neighbors_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v in colors:
                    forbidden.add(colors[v])
                m ^= lsb
                
            # Assign smallest available color
            c = 1
            while c in forbidden:
                c += 1
                
            colors[u] = c
            w = weights[u]
            if c not in color_max_weight or w > color_max_weight[c]:
                color_max_weight[c] = w
                
        return sum(color_max_weight.values())

    # Initial incumbent: Greedy solution
    # Pick highest weight node, remove neighbors, repeat.
    P_mask = (1 << (N - 1)) - 1 # All bits set
    R_greedy = []
    while P_mask:
        # Pick node with max weight in P_mask
        # Iterate order
        u = -1
        for i in order:
            if (P_mask >> i) & 1:
                u = i
                break
        if u == -1:
            break
        R_greedy.append(u)
        # Remove u and its coprime neighbors from P_mask
        # Keep only nodes NOT coprime to u? 
        # No, we want to keep nodes that are coprime to u? 
        # Wait. We are building a clique in the coprime graph.
        # If we pick u, all other nodes in the clique MUST be coprime to u.
        # So we should RESTRICT P to nodes coprime to u.
        # P_new = P & coprime_masks[u]
        P_mask &= coprime_masks[u]
        P_mask &= ~(1 << u) # Remove u itself
        
    best_weight = sum(weights[u] for u in R_greedy)
    best_clique_indices = list(R_greedy)
    
    # Recursive search
    import sys
    sys.setrecursionlimit(10000)
    
    def search(R, P_mask, current_weight):
        nonlocal best_weight, best_clique_indices
        
        if time.time() > deadline - 20: # Leave 20s for polish
            return
            
        if P_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique_indices = list(R)
            return
            
        ub = greedy_color_bound(P_mask)
        if current_weight + ub <= best_weight:
            return
            
        # Branching
        # Iterate nodes in P_mask sorted by weight descending
        # We can reuse the order list
        for u in order:
            if not ((P_mask >> u) & 1):
                continue
                
            # Branch on including u
            new_R = R + [u]
            new_P_mask = P_mask & coprime_masks[u]
            new_P_mask &= ~(1 << u) # u is not in new_P_mask anyway as it's not coprime to itself? 
                                    # gcd(u,u)=u>1 (unless u=1). 
                                    # If u=1, gcd(1,1)=1. So 1 is coprime to 1.
                                    # But we need distinct integers. So we must remove u.
            
            search(new_R, new_P_mask, current_weight + weights[u])
            
            # Remove u from P_mask for subsequent branches
            P_mask &= ~(1 << u)
            
            if time.time() > deadline - 20:
                return

    # Start search
    P_start = (1 << (N - 1)) - 1
    search([], P_start, 0.0)
    
    # Local Search Polish
    # Convert best_clique_indices to a set of values for quick lookup? 
    # Or keep indices.
    # S is the set of selected indices.
    S = set(best_clique_indices)
    current_w = best_weight
    
    def get_value(indices):
        return sum(weights[i] for i in indices)
        
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1-for-1 swap
        # Remove a, add b. b not in S, a in S.
        # b must be coprime to all x in S \ {a}
        # w_b > w_a
        for a in list(S):
            # Try all b not in S
            # Optimization: only try b with w_b > w_a
            for b in order:
                if b in S:
                    continue
                if weights[b] <= weights[a]:
                    continue # No gain in 1-1 swap if weight is not higher? 
                             # Actually, we need w_b > w_a for strict improvement.
                # Check coprime condition
                ok = True
                for x in S:
                    if x == a:
                        continue
                    # Check if b and x are coprime
                    # i.e., bit x is set in coprime_masks[b]
                    if not ((coprime_masks[b] >> x) & 1):
                        ok = False
                        break
                if ok:
                    # Perform swap
                    S.remove(a)
                    S.add(b)
                    current_w += weights[b] - weights[a]
                    improved = True
                    break # Restart loop after improvement? Or continue?
                    # Restarting is safer to avoid missing other swaps, but slower.
                    # Given small N, restart is fine.
            if improved:
                break
                
        if improved:
            continue
            
        # 2-for-1 swap
        # Remove a, c. Add b.
        # b not in S. a, c in S.
        # b coprime to all x in S \ {a, c}
        # w_b > w_a + w_c
        # Also a and c must be coprime? No, they are in S, so they are coprime.
        # We iterate pairs a, c from S.
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                c = S_list[j]
                target_w = weights[a] + weights[c]
                # Try b not in S with w_b > target_w
                for b in order:
                    if b in S:
                        continue
                    if weights[b] <= target_w:
                        continue
                    ok = True
                    for x in S:
                        if x == a or x == c:
                            continue
                        if not ((coprime_masks[b] >> x) & 1):
                            ok = False
                            break
                    if ok:
                        S.remove(a)
                        S.remove(c)
                        S.add(b)
                        current_w += weights[b] - target_w
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        if improved:
            continue

        # 1-for-2 swap
        # Remove a. Add b, c.
        # b, c not in S, b != c.
        # b, c coprime to each other.
        # b, c coprime to all x in S \ {a}
        # w_b + w_c > w_a
        for a in list(S):
            # Try pairs b, c
            # Optimization: w_b + w_c > w_a
            # Since weights are positive, we can limit search.
            for i in range(len(order)):
                b = order[i]
                if b in S:
                    continue
                if weights[b] >= weights[a]:
                    # If w_b >= w_a, then even w_c=0 (impossible) might not help? 
                    # We need w_b + w_c > w_a.
                    # If w_b > w_a, we just need w_c > 0 and coprime conditions.
                    # But b alone would have been a 1-1 swap candidate if coprime.
                    # Since 1-1 failed, b is not coprime to S\{a}.
                    # So b must be coprime to S\{a} for this to work.
                    pass 
                
                for j in range(i + 1, len(order)):
                    c = order[j]
                    if c in S:
                        continue
                    if weights[b] + weights[c] <= weights[a]:
                        continue
                    # Check b, c coprime
                    if not ((coprime_masks[b] >> c) & 1):
                        continue
                    ok = True
                    for x in S:
                        if x == a:
                            continue
                        if not ((coprime_masks[b] >> x) & 1) or not ((coprime_masks[c] >> x) & 1):
                            ok = False
                            break
                    if ok:
                        S.remove(a)
                        S.add(b)
                        S.add(c)
                        current_w += weights[b] + weights[c] - weights[a]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Update best
    if current_w > best_weight:
        best_weight = current_w
        best_clique_indices = list(S)
        
    # Convert indices back to values
    result = [i + 1 for i in best_clique_indices]
    return result

if __name__ == "__main__":
    res = construct()
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    print(f"Count: {len(res)}")

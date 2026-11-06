import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = 58.0 # Safety margin
    
    N = 256
    all_nodes = list(range(2, N + 1))
    
    # 1. Identify Universal Primes
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
            
    universal_primes = [v for v in all_nodes if v > 128 and is_prime(v)]
    universal_set = set(universal_primes)
    
    # 2. Active Nodes (Non-universal)
    active_nodes = [v for v in all_nodes if v not in universal_set]
    M = len(active_nodes)
    node_to_idx = {v: i for i, v in enumerate(active_nodes)}
    
    # 3. Build Conflict/Compat Masks for Active Nodes
    # compat[i] is a bitmask of active nodes compatible with active_nodes[i]
    compat_mask = [0] * M
    for i in range(M):
        a = active_nodes[i]
        mask = 0
        for j in range(M):
            if i == j: continue
            b = active_nodes[j]
            # Check if lcm(a,b) > 256
            # a*b > 256 * gcd(a,b)
            if a * b > 256 * gcd(a, b):
                mask |= (1 << j)
        compat_mask[i] = mask
        
    weights = [1.0 / v for v in active_nodes]
    
    # 4. Heuristic Initial Solution
    # Greedy: Pick highest weight (smallest number) compatible with current set
    # Or just use the smallest number 2?
    # Let's try a few simple heuristics to get a good lower bound quickly
    best_clique = []
    best_weight = 0.0
    
    # Heuristic 1: Greedy by weight
    def greedy_solve(order):
        sel = []
        used = 0
        for i in order:
            if used & ~compat_mask[i]: # Check if i conflicts with any in sel?
                # Wait, compat_mask[i] has bits for compatible nodes.
                # We need to check if any node in 'sel' is NOT in compat_mask[i].
                # Or simpler: maintain 'used' as bitset of selected nodes.
                # If (used & ~compat_mask[i]) != 0, then i conflicts with some selected.
                # Actually, compat_mask[i] includes i? No, i!=j.
                # So if (used & ~compat_mask[i]) != 0, conflict.
                pass
            # Let's implement properly
            sel = []
            used = 0
            for i in order:
                if used & ~compat_mask[i]:
                    continue
                sel.append(i)
                used |= (1 << i)
            return sel
            
    # Order by weight descending (smallest numbers first)
    order1 = sorted(range(M), key=lambda i: weights[i], reverse=True)
    h1 = greedy_solve(order1)
    w1 = sum(weights[i] for i in h1)
    if w1 > best_weight:
        best_weight = w1
        best_clique = h1[:]
        
    # Order by conflict count ascending
    conflict_counts = [bin(~compat_mask[i] & ((1<<M)-1)).count('1') for i in range(M)]
    # Note: ~compat_mask[i] might have bits beyond M set. Mask it.
    all_mask = (1 << M) - 1
    conflict_counts = [bin((all_mask ^ compat_mask[i])).count('1') for i in range(M)]
    order2 = sorted(range(M), key=lambda i: conflict_counts[i])
    h2 = greedy_solve(order2)
    w2 = sum(weights[i] for i in h2)
    if w2 > best_weight:
        best_weight = w2
        best_clique = h2[:]

    # 5. Branch and Bound
    sys.setrecursionlimit(10000)
    
    def weighted_color_bound(candidates_mask):
        if candidates_mask == 0:
            return 0.0
        
        # Get list of candidate indices
        cands = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cands.append(idx)
            temp ^= lsb
            
        # Sort by weight descending
        cands.sort(key=lambda i: weights[i], reverse=True)
        
        # Greedy Coloring
        # colors[v] = color index
        colors = [-1] * len(cands)
        color_classes = [] # List of sets of indices (in cands list)
        
        for idx_pos, v in enumerate(cands):
            placed = False
            for c, cls in enumerate(color_classes):
                # Can v be in class c?
                # v must not conflict with any u in cls.
                # i.e., u must be in compat_mask[v].
                can_place = True
                for u in cls:
                    if not (compat_mask[v] >> u) & 1:
                        can_place = False
                        break
                if can_place:
                    cls.append(v)
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                
        # Bound: Sum of max weight in each color class
        bound = 0.0
        for cls in color_classes:
            max_w = max(weights[v] for v in cls)
            bound += max_w
            
        return bound

    def bb(candidates_mask, current_clique, current_weight):
        nonlocal best_clique, best_weight
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
            
        ub = current_weight + weighted_color_bound(candidates_mask)
        if ub <= best_weight:
            return
            
        # Branching: Pick a vertex
        # Strategy: Pick vertex with highest weight among candidates
        # To speed up, we can iterate through candidates in weight order
        cands = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cands.append(idx)
            temp ^= lsb
            
        cands.sort(key=lambda i: weights[i], reverse=True)
        
        for v in cands:
            if time.time() > deadline:
                return
                
            # Include v
            new_clique = current_clique + [v]
            new_weight = current_weight + weights[v]
            
            if new_weight > best_weight:
                # Update best if this is a valid clique (it is by construction)
                # We don't need to wait for leaf to update if we assume we can extend?
                # No, we must ensure it's a maximal clique or just a valid one.
                # Any valid clique is a candidate for best.
                best_weight = new_weight
                best_clique = new_clique[:]
                
            # New candidates for the branch including v
            new_cands = candidates_mask & compat_mask[v] & ~(1 << v)
            
            bb(new_cands, new_clique, new_weight)
            
            # Exclude v
            candidates_mask &= ~(1 << v)
            
            # Pruning check for exclude branch
            if candidates_mask == 0:
                break
            ub_excl = current_weight + weighted_color_bound(candidates_mask)
            if ub_excl <= best_weight:
                break
                
    # Start B&B
    # Initial candidates: all active nodes
    initial_mask = (1 << M) - 1
    bb(initial_mask, [], 0.0)
    
    # 6. Final Result
    # Convert best_clique indices back to values
    result_values = [active_nodes[i] for i in best_clique]
    result_values.extend(universal_primes)
    
    # Optional: Local Search Polish
    # Given B&B is strong, this might not be needed, but safe to add if time remains
    if time.time() < deadline - 2:
        # Simple add-only local search
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            used_vals = set(result_values)
            for v in all_nodes:
                if v in used_vals:
                    continue
                # Check compatibility with all in result_values
                ok = True
                for u in result_values:
                    if u * v <= 256 * gcd(u, v):
                        ok = False
                        break
                if ok:
                    result_values.append(v)
                    used_vals.add(v)
                    improved = True
                    
    if not result_values:
        return [2]
        
    return result_values

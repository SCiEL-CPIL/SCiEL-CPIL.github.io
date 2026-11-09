import time
import random
import numpy as np
from collections import deque

N = 512
DEADLINE = 55.0  # Safety margin within 90s evaluator limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs and build conflict graph
    # For each element x, store which other elements it conflicts with (i.e., pairs that would complete a 4-AP)
    # Actually, for MIS formulation: two elements conflict if they can be part of the same 4-AP
    # But that's not quite right. Let me think again.
    
    # A set S is 4-AP-free if there is no 4-AP entirely within S.
    # This is NOT the same as an independent set in a graph where edges represent 4-AP membership.
    # It's a hypergraph independent set problem.
    
    # However, we can still use a branch-and-bound approach:
    # - Start with a greedy solution as lower bound
    # - Branch on whether to include or exclude elements
    # - Use bounds to prune
    
    # Let's precompute for each element x, the list of 4-APs that contain x
    ap_triples = [[] for _ in range(N + 1)]
    all_aps = []  # List of all 4-APs as tuples (a, b, c, e)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            all_aps.append((a, b, c, e))
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        for (a, b, c) in ap_triples[x]:
            if (mask >> a & 1) and (mask >> b & 1) and (mask >> c & 1):
                return False
        return True
    
    def greedy_construct(order=None):
        """Greedy construction."""
        mask = 0
        if order is None:
            order = list(range(1, N + 1))
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask
    
    def polish(mask, deadline):
        """Polish a solution by trying to add elements."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if not (mask >> x & 1) and can_add(mask, x):
                    mask |= (1 << x)
                    improved = True
        return mask
    
    # Get initial lower bound from multiple greedy restarts
    best_mask = 0
    best_size = 0
    
    # Deterministic greedy
    mask = greedy_construct()
    mask = polish(mask, deadline - 20)
    best_size = bin(mask).count('1')
    best_mask = mask
    
    # Randomized greedy restarts
    rng = random.Random(42)
    for trial in range(20):
        if time.time() >= deadline - 25:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = greedy_construct(order)
        mask = polish(mask, deadline - 25)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Now try branch-and-bound to improve
    # We'll use a recursive approach with pruning
    # Key insight: if we have a partial assignment, we can compute an upper bound
    # on the maximum size we can achieve
    
    # For B&B, we need to decide the order of branching.
    # Use Minimum Remaining Values (MRV) heuristic: branch on elements that are
    # most constrained (appear in many 4-APs with other undecided elements)
    
    # Precompute degree (number of 4-APs each element is in)
    degree = [0] * (N + 1)
    for x in range(1, N + 1):
        degree[x] = len(ap_triples[x])
    
    # Sort elements by degree descending for branching order
    # Elements with higher degree are more constraining
    branch_order = sorted(range(1, N + 1), key=lambda x: -degree[x])
    
    # B&B state:
    # - mask: elements included
    # - forbidden: elements that cannot be added (because they'd create a 4-AP)
    # - undecided: elements not yet decided
    
    # Actually, let's use a simpler approach:
    # - We maintain a set S (included) and a set of candidates that can still be added
    # - At each step, pick an element from candidates, branch on include/exclude
    
    # But this is still exponential. Let's limit the depth.
    
    # Alternative: use local search with more sophisticated moves
    # Try 2-exchange: remove two elements, add one or two different elements
    
    def local_search_advanced(mask, rng, deadline):
        """Advanced local search with 1-add, 1-swap, and 2-swap moves."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # 1. Try adding any blocked element
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            for x in blocked_list:
                if time.time() >= deadline:
                    return mask, S_size
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
                    blocked.discard(x)
                    improved = True
                    no_improve = 0
                    break
            
            if improved:
                continue
            
            # 2. Try 1-swap: remove r, add a
            swap_found = False
            for a in blocked_list[:20]:
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                for r in S_list[:50]:
                    if time.time() >= deadline:
                        return mask, S_size
                    
                    mask_without_r = mask & ~(1 << r)
                    if can_add(mask_without_r, a):
                        mask = mask_without_r | (1 << a)
                        blocked.discard(a)
                        blocked.add(r)
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                no_improve = 0
                continue
            
            # 3. Try 2-swap: remove r1, r2, add a1, a2 (size-preserving or size-increasing)
            # Only try if we haven't improved in a while
            if no_improve > 5:
                two_swap_found = False
                blocked_list2 = list(blocked)
                rng.shuffle(blocked_list2)
                
                for a1 in blocked_list2[:10]:
                    if time.time() >= deadline:
                        return mask, S_size
                    
                    for a2 in blocked_list2[:10]:
                        if a1 == a2:
                            continue
                        if time.time() >= deadline:
                            return mask, S_size
                        
                        S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                        rng.shuffle(S_list)
                        
                        for r1 in S_list[:30]:
                            if time.time() >= deadline:
                                return mask, S_size
                            for r2 in S_list[:30]:
                                if r1 == r2:
                                    continue
                                if time.time() >= deadline:
                                    return mask, S_size
                                
                                mask_without = mask & ~(1 << r1) & ~(1 << r2)
                                if can_add(mask_without, a1) and can_add(mask_without | (1 << a1), a2):
                                    # Perform 2-swap
                                    mask = mask_without | (1 << a1) | (1 << a2)
                                    blocked.discard(a1)
                                    blocked.discard(a2)
                                    blocked.add(r1)
                                    blocked.add(r2)
                                    two_swap_found = True
                                    break
                                if two_swap_found:
                                    break
                            if two_swap_found:
                                break
                        if two_swap_found:
                            break
                    if two_swap_found:
                        break
                
                if two_swap_found:
                    no_improve = 0
                    continue
            
            no_improve += 1
        
        return mask, S_size
    
    # Run advanced local search on the best solution
    if time.time() < deadline - 5:
        best_mask, best_size = local_search_advanced(best_mask, rng, deadline - 3)
    
    # Final polish
    if time.time() < deadline - 1:
        best_mask = polish(best_mask, deadline)
        best_size = bin(best_mask).count('1')
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")

import math
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 55.0  # 5 second margin

    # 1. Precompute LCM and Weights
    # Denominators are 2..48
    denoms = list(range(2, 49))
    n = len(denoms)
    
    D = 1
    for d in denoms:
        D = D * d // math.gcd(D, d)
        
    # w[i] is the integer weight corresponding to 1/denoms[i]
    # sum(w[i] for i in selected) < D is equivalent to sum(1/denoms[i]) < 1
    w = [D // d for d in denoms]
    
    # 2. Local Search Setup
    # We want to maximize sum(w[i]) subject to sum(w[i]) < D
    # Deficit = D - sum(w[i])
    
    best_W = -1
    best_sel_set = None
    best_sel_list = None
    
    # Helper to update best
    def update_best(sel_set, total_W):
        nonlocal best_W, best_sel_set, best_sel_list
        if total_W > best_W:
            best_W = total_W
            best_sel_set = sel_set
            best_sel_list = list(sel_set)

    # Heuristic Initial Solution: Greedy
    # Sort indices by weight descending (i.e., by denominator ascending)
    # But we want to pack tightly. 
    # A good initial guess is often a random permutation greedy fill.
    
    def greedy_fill(perm):
        sel = set()
        total = 0
        for i in perm:
            if total + w[i] < D:
                sel.add(i)
                total += w[i]
        return sel, total

    def local_search(sel, total, rng):
        # Returns improved sel, total
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            best_move = None
            best_new_total = total
            
            # 1. Single Add
            for i in range(n):
                if i not in sel:
                    new_t = total + w[i]
                    if new_t < D and new_t > best_new_total:
                        best_new_total = new_t
                        best_move = ('add', i)
            
            # 2. Single Remove (only useful if we are stuck? No, we want to maximize, so removal is only part of a swap)
            # But sometimes removing a large item allows adding two smaller ones that sum to more.
            # That's covered by swaps.
            
            # 3. 1-out 1-in Swap
            sel_list = list(sel)
            for r in sel_list:
                base = total - w[r]
                for a in range(n):
                    if a not in sel:
                        new_t = base + w[a]
                        if new_t < D and new_t > best_new_total:
                            best_new_total = new_t
                            best_move = ('swap', r, a)
            
            # 4. 1-out 2-in Swap
            # This is O(N^2) per removal, so O(N^3) total. 
            # With N=47, 47^3 ~ 100k ops. Fast enough.
            for r in sel_list:
                base = total - w[r]
                for a in range(n):
                    if a not in sel:
                        t1 = base + w[a]
                        if t1 >= D:
                            continue
                        for b in range(n):
                            if b not in sel and b != a:
                                new_t = t1 + w[b]
                                if new_t < D and new_t > best_new_total:
                                    best_new_total = new_t
                                    best_move = ('swap2', r, a, b)
            
            # 5. 2-out 1-in Swap
            L = len(sel_list)
            for i in range(L):
                r1 = sel_list[i]
                for j in range(i + 1, L):
                    r2 = sel_list[j]
                    base = total - w[r1] - w[r2]
                    for a in range(n):
                        if a not in sel:
                            new_t = base + w[a]
                            if new_t < D and new_t > best_new_total:
                                best_new_total = new_t
                                best_move = ('swap21', r1, r2, a)

            if best_move:
                if best_move[0] == 'add':
                    i = best_move[1]
                    sel.add(i)
                elif best_move[0] == 'swap':
                    r, a = best_move[1], best_move[2]
                    sel.remove(r)
                    sel.add(a)
                elif best_move[0] == 'swap2':
                    r, a, b = best_move[1], best_move[2], best_move[3]
                    sel.remove(r)
                    sel.add(a)
                    sel.add(b)
                elif best_move[0] == 'swap21':
                    r1, r2, a = best_move[1], best_move[2], best_move[3]
                    sel.remove(r1)
                    sel.remove(r2)
                    sel.add(a)
                
                total = best_new_total
                improved = True
        
        return sel, total

    # Run multiple restarts
    # Since local search can get stuck in local optima, random restarts help.
    # We have ~55 seconds.
    # Each local search iteration is fast.
    
    rng = random.Random(42)
    
    # First, try a deterministic greedy start (sorted by weight desc)
    perm_sorted = sorted(range(n), key=lambda i: -w[i])
    sel0, total0 = greedy_fill(perm_sorted)
    sel0, total0 = local_search(sel0, total0, rng)
    update_best(sel0, total0)
    
    # Then random restarts
    # Estimate cost: 
    # Local search loop runs until no improvement. 
    # In worst case, many iterations. But usually converges fast.
    # Let's do as many as possible within time limit.
    
    while time.time() < DEADLINE - 5.0: # Keep 5s buffer for final polish
        perm = list(range(n))
        rng.shuffle(perm)
        sel, total = greedy_fill(perm)
        sel, total = local_search(sel, total, rng)
        update_best(sel, total)
        
        # Early exit if we found the theoretical max?
        # Max possible sum is < D.
        # If D - total is very small, we might be done.
        # But we don't know the exact min deficit.
        # However, if we hit a deficit of 1 (i.e., total = D-1), that's likely optimal or near it.
        # Given D is huge, deficit 1 is extremely rare.
        # Let's just run until time limit.

    # Final Polish: 
    # Take the best solution found and try one more round of very careful local search
    # without the "improved" break, just scanning everything once more to ensure no simple move was missed due to ordering.
    if best_sel_set is not None:
        final_sel = set(best_sel_set)
        final_total = best_W
        
        # Check all adds
        for i in range(n):
            if i not in final_sel:
                nt = final_total + w[i]
                if nt < D and nt > final_total:
                    final_sel.add(i)
                    final_total = nt
        
        # Check all swaps (1-1)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            best_move = None
            best_new_total = final_total
            sel_list = list(final_sel)
            for r in sel_list:
                base = final_total - w[r]
                for a in range(n):
                    if a not in final_sel:
                        nt = base + w[a]
                        if nt < D and nt > best_new_total:
                            best_new_total = nt
                            best_move = ('swap', r, a)
            if best_move:
                r, a = best_move[1], best_move[2]
                final_sel.remove(r)
                final_sel.add(a)
                final_total = best_new_total
                changed = True
        
        if final_total > best_W:
            update_best(final_sel, final_total)

    if best_sel_list is None or len(best_sel_list) == 0:
        # Fallback: return index 0 (1/2) which is valid (1/2 < 1)
        return [0]
        
    return best_sel_list

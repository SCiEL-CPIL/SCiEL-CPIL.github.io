import time
import numpy as np
import random

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_A = []
    best_B = []
    best_score = 0
    
    def is_valid(A, B):
        """Verify the sets are valid Sidon sets with disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                if i != j:
                    d = abs(A[i] - A[j])
                    diffs_A.add(d)
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                if i != j:
                    d = abs(B[i] - B[j])
                    diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
            
        return True

    def _can_add(x, sums, diffs_own, diffs_other, elements):
        """Check if x can be added to a set with given state."""
        if (sums >> (2 * x)) & 1:
            return False
        for e in elements:
            if (sums >> (x + e)) & 1:
                return False
            d = abs(x - e)
            if d > 0 and (((diffs_own >> d) & 1) or ((diffs_other >> d) & 1)):
                return False
        return True

    def greedy_build(seed, strategy=0):
        """Run a single randomized greedy construction using bitsets.
        strategy: 0=balanced, 1=fill_A_first, 2=fill_B_first, 3=random_order
        """
        rng = random.Random(seed)
        A = []
        B = []
        
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Determine order based on strategy
            if strategy == 0:  # balanced
                if len(A) < len(B):
                    order = [0, 1]
                elif len(B) < len(A):
                    order = [1, 0]
                else:
                    order = [0, 1] if rng.random() < 0.5 else [1, 0]
            elif strategy == 1:  # fill A first
                if len(A) < MAX_EACH:
                    order = [0, 1]
                else:
                    order = [1, 0]
            elif strategy == 2:  # fill B first
                if len(B) < MAX_EACH:
                    order = [1, 0]
                else:
                    order = [0, 1]
            else:  # random
                order = [0, 1] if rng.random() < 0.5 else [1, 0]
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    if _can_add(x, sums_A, diffs_A, diffs_B, A):
                        A.append(x)
                        for a in A:
                            sums_A |= (1 << (x + a))
                        for a in A:
                            d = abs(x - a)
                            if d > 0:
                                diffs_A |= (1 << d)
                        added = True
                        break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    if _can_add(x, sums_B, diffs_B, diffs_A, B):
                        B.append(x)
                        for b in B:
                            sums_B |= (1 << (x + b))
                        for b in B:
                            d = abs(x - b)
                            if d > 0:
                                diffs_B |= (1 << d)
                        added = True
                        break
        
        return A, B

    def local_search_polish(A, B, max_iters=1000, rng=None):
        """Try to add more elements and try swaps via local search."""
        if rng is None:
            rng = random.Random()
        A = list(A)
        B = list(B)
        if not A or not B:
            return A, B
        
        # Rebuild bitsets from current A, B
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A |= (1 << (A[i] + A[j]))
                if i != j:
                    diffs_A |= (1 << abs(A[i] - A[j]))
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B |= (1 << (B[i] + B[j]))
                if i != j:
                    diffs_B |= (1 << abs(B[i] - B[j]))
        
        in_A = set(A)
        in_B = set(B)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in in_A and x not in in_B]
            if not candidates:
                # If all numbers are used, we can only try swaps
                # But candidates list is for ADD. Swaps use existing elements?
                # No, swaps remove one and add one from 1..N.
                # If candidates is empty, it means A and B cover [1, N].
                # Then we can only try to swap elements between A and B or within sets?
                # Actually, if A and B cover all, we can't add.
                # We can still try to swap: remove x from A, try to add y to A where y is in B?
                # If y is in B, adding to A requires y not in A (true) and diffs/sums ok.
                # But y is already in B. The task says A and B need not be disjoint.
                # So y can be in both.
                # So candidates for adding to A should be [x for x in 1..N if x not in A].
                # Candidates for adding to B should be [x for x in 1..N if x not in B].
                # My current `candidates` list is `not in A and not in B`.
                # This restricts us to only adding elements that are in NEITHER set.
                # This is suboptimal! We should be able to add an element from B to A.
                pass
            
            # Corrected Candidate Logic
            # For adding to A: any x not in A.
            # For adding to B: any x not in B.
            
            # Try adding to A
            if len(A) < MAX_EACH:
                # Pick random candidates not in A
                # To be fast, we can just pick random x in 1..N and check if in A
                for _ in range(20):
                    if time.time() > deadline: break
                    x = rng.randint(1, N)
                    if x in in_A: continue
                    if _can_add(x, sums_A, diffs_A, diffs_B, A):
                        A.append(x)
                        in_A.add(x)
                        for a in A:
                            sums_A |= (1 << (x + a))
                        for a in A:
                            d = abs(x - a)
                            if d > 0:
                                diffs_A |= (1 << d)
                        improved = True
                        break
            
            # Try adding to B
            if not improved and len(B) < MAX_EACH:
                for _ in range(20):
                    if time.time() > deadline: break
                    x = rng.randint(1, N)
                    if x in in_B: continue
                    if _can_add(x, sums_B, diffs_B, diffs_A, B):
                        B.append(x)
                        in_B.add(x)
                        for b in B:
                            sums_B |= (1 << (x + b))
                        for b in B:
                            d = abs(x - b)
                            if d > 0:
                                diffs_B |= (1 << d)
                        improved = True
                        break
            
            # Try swaps: remove one element, add another
            if not improved and (len(A) > 1 or len(B) > 1):
                # Swap in A
                if len(A) > 1:
                    idx = rng.randrange(len(A))
                    removed = A[idx]
                    A.pop(idx)
                    in_A.discard(removed)
                    
                    # Rebuild sums/diffs for A
                    sums_A_new = 0
                    diffs_A_new = 0
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            sums_A_new |= (1 << (A[i] + A[j]))
                            if i != j:
                                diffs_A_new |= (1 << abs(A[i] - A[j]))
                    
                    # Try to find a replacement
                    found = False
                    # Try random replacements
                    for _ in range(50):
                        x = rng.randint(1, N)
                        if x in in_A: continue
                        if _can_add(x, sums_A_new, diffs_A_new, diffs_B, A):
                            A.append(x)
                            in_A.add(x)
                            # Rebuild full state for A
                            sums_A = 0
                            diffs_A = 0
                            for i in range(len(A)):
                                for j in range(i, len(A)):
                                    sums_A |= (1 << (A[i] + A[j]))
                                    if i != j:
                                        diffs_A |= (1 << abs(A[i] - A[j]))
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Restore
                        A.insert(idx, removed)
                        in_A.add(removed)
                        sums_A = 0
                        diffs_A = 0
                        for i in range(len(A)):
                            for j in range(i, len(A)):
                                sums_A |= (1 << (A[i] + A[j]))
                                if i != j:
                                    diffs_A |= (1 << abs(A[i] - A[j]))
                
                # Swap in B
                if not improved and len(B) > 1:
                    idx = rng.randrange(len(B))
                    removed = B[idx]
                    B.pop(idx)
                    in_B.discard(removed)
                    
                    sums_B_new = 0
                    diffs_B_new = 0
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B_new |= (1 << (B[i] + B[j]))
                            if i != j:
                                diffs_B_new |= (1 << abs(B[i] - B[j]))
                    
                    found = False
                    for _ in range(50):
                        x = rng.randint(1, N)
                        if x in in_B: continue
                        if _can_add(x, sums_B_new, diffs_B_new, diffs_A, B):
                            B.append(x)
                            in_B.add(x)
                            sums_B = 0
                            diffs_B = 0
                            for i in range(len(B)):
                                for j in range(i, len(B)):
                                    sums_B |= (1 << (B[i] + B[j]))
                                    if i != j:
                                        diffs_B |= (1 << abs(B[i] - B[j]))
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        B.insert(idx, removed)
                        in_B.add(removed)
                        sums_B = 0
                        diffs_B = 0
                        for i in range(len(B)):
                            for j in range(i, len(B)):
                                sums_B |= (1 << (B[i] + B[j]))
                                if i != j:
                                    diffs_B |= (1 << abs(B[i] - B[j]))
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return A, B

    def perturb(A, B, rng):
        """Randomly perturb the solution to escape local optima."""
        A = list(A)
        B = list(B)
        
        # Remove 1 or 2 random elements from A
        if len(A) > 2:
            k = rng.randint(1, 2)
            for _ in range(k):
                if A:
                    idx = rng.randrange(len(A))
                    A.pop(idx)
        
        # Remove 1 or 2 random elements from B
        if len(B) > 2:
            k = rng.randint(1, 2)
            for _ in range(k):
                if B:
                    idx = rng.randrange(len(B))
                    B.pop(idx)
                    
        return A, B

    # Main Loop
    rng_global = random.Random(42)
    strategies = [0, 0, 0, 1, 2, 3]
    
    # Initial phase: Greedy + Polish
    current_A, current_B = [], []
    
    for i in range(500): # Increased restarts
        if time.time() > deadline:
            break
        
        strategy = strategies[i % len(strategies)]
        
        # If we have a current solution, we can either start fresh or perturb
        if i > 0 and (current_A or current_B):
            # 50% chance to perturb current best, 50% fresh greedy
            if rng_global.random() < 0.5:
                A, B = perturb(current_A, current_B, rng_global)
            else:
                A, B = greedy_build(i, strategy)
        else:
            A, B = greedy_build(i, strategy)
            
        A2, B2 = local_search_polish(A, B, max_iters=200, rng=rng_global)
        
        if is_valid(A2, B2):
            score = len(A2) + len(B2)
            if score > best_score:
                best_score = score
                best_A = A2
                best_B = B2
                current_A, current_B = A2, B2
                
    # Fallback
    if not best_A:
        best_A = [1]
        best_B = [3]
        
    return [best_A, best_B]

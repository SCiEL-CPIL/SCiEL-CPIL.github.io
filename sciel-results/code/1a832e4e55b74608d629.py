"""Rewritten program for two Sidon sets with disjoint difference sets.
Uses degree-based greedy with randomized restarts and local improvement.
"""
import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # Safety margin for 90s limit

def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            s = ai + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
            
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        bi = B[i]
        for j in range(i, len(B)):
            s = bi + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
            
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - ai)
                
    diffsB = set()
    for i in range(len(B)):
        bi = B[i]
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - bi)
                
    if diffsA & diffsB:
        return False
        
    # Check bounds
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
            
    return True

def _greedy_construct(rng, noise_scale=0.5):
    """
    Greedy construction with degree-based heuristic.
    At each step, calculate the 'cost' (conflicts) of adding each remaining candidate
    to A or B. Pick the one with the lowest cost. Add random noise to break ties
    and escape local minima.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()
    
    remaining = set(range(1, N + 1))
    
    while remaining:
        best_score = float('inf')
        best_action = None # (side, element)
        
        # To speed up, we don't evaluate all remaining elements if the set is large.
        # But N=256 is small enough that O(N^2) per step is okay if steps are few.
        # However, we need to be careful with time.
        
        for x in remaining:
            # Check if x can be added to A
            valid_A = True
            # Check sums
            for a in A:
                if (a + x) in sumsA:
                    valid_A = False
                    break
            if valid_A and (x + x) in sumsA:
                valid_A = False
            
            # Check diffs internal
            if valid_A:
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 in diffsA or d2 in diffsA:
                        valid_A = False
                        break
            
            # Check diffs disjoint with B
            if valid_A:
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 in diffsB or d2 in diffsB:
                        valid_A = False
                        break
                        
            score_A = 10000 if not valid_A else 0
            if valid_A:
                # Heuristic: prefer adding to the set that is currently smaller (balance)
                # and maybe penalize if x is "crowded" (many conflicts with existing)
                # A simple proxy: number of conflicts avoided?
                # Let's just use size balance and random noise.
                score_A = len(A) - len(B) + rng.uniform(-noise_scale, noise_scale)
            
            # Check if x can be added to B
            valid_B = True
            for b in B:
                if (b + x) in sumsB:
                    valid_B = False
                    break
            if valid_B and (x + x) in sumsB:
                valid_B = False
                
            if valid_B:
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    if d1 in diffsB or d2 in diffsB:
                        valid_B = False
                        break
                        
            if valid_B:
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    if d1 in diffsA or d2 in diffsA:
                        valid_B = False
                        break
                        
            score_B = 10000 if not valid_B else 0
            if valid_B:
                score_B = len(B) - len(A) + rng.uniform(-noise_scale, noise_scale)
                
            # Pick the best action
            if score_A < best_score:
                best_score = score_A
                best_action = (0, x)
            if score_B < best_score:
                best_score = score_B
                best_action = (1, x)
                
        if best_action is None or best_score >= 10000:
            break
            
        side, x = best_action
        if side == 0:
            A.append(x)
            # Update sumsA
            for a in A[:-1]:
                sumsA.add(a + x)
            sumsA.add(x + x)
            # Update diffsA
            for a in A[:-1]:
                diffsA.add(x - a)
                diffsA.add(a - x)
        else:
            B.append(x)
            for b in B[:-1]:
                sumsB.add(b + x)
            sumsB.add(x + x)
            for b in B[:-1]:
                diffsB.add(x - b)
                diffsB.add(b - x)
                
        remaining.discard(x)
        
    return A, B

def _local_improve(A, B, rng, t0):
    """
    Try to improve the solution by swapping elements.
    """
    A = list(A)
    B = list(B)
    
    # Recompute state
    def get_state(lst):
        sums = set()
        diffs = set()
        for i in range(len(lst)):
            li = lst[i]
            for j in range(i, len(lst)):
                sums.add(li + lst[j])
            for j in range(len(lst)):
                if i != j:
                    diffs.add(lst[j] - li)
        return sums, diffs

    sumsA, diffsA = get_state(A)
    sumsB, diffsB = get_state(B)
    used = set(A) | set(B)
    
    improved = True
    while improved and time.time() - t0 < DEADLINE - 2:
        improved = False
        
        # Try to add any missing element
        for x in range(1, N + 1):
            if x in used:
                continue
                
            # Try A
            valid = True
            for a in A:
                if (a + x) in sumsA or (x + x) in sumsA:
                    valid = False
                    break
                if (x - a) in diffsA or (a - x) in diffsA:
                    valid = False
                    break
            if valid:
                for a in A:
                    if (x - a) in diffsB or (a - x) in diffsB:
                        valid = False
                        break
            if valid:
                A.append(x)
                used.add(x)
                # Update state
                for a in A[:-1]:
                    sumsA.add(a + x)
                sumsA.add(x + x)
                for a in A[:-1]:
                    diffsA.add(x - a)
                    diffsA.add(a - x)
                improved = True
                break # Restart loop to re-evaluate all
                
            # Try B
            valid = True
            for b in B:
                if (b + x) in sumsB or (x + x) in sumsB:
                    valid = False
                    break
                if (x - b) in diffsB or (b - x) in diffsB:
                    valid = False
                    break
            if valid:
                for b in B:
                    if (x - b) in diffsA or (b - x) in diffsA:
                        valid = False
                        break
            if valid:
                B.append(x)
                used.add(x)
                for b in B[:-1]:
                    sumsB.add(b + x)
                sumsB.add(x + x)
                for b in B[:-1]:
                    diffsB.add(x - b)
                    diffsB.add(b - x)
                improved = True
                break # Restart loop
                
        if improved:
            continue
            
        # If no additions, try swaps
        # Pick a random element from A and try to swap with a random element from B or vice versa
        # Or just try to remove one and see if we can add two?
        # Simple swap: remove a from A, add b to A, remove c from B, add d to B? Too complex.
        # Let's try: Remove a from A. Can we add a better element?
        # This is expensive. Let's stick to simple additions and maybe random restarts.
        
        # If stuck, break out of local improvement
        break
        
    return A, B

def construct():
    t0 = time.time()
    best_A, best_B = [], []
    best_size = 0
    
    # Run multiple restarts with different noise levels and seeds
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
            
        seed = 42 + i
        rng = random.Random(seed)
        
        # Vary noise scale to explore different parts of the solution space
        # Early runs: low noise (greedy), Later runs: high noise (exploratory)
        noise_scale = 0.1 + (i / num_restarts) * 2.0
        
        A, B = _greedy_construct(rng, noise_scale)
        
        # Local improvement
        A, B = _local_improve(A, B, rng, t0)
        
        if _validate(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A, best_B = list(A), list(B)
                
                # Early stopping if we hit the theoretical max?
                # Max size is limited by Sidon property and diff disjointness.
                # 128 is the max possible (64+64).
                if best_size == 128:
                    break

    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    if not _validate(best_A, best_B):
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]

import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid_sidon(arr):
        arr_set = set(arr)
        if len(arr_set) != len(arr):
            return False
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(arr):
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                sums.add(a + arr[j])
        return sums

    def is_maximal(arr, sums=None):
        if sums is None:
            sums = get_sums(arr)
        arr_set = set(arr)
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            # Check if x can be added
            # Conflict if (x+a) in sums for any a in arr, or 2x in sums
            addable = True
            for a in arr:
                if (x + a) in sums:
                    addable = False
                    break
            if addable:
                if (2 * x) in sums:
                    addable = False
            if addable:
                return False
        return True

    def greedy_construct(order):
        S = []
        S_set = set()
        sums = set()
        for x in order:
            if x in S_set:
                continue
            # Check if x is addable to S
            addable = True
            for a in S:
                if (x + a) in sums:
                    addable = False
                    break
            if addable:
                if (2 * x) in sums:
                    addable = False
            
            if addable:
                S.append(x)
                S_set.add(x)
                # Update sums
                for a in S[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        
        # Verify it is maximal?
        # The greedy process adds every element that *can* be added in the given order.
        # Does this guarantee maximality?
        # If we iterate through all elements 1..N, and add if possible, 
        # then for any remaining element y, it was not addable when its turn came.
        # However, adding elements later *might* have blocked it? No, adding elements only creates more sums, 
        # which can only *block* further additions, not unblock them.
        # So if y was not addable at step k, it will not be addable at step k+1.
        # Thus, iterating through all elements ensures the final set is maximal.
        
        return S

    def shrink_maximal(S):
        # Try to remove elements one by one.
        # We iterate through the set multiple times until no more removals are possible.
        S = S[:]
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                # Try removing S[i]
                candidate = S[:i] + S[i+1:]
                if not candidate: # Empty set is not maximal (unless N=0, which is not the case)
                    continue
                # Check if candidate is Sidon (it should be, since it's a subset of a Sidon set)
                # Check if candidate is Maximal
                # Optimization: We don't need to check Sidon again if we trust the subset property,
                # but is_maximal does not check Sidon. We assume input S is Sidon.
                # Subsets of Sidon sets are Sidon.
                
                if is_maximal(candidate):
                    S = candidate
                    changed = True
                    break # Restart loop from beginning of new list? Or just continue? 
                    # Breaking ensures we re-evaluate indices correctly if list shrinks.
                    # Actually, just breaking the inner for loop and restarting while loop is safer.
        
        return S

    best_set = None
    best_size = N + 1
    rng = random.Random(42)

    # 1. Try standard greedy orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(rng.sample(range(1, N + 1), N)),
        list(rng.sample(range(1, N + 1), N)),
        list(rng.sample(range(1, N + 1), N)),
    ]
    
    # Generate more random orders if time permits
    while time.time() < deadline - 5.0:
        orders.append(list(rng.sample(range(1, N + 1), N)))
        if len(orders) > 50:
            break

    for order in orders:
        if time.time() > deadline:
            break
        
        S = greedy_construct(order)
        
        # Shrink it
        S_shrunk = shrink_maximal(S)
        
        if len(S_shrunk) < best_size:
            best_size = len(S_shrunk)
            best_set = S_shrunk[:]
            
            # If we get very small, we might stop?
            # Theoretical minimum is around sqrt(N) * constant.
            # For N=256, sqrt(256)=16. Maximal Sidon sets are often larger.
            # If we find size 15, we are doing great.
            if best_size <= 12:
                break

    if best_set is None:
        # Fallback
        S = greedy_construct(list(range(1, N + 1)))
        best_set = shrink_maximal(S)

    return best_set

# EVOLVE-BLOCK-END

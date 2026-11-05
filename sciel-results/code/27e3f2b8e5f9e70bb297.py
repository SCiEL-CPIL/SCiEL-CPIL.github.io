# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = 4096
    target_size = 80
    max_sum = 3 * n  # 12288

    # Bitset for used triple sums
    # We'll use a Python set for flexibility, but optimize the inner loop

    def greedy_build():
        """Greedy construction using efficient collision checking."""
        S = []
        used_sums = set()
        
        for x in range(1, n + 1):
            if len(S) >= target_size:
                break
            
            k = len(S)
            if k == 0:
                S.append(x)
                used_sums.add(3 * x)
                continue
            
            # Check if x can be added without collisions
            # New sums: 3x, 2x+y, x+2y, x+y+z for y,z in S (y<=z)
            valid = True
            
            # 3x
            if 3 * x in used_sums:
                continue
            
            # 2x + y for y in S
            ok = True
            for y in S:
                if 2 * x + y in used_sums:
                    ok = False
                    break
            if not ok:
                continue
            
            # x + 2y for y in S
            for y in S:
                if x + 2 * y in used_sums:
                    ok = False
                    break
            if not ok:
                continue
            
            # x + y + z for y, z in S, y <= z
            for i in range(k):
                for j in range(i, k):
                    if x + S[i] + S[j] in used_sums:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Add x and update used_sums
            S.append(x)
            used_sums.add(3 * x)
            for y in S:
                used_sums.add(2 * x + y)
                used_sums.add(x + 2 * y)
            for i in range(k):
                for j in range(i, k):
                    used_sums.add(x + S[i] + S[j])
        
        return S

    def try_extend(S, used_sums, max_add=10):
        """Try to extend set S by adding elements."""
        added = 0
        for x in range(1, n + 1):
            if len(S) >= target_size or added >= max_add:
                break
            if x in S:
                continue
            
            k = len(S)
            
            # Check collision
            if 3 * x in used_sums:
                continue
            
            ok = True
            for y in S:
                if 2 * x + y in used_sums:
                    ok = False
                    break
            if not ok:
                continue
            
            for y in S:
                if x + 2 * y in used_sums:
                    ok = False
                    break
            if not ok:
                continue
            
            for i in range(k):
                for j in range(i, k):
                    if x + S[i] + S[j] in used_sums:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Add x
            S.append(x)
            used_sums.add(3 * x)
            for y in S[:-1]:  # all existing elements before x
                used_sums.add(2 * x + y)
                used_sums.add(x + 2 * y)
            for i in range(k):
                for j in range(i, k):
                    used_sums.add(x + S[i] + S[j])
            added += 1
        
        return S, used_sums

    def all_triple_sums(S):
        """Compute all triple sums for set S."""
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums

    def local_search_swap(S, max_iterations=2000):
        """Try to improve set S by swapping elements to allow extension."""
        S = sorted(S)
        used_sums = all_triple_sums(S)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Try to extend first
            S, used_sums = try_extend(S, used_sums, max_add=5)
            
            if len(S) >= target_size:
                break
            
            # Try a swap: remove one element, try to add a better one
            i = random.randint(0, len(S) - 1)
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            
            # Try to add a new element
            best_new = None
            for x in range(1, n + 1):
                if x in S_temp:
                    continue
                if x == removed:
                    continue
                
                k = len(S_temp)
                if k == 0:
                    best_new = x
                    break
                
                if 3 * x in used_sums:
                    continue
                
                ok = True
                for y in S_temp:
                    if 2 * x + y in used_sums:
                        ok = False
                        break
                if not ok:
                    continue
                
                for y in S_temp:
                    if x + 2 * y in used_sums:
                        ok = False
                        break
                if not ok:
                    continue
                
                for a in range(k):
                    for b in range(a, k):
                        if x + S_temp[a] + S_temp[b] in used_sums:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                best_new = x
                break
            
            if best_new is not None:
                S_temp.append(best_new)
                S_temp.sort()
                # Check validity
                new_sums = all_triple_sums(S_temp)
                if len(new_sums) == (len(S_temp) * (len(S_temp)+1) * (len(S_temp)+2)) // 6:
                    S = S_temp
                    used_sums = new_sums
                    # Try to extend
                    S, used_sums = try_extend(S, used_sums, max_add=5)
        
        return S

    # Step 1: Greedy build
    best_set = greedy_build()
    best_size = len(best_set)
    
    # Step 2: Try to extend
    used_sums = all_triple_sums(best_set)
    best_set, used_sums = try_extend(best_set, used_sums, max_add=20)
    best_size = len(best_set)
    
    # Step 3: Local search if we haven't reached target
    if best_size < target_size and time.time() < deadline - 30:
        improved = local_search_swap(best_set, max_iterations=3000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Step 4: Final extension attempt
    if len(best_set) < target_size and time.time() < deadline - 5:
        S = sorted(best_set)
        used_sums = all_triple_sums(S)
        S, used_sums = try_extend(S, used_sums, max_add=30)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    return best_set
# EVOLVE-BLOCK-END

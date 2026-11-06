# EVOLVE-BLOCK-START
import time
import random
import math

def _check_sidon(S):
    """Check if all pair sums (with replacement) in S are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _get_diffs(S):
    """Return set of nonzero absolute differences in S."""
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _can_add_to_sidon(S, x, forbidden_diffs):
    """Check if x can be added to Sidon set S without violating constraints."""
    # Check Sidon property with x
    sums = set()
    for s in S:
        if x + x in sums:
            return False
        sums.add(x + x)
        if s + x in sums:
            return False
        sums.add(s + x)
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    
    # Check difference conflicts
    for s in S:
        d = abs(x - s)
        if d > 0 and d in forbidden_diffs:
            return False
    
    return True

def _get_new_diffs(S, x):
    """Get new differences when adding x to S."""
    new_diffs = set()
    for s in S:
        d = abs(x - s)
        if d > 0:
            new_diffs.add(d)
    return new_diffs

def _greedy_build(A, B, diffs_A, diffs_B, order, deadline):
    """Greedy build from current state."""
    for x in order:
        if time.time() > deadline:
            break
        # Try adding to A
        if _can_add_to_sidon(A, x, diffs_B):
            new_diffs = _get_new_diffs(A, x)
            A.append(x)
            diffs_A.update(new_diffs)
        # Try adding to B
        elif _can_add_to_sidon(B, x, diffs_A):
            new_diffs = _get_new_diffs(B, x)
            B.append(x)
            diffs_B.update(new_diffs)
    return A, B

def _local_search(A, B, diffs_A, diffs_B, deadline, max_iter=5000):
    """Local search: try swaps, removals, and insertions to improve."""
    A = list(A)
    B = list(B)
    diffs_A = set(diffs_A)
    diffs_B = set(diffs_B)
    
    universe_set = set(range(1, 257))
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try removing an element from A and adding to B (or vice versa)
        # First, try simple swaps: remove from A, try to add to B
        if A:
            idx = random.randint(0, len(A) - 1)
            x = A[idx]
            A_temp = A[:idx] + A[idx+1:]
            diffs_A_temp = _get_diffs(A_temp)
            
            # Check if removing x from A allows us to add something else
            # Try adding x to B
            if _can_add_to_sidon(B, x, diffs_B):
                new_diffs = _get_new_diffs(B, x)
                # Check no conflict with A_temp's diffs
                if not new_diffs.intersection(diffs_A_temp):
                    B.append(x)
                    diffs_B.update(new_diffs)
                    A = A_temp
                    diffs_A = diffs_A_temp
                    improved = True
        
        # Try removing from B and adding to A
        if not improved and B:
            idx = random.randint(0, len(B) - 1)
            x = B[idx]
            B_temp = B[:idx] + B[idx+1:]
            diffs_B_temp = _get_diffs(B_temp)
            
            if _can_add_to_sidon(A, x, diffs_A):
                new_diffs = _get_new_diffs(A, x)
                if not new_diffs.intersection(diffs_B_temp):
                    A.append(x)
                    diffs_A.update(new_diffs)
                    B = B_temp
                    diffs_B = diffs_B_temp
                    improved = True
        
        # Try inserting a new element from universe
        if not improved:
            # Find elements not in A or B
            candidates = [x for x in range(1, 257) if x not in set(A) and x not in set(B)]
            if candidates:
                x = random.choice(candidates)
                # Try adding to A
                if _can_add_to_sidon(A, x, diffs_B):
                    new_diffs = _get_new_diffs(A, x)
                    if not new_diffs.intersection(diffs_A):
                        A.append(x)
                        diffs_A.update(new_diffs)
                        improved = True
                # Try adding to B
                elif _can_add_to_sidon(B, x, diffs_A):
                    new_diffs = _get_new_diffs(B, x)
                    if not new_diffs.intersection(diffs_B):
                        B.append(x)
                        diffs_B.update(new_diffs)
                        improved = True
        
        # Try swapping two elements between A and B
        if not improved and A and B:
            idx_a = random.randint(0, len(A) - 1)
            idx_b = random.randint(0, len(B) - 1)
            x_a = A[idx_a]
            x_b = B[idx_b]
            
            A_temp = A[:idx_a] + A[idx_a+1:]
            B_temp = B[:idx_b] + B[idx_b+1:]
            diffs_A_temp = _get_diffs(A_temp)
            diffs_B_temp = _get_diffs(B_temp)
            
            # Check if x_b can be added to A_temp and x_a can be added to B_temp
            if _can_add_to_sidon(A_temp, x_b, diffs_B_temp):
                new_diffs_a = _get_new_diffs(A_temp, x_b)
                if not new_diffs_a.intersection(diffs_A_temp) and not new_diffs_a.intersection(diffs_B_temp):
                    if _can_add_to_sidon(B_temp, x_a, diffs_A_temp | new_diffs_a):
                        new_diffs_b = _get_new_diffs(B_temp, x_a)
                        if not new_diffs_b.intersection(diffs_B_temp) and not new_diffs_b.intersection(new_diffs_a):
                            A = A_temp + [x_b]
                            B = B_temp + [x_a]
                            diffs_A = diffs_A_temp | new_diffs_a
                            diffs_B = diffs_B_temp | new_diffs_b
                            improved = True
        
        if not improved:
            # No improvement in this iteration, try a few more random attempts
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Random removal and re-add
                if A:
                    idx = random.randint(0, len(A) - 1)
                    x = A.pop(idx)
                    diffs_A = _get_diffs(A)
                    # Try to add x back to A or B
                    if _can_add_to_sidon(A, x, diffs_B):
                        A.append(x)
                        diffs_A.update(_get_new_diffs(A[:-1], x))
                    elif _can_add_to_sidon(B, x, diffs_A):
                        B.append(x)
                        diffs_B.update(_get_new_diffs(B[:-1], x))
                elif B:
                    idx = random.randint(0, len(B) - 1)
                    x = B.pop(idx)
                    diffs_B = _get_diffs(B)
                    if _can_add_to_sidon(B, x, diffs_A):
                        B.append(x)
                        diffs_B.update(_get_new_diffs(B[:-1], x))
                    elif _can_add_to_sidon(A, x, diffs_B):
                        A.append(x)
                        diffs_A.update(_get_new_diffs(A[:-1], x))
    
    return A, B, diffs_A, diffs_B

def construct():
    start = time.time()
    deadline = start + 75  # safety margin for 90s limit
    
    best_total = 0
    best_A = []
    best_B = []
    best_diffs_A = set()
    best_diffs_B = set()
    
    # Strategy 1: Multiple random restarts with greedy + local search
    for trial in range(150):
        if time.time() > deadline:
            break
        
        # Create random ordering
        universe = list(range(1, 257))
        random.shuffle(universe)
        
        # Greedy build
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        A, B = _greedy_build(A, B, diffs_A, diffs_B, universe, deadline)
        diffs_A = _get_diffs(A)
        diffs_B = _get_diffs(B)
        
        total = len(A) + len(B)
        
        # Local search polish
        A_ls, B_ls, diffs_A_ls, diffs_B_ls = _local_search(A, B, diffs_A, diffs_B, deadline, max_iter=2000)
        total_ls = len(A_ls) + len(B_ls)
        
        if total_ls > total:
            A, B, diffs_A, diffs_B = A_ls, B_ls, diffs_A_ls, diffs_B_ls
            total = total_ls
        
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
            best_diffs_A = set(diffs_A)
            best_diffs_B = set(diffs_B)
    
    # Strategy 2: Difference-pooling approach
    # Pre-assign differences to pools, then build sets within each pool
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Randomly partition differences into two pools
        all_diffs = list(range(1, 256))
        random.shuffle(all_diffs)
        half = len(all_diffs) // 2
        pool_A = set(all_diffs[:half])
        pool_B = set(all_diffs[half:])
        
        # Build A using only differences from pool_A
        A = []
        diffs_A = set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _can_add_to_sidon(A, x, pool_B - pool_A):  # forbidden are diffs not in pool_A
                new_diffs = _get_new_diffs(A, x)
                if new_diffs.issubset(pool_A):
                    A.append(x)
                    diffs_A.update(new_diffs)
        
        # Build B using only differences from pool_B
        B = []
        diffs_B = set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _can_add_to_sidon(B, x, pool_A - pool_B):
                new_diffs = _get_new_diffs(B, x)
                if new_diffs.issubset(pool_B):
                    B.append(x)
                    diffs_B.update(new_diffs)
        
        total = len(A) + len(B)
        
        # Local search
        A_ls, B_ls, diffs_A_ls, diffs_B_ls = _local_search(A, B, diffs_A, diffs_B, deadline, max_iter=1000)
        total_ls = len(A_ls) + len(B_ls)
        
        if total_ls > total:
            A, B, diffs_A, diffs_B = A_ls, B_ls, diffs_A_ls, diffs_B_ls
            total = total_ls
        
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
            best_diffs_A = set(diffs_A)
            best_diffs_B = set(diffs_B)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END

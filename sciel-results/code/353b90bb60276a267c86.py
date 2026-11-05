"""Improved program with proper local search: additions, moves, removals, and restarts."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Helper to check if adding x to S (list) is valid Sidon-wise
    def is_sidon_valid(S, x, sums_set):
        if x + x in sums_set:
            return False
        for s in S:
            if x + s in sums_set:
                return False
        return True

    # Helper to check if adding x to S keeps diffs disjoint from other_diffs
    def is_diff_disjoint_valid(S, x, other_diffs):
        for s in S:
            if (x - s) in other_diffs:
                return False
            if (s - x) in other_diffs:
                return False
        return True

    # Calculate sums set for a list S
    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    # Calculate diffs set for a list S
    def get_diffs(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    # Greedy construction with multiple strategies
    def greedy_construct(strategy=0):
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diffs_a = set()
        diffs_b = set()
        
        if strategy == 0:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif strategy == 1:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            # Prefer smaller numbers first for A, larger for B
            candidates.sort(key=lambda x: random.random())
        elif strategy == 2:
            # Alternate between small and large
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            candidates = sorted(candidates, key=lambda x: random.random())
        elif strategy == 3:
            # Start from middle
            mid = N // 2
            candidates = list(range(mid, N + 1)) + list(range(1, mid))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if is_sidon_valid(A, x, sums_a) and is_diff_disjoint_valid(A, x, diffs_b):
                A.append(x)
                for a in A[:-1]:
                    sums_a.add(x + a)
                    diffs_a.add(x - a)
                    diffs_a.add(a - x)
                sums_a.add(x + x)
                continue
            
            # Try adding to B
            if is_sidon_valid(B, x, sums_b) and is_diff_disjoint_valid(B, x, diffs_a):
                B.append(x)
                for b in B[:-1]:
                    sums_b.add(x + b)
                    diffs_b.add(x - b)
                    diffs_b.add(b - x)
                sums_b.add(x + x)
                continue
        
        return A, B

    # Local search with proper moves
    def local_search(A, B, max_time=20.0):
        sums_a = get_sums(A)
        sums_b = get_sums(B)
        diffs_a = get_diffs(A)
        diffs_b = get_diffs(B)
        used_elements = set(A) | set(B)
        
        best_A = list(A)
        best_B = list(B)
        best_size = len(A) + len(B)
        
        ls_start = time.time()
        temperature = 5.0
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - ls_start < max_time:
            temperature *= cooling
            
            # Check for restart
            if no_improve_count > max_no_improve and temperature < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add an unused element
                unused = [x for x in range(1, N + 1) if x not in used_elements]
                if not unused:
                    continue
                x = random.choice(unused)
                
                # Try adding to A
                if is_sidon_valid(A, x, sums_a) and is_diff_disjoint_valid(A, x, diffs_b):
                    A.append(x)
                    used_elements.add(x)
                    for a in A[:-1]:
                        sums_a.add(x + a)
                        diffs_a.add(x - a)
                        diffs_a.add(a - x)
                    sums_a.add(x + x)
                    no_improve_count = 0
                    if len(A) + len(B) > best_size:
                        best_size = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    continue
                
                # Try adding to B
                if is_sidon_valid(B, x, sums_b) and is_diff_disjoint_valid(B, x, diffs_a):
                    B.append(x)
                    used_elements.add(x)
                    for b in B[:-1]:
                        sums_b.add(x + b)
                        diffs_b.add(x - b)
                        diffs_b.add(b - x)
                    sums_b.add(x + x)
                    no_improve_count = 0
                    if len(A) + len(B) > best_size:
                        best_size = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    continue
                
                no_improve_count += 1
                
            elif move_type < 0.6:
                # Move element from A to B
                if not A:
                    continue
                i = random.randint(0, len(A) - 1)
                x = A[i]
                
                # Check if B U {x} is valid given A \ {x}
                # First, check if x can be added to B with current diffs_a (minus x's contributions)
                # For safety, we'll do a quick check and then recompute if needed
                
                # Check Sidon validity for B U {x}
                if not is_sidon_valid(B, x, sums_b):
                    no_improve_count += 1
                    continue
                
                # Check diff disjointness: new diffs from x to B must not intersect diffs of A\{x}
                # diffs_a contains diffs involving x. We need diffs of A\{x}.
                # For simplicity, check against current diffs_a (which is a superset)
                # This is conservative but safe.
                if not is_diff_disjoint_valid(B, x, diffs_a):
                    no_improve_count += 1
                    continue
                
                # Apply the move: remove x from A, add to B
                A.pop(i)
                used_elements.discard(x)
                
                # Recompute sums and diffs for A
                sums_a = get_sums(A)
                diffs_a = get_diffs(A)
                
                # Now check if x can be added to B with updated diffs_a
                if is_sidon_valid(B, x, sums_b) and is_diff_disjoint_valid(B, x, diffs_a):
                    B.append(x)
                    used_elements.add(x)
                    for b in B[:-1]:
                        sums_b.add(x + b)
                        diffs_b.add(x - b)
                        diffs_b.add(b - x)
                    sums_b.add(x + x)
                    no_improve_count = 0
                    if len(A) + len(B) > best_size:
                        best_size = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                else:
                    # Revert: put x back in A
                    A.append(x)
                    used_elements.add(x)
                    sums_a = get_sums(A)
                    diffs_a = get_diffs(A)
                    no_improve_count += 1
                    
            elif move_type < 0.8:
                # Move element from B to A
                if not B:
                    continue
                i = random.randint(0, len(B) - 1)
                x = B[i]
                
                if not is_sidon_valid(A, x, sums_a):
                    no_improve_count += 1
                    continue
                
                if not is_diff_disjoint_valid(A, x, diffs_b):
                    no_improve_count += 1
                    continue
                
                # Apply the move: remove x from B, add to A
                B.pop(i)
                used_elements.discard(x)
                
                # Recompute sums and diffs for B
                sums_b = get_sums(B)
                diffs_b = get_diffs(B)
                
                # Now check if x can be added to A with updated diffs_b
                if is_sidon_valid(A, x, sums_a) and is_diff_disjoint_valid(A, x, diffs_b):
                    A.append(x)
                    used_elements.add(x)
                    for a in A[:-1]:
                        sums_a.add(x + a)
                        diffs_a.add(x - a)
                        diffs_a.add(a - x)
                    sums_a.add(x + x)
                    no_improve_count = 0
                    if len(A) + len(B) > best_size:
                        best_size = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                else:
                    # Revert: put x back in B
                    B.append(x)
                    used_elements.add(x)
                    sums_b = get_sums(B)
                    diffs_b = get_diffs(B)
                    no_improve_count += 1
                    
            else:
                # Kick: Remove a random element to try to unblock
                if random.random() < 0.3:
                    if A:
                        i = random.randint(0, len(A) - 1)
                        x = A.pop(i)
                        used_elements.discard(x)
                        sums_a = get_sums(A)
                        diffs_a = get_diffs(A)
                    elif B:
                        i = random.randint(0, len(B) - 1)
                        x = B.pop(i)
                        used_elements.discard(x)
                        sums_b = get_sums(B)
                        diffs_b = get_diffs(B)
                    no_improve_count += 1
        
        return best_A, best_B, best_size

    # Main search: multiple restarts with different greedy strategies
    best_A = []
    best_B = []
    best_size = 0
    
    # Try multiple greedy starts with local search
    for restart in range(100):  # Will be limited by time
        if time.time() > deadline - 5.0:
            break
        
        strategy = restart % 4
        A, B = greedy_construct(strategy)
        
        # Run local search with time limit
        ls_time_left = min(15.0, deadline - time.time() - 3.0)
        if ls_time_left < 1.0:
            break
        
        A, B, size = local_search(A, B, max_time=ls_time_left)
        
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
        
        # Early stop if we've hit the theoretical max
        if best_size >= 128:  # 64+64 is the max
            break
    
    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]
    elif not best_A:
        best_A = [1]
    elif not best_B:
        best_B = [1]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END

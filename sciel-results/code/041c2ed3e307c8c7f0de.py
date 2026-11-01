import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a q-ary covering code of length n and radius r.
    Uses Greedy Set Cover initialization followed by Perturb-and-Repair
    local search with random restarts.
    """
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    N = q ** n  # 6561
    start_time = time.time()
    TIME_LIMIT = 115
    
    # Precompute neighbors
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        w = i
        for p in range(n):
            p3 = pow3[p]
            digit = (w // p3) % 3
            base = w - digit * p3
            for new_d in range(3):
                if new_d == digit:
                    continue
                j = base + new_d * p3
                neighbors[i].append(j)
                
    # Convert neighbors to sets for faster intersection/union? 
    # Or just lists. Lists are fine for membership check if we use a boolean array for "in_code".
    
    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    code_set = set()
    in_code = np.zeros(N, dtype=bool)
    
    # Gain calculation: number of uncovered neighbors
    gain = np.zeros(N, dtype=np.int32)
    for c in range(N):
        for j in neighbors[c]:
            if uncovered[j]:
                gain[c] += 1
                
    while np.any(uncovered):
        best_c = np.argmax(gain)
        if gain[best_c] == 0: break
        code_set.add(best_c)
        in_code[best_c] = True
        for j in neighbors[best_c]:
            if uncovered[j]:
                uncovered[j] = False
                for c_prime in neighbors[j]:
                    gain[c_prime] -= 1
        if time.time() - start_time > TIME_LIMIT - 10:
            break

    # Ensure validity
    if not np.all(uncovered == False):
        for i in range(N):
            if uncovered[i]:
                code_set.add(i)
                in_code[i] = True
                for j in neighbors[i]:
                    uncovered[j] = False

    best_code = set(code_set)
    best_size = len(best_code)
    
    # 2. Perturb and Repair
    # We maintain count_cover for the current best code to speed up checks?
    # Or we recompute? N=6561 is small.
    
    # Helper: Compute coverage counts for a given code set
    def get_counts(code_set):
        counts = np.zeros(N, dtype=np.int32)
        for c in code_set:
            for j in neighbors[c]:
                counts[j] += 1
        return counts
        
    # Helper: Clean code (remove redundant)
    def clean_code(code_set):
        counts = get_counts(code_set)
        new_code = set(code_set)
        for c in list(code_set):
            # Check if c is redundant
            # c is redundant if for all j in neighbors[c], counts[j] >= 2
            # Note: counts includes c. So if counts[j] >= 2, there is another word covering j.
            is_redundant = True
            for j in neighbors[c]:
                if counts[j] < 2:
                    is_redundant = False
                    break
            if is_redundant:
                new_code.remove(c)
                # Update counts for removed c
                for j in neighbors[c]:
                    counts[j] -= 1
        return new_code

    # Helper: Repair code (add words to cover uncovered points)
    def repair_code(code_set, counts):
        # Find uncovered points
        uncovered_mask = (counts == 0)
        if not np.any(uncovered_mask):
            return code_set, counts
        
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # Greedy repair
        while len(uncovered_indices) > 0:
            # Find word that covers most uncovered points
            best_c = -1
            max_covered = 0
            # We can't check all N words efficiently without precomputation.
            # But N=6561 is small.
            # Optimization: Only check words that are neighbors of some uncovered point?
            # Let's just iterate all N.
            for c in range(N):
                if c in code_set:
                    continue
                covered = 0
                for j in neighbors[c]:
                    if counts[j] == 0:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_c = c
                    
            if best_c == -1 or max_covered == 0:
                # Should not happen if space is finite
                break
                
            code_set.add(best_c)
            for j in neighbors[best_c]:
                counts[j] += 1
                if counts[j] == 1:
                    # This point is now covered, remove from uncovered list
                    # We are using a numpy array, so we just rely on counts == 0 check
                    pass
            
            # Update uncovered_indices
            # This is inefficient. Let's just recompute uncovered_mask?
            # Or just loop again.
            # Given small N, recomputing is fine.
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            
        return code_set, counts

    # Main Loop
    current_code = set(best_code)
    current_counts = get_counts(current_code)
    
    no_improvement_count = 0
    perturb_k = 1
    
    while time.time() - start_time < TIME_LIMIT - 5:
        # Perturb: Remove k random words
        if len(current_code) < perturb_k:
            perturb_k = 1
        else:
            # Increase perturb_k slowly if stuck
            if no_improvement_count > 50:
                perturb_k = min(len(current_code) // 2, perturb_k + 1)
            elif no_improvement_count > 20:
                perturb_k = max(1, perturb_k - 1)
            
        words_to_remove = random.sample(list(current_code), min(perturb_k, len(current_code)))
        
        # Apply removal
        temp_code = set(current_code)
        temp_counts = current_counts.copy()
        for c in words_to_remove:
            temp_code.remove(c)
            for j in neighbors[c]:
                temp_counts[j] -= 1
                
        # Repair
        temp_code, temp_counts = repair_code(temp_code, temp_counts)
        
        # Clean
        temp_code = clean_code(temp_code)
        
        # Check improvement
        if len(temp_code) < len(current_code):
            current_code = temp_code
            current_counts = get_counts(current_code)
            if len(current_code) < best_size:
                best_code = set(current_code)
                best_size = len(current_code)
                no_improvement_count = 0
            else:
                no_improvement_count = 0 # Reset if we improved current, even if not global best?
                # Actually, if we improved current but not global, we are in a new basin.
        else:
            # Accept if same size? Or reject?
            # If same size, we might have escaped a local minimum for size, but size is same.
            # Let's accept if it's different and we are stuck?
            # For now, reject to keep it simple. Or accept with probability?
            # Let's just reject to focus on size reduction.
            no_improvement_count += 1
            
        # If we are stuck for a long time, restart from best?
        if no_improvement_count > 200:
            current_code = set(best_code)
            current_counts = get_counts(current_code)
            no_improvement_count = 0
            perturb_k = 1

    return [list(itertools.product(range(3), repeat=8))[i] for i in best_code]

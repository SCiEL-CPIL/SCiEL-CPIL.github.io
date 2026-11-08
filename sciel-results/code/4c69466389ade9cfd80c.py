import math
import random
import time

def construct():
    # Set random seed for reproducibility
    random.seed(42)
    
    def calc_score(A):
        if not A or 0 not in A:
            return -1
        n = len(A)
        if n < 2:
            return -1
        max_a = max(A)
        if max_a == 0:
            return -1
        # Compute sumset and diffset
        sum_set = set()
        diff_set = set()
        for i in range(n):
            ai = A[i]
            for j in range(n):
                aj = A[j]
                sum_set.add(ai + aj)
                diff_set.add(ai - aj)
        P = len(sum_set)
        D = len(diff_set)
        if P == 0:
            return -1
        ratio = D / P
        if ratio <= 0:
            return -1
        denom = math.log(2 * max_a + 1)
        if denom <= 0:
            return -1
        return 1.0 + math.log(ratio) / denom

    def is_sidon(A):
        n = len(A)
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d = A[i] - A[j]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def generate_mian_chowla(n):
        """Generates the first n elements of the Mian-Chowla sequence."""
        seq = [0]
        if n == 1:
            return seq
        seq.append(1)
        for k in range(2, n):
            candidate = seq[-1] + 1
            while True:
                # Check if candidate can be added to keep Sidon property
                # Sums
                ok = True
                existing_sums = set()
                for i in range(len(seq)):
                    for j in range(i, len(seq)):
                        existing_sums.add(seq[i] + seq[j])
                
                for i in range(len(seq)):
                    s = seq[i] + candidate
                    if s in existing_sums:
                        ok = False
                        break
                    if ok:
                        existing_sums.add(s)
                
                if not ok:
                    candidate += 1
                    continue
                
                # Diffs
                existing_diffs = set()
                for i in range(len(seq)):
                    for j in range(len(seq)):
                        if i != j:
                            existing_diffs.add(seq[i] - seq[j])
                
                for i in range(len(seq)):
                    d1 = seq[i] - candidate
                    d2 = candidate - seq[i]
                    if d1 in existing_diffs or d2 in existing_diffs:
                        ok = False
                        break
                    if ok:
                        existing_diffs.add(d1)
                        existing_diffs.add(d2)
                
                if ok:
                    seq.append(candidate)
                    break
                else:
                    candidate += 1
        return seq

    best_score = 0.0
    best_set = [0, 1]
    
    # 1. Try Mian-Chowla sequences of various sizes
    max_n = 40
    mc_seq = generate_mian_chowla(max_n)
    for n in range(2, max_n + 1):
        A = mc_seq[:n]
        s = calc_score(A)
        if s > best_score:
            best_score = s
            best_set = A[:]

    # 2. Local Search / Simulated Annealing
    # Start from the best Mian-Chowla set
    current_set = best_set[:]
    current_score = best_score
    
    # Define a neighborhood: replace one element with a random one, 
    # or swap two elements, or add/remove? 
    # Keep size fixed for now, or allow size changes?
    # Let's try to optimize the set of a fixed size n by local search.
    
    def local_search(start_set, iterations=5000, temp_start=1.0, cooling=0.999):
        current = start_set[:]
        c_score = calc_score(current)
        best_local = current[:]
        best_local_score = c_score
        n = len(current)
        
        for i in range(iterations):
            # Generate a neighbor
            # Strategy: pick a random index, replace value with a random value 
            # in [0, 2*max(current)+10]
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            max_val = max(current)
            # New value range: 0 to 2*max_val + 10
            new_val = random.randint(0, 2 * max_val + 10)
            
            if new_val == old_val:
                continue
                
            # Ensure 0 is still in the set
            # If we replace 0, we must ensure 0 remains. 
            # Simplest: don't replace 0. Or if we do, add 0 elsewhere?
            # Let's fix 0 at index 0 for simplicity in local search.
            if idx == 0:
                continue
                
            current[idx] = new_val
            
            # Check if set is valid (distinct elements)
            if len(set(current)) < n:
                current[idx] = old_val
                continue
                
            n_score = calc_score(current)
            
            # Acceptance criterion
            if n_score > c_score or (temp_start > 0 and random.random() < math.exp((n_score - c_score) / temp_start)):
                c_score = n_score
                if c_score > best_local_score:
                    best_local_score = c_score
                    best_local = current[:]
            else:
                current[idx] = old_val
                
            temp_start *= cooling
            
        return best_local, best_local_score

    # Run local search on a few promising sizes
    for n in [10, 15, 20, 25, 30]:
        if n > max_n:
            break
        start_set = mc_seq[:n]
        # Run multiple restarts
        for restart in range(3):
            res_set, res_score = local_search(start_set, iterations=2000, temp_start=0.5, cooling=0.9995)
            if res_score > best_score:
                best_score = res_score
                best_set = res_set

    return best_set

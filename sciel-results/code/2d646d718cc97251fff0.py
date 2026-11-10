"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Denominators are 2, 3, ..., 48
    # Zero-based index i corresponds to denominator d = i + 2
    # We need a nonempty subset of indices such that sum(1/(i+2)) < 1
    # and we want to MINIMIZE the deficit 1 - sum, i.e., MAXIMIZE the sum subject to sum < 1.
    
    best_sum = Fraction(0)
    best_indices = []
    
    def update_best(s, indices):
        nonlocal best_sum, best_indices
        if s < 1 and s > best_sum:
            best_sum = s
            best_indices = list(indices)
            
    def elapsed():
        return time.time() - start_time

    # Step 1: Greedy initial solution
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if selected:
        update_best(total, selected)
    
    # Step 2: Known good solution (1/1806 deficit)
    candidate = [0, 1, 5, 41]
    s = Fraction(1, 2) + Fraction(1, 3) + Fraction(1, 7) + Fraction(1, 43)
    update_best(s, candidate)
    
    # Step 2.5: Known better solution (1/4070 deficit) found via analysis
    candidate2 = [1, 2, 3, 4, 35, 42]
    s2 = Fraction(1, 3) + Fraction(1, 4) + Fraction(1, 5) + Fraction(1, 6) + Fraction(1, 37) + Fraction(1, 44)
    update_best(s2, candidate2)

    # Step 3: Brute force over first 13 denominators (indices 0-12, denoms 2-14)
    # 2^13 = 8192 subsets, very fast
    for mask in range(1, 1 << 13):
        s = Fraction(0)
        indices = []
        for i in range(13):
            if mask & (1 << i):
                s += Fraction(1, i + 2)
                indices.append(i)
        if s < 1:
            update_best(s, indices)
        if elapsed() > DEADLINE:
            return best_indices
    
    # Step 4: Targeted search - base subsets from first 7 denominators + 1 additional
    for size in range(1, 5):
        for combo in combinations(range(7), size):
            base_sum = Fraction(0)
            for i in combo:
                base_sum += Fraction(1, i + 2)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            best_single = None
            best_single_val = Fraction(0)
            for j in range(7, 47):
                if j in combo:
                    continue
                val = Fraction(1, j + 2)
                if base_sum + val < 1:
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single is not None:
                total_sum = base_sum + best_single_val
                indices = list(combo) + [best_single]
                update_best(total_sum, indices)
        if elapsed() > DEADLINE:
            return best_indices

    # Step 5: Targeted search - base subsets from first 7 denominators + 2 additional
    # Focus on base sizes 3 and 4
    for size in range(3, 5):
        for combo in combinations(range(7), size):
            base_sum = Fraction(0)
            for i in combo:
                base_sum += Fraction(1, i + 2)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            
            best_pair_sum = Fraction(0)
            best_pair = None
            
            for k in range(7, 47):
                if k in combo:
                    continue
                val_k = Fraction(1, k + 2)
                if base_sum + val_k >= 1:
                    continue
                rem_target = Fraction(1) - base_sum - val_k
                if rem_target <= 0:
                    continue
                best_l = None
                best_l_val = Fraction(0)
                for l in range(k + 1, 47):
                    if l in combo:
                        continue
                    val_l = Fraction(1, l + 2)
                    if base_sum + val_k + val_l < 1:
                        if val_l > best_l_val:
                            best_l_val = val_l
                            best_l = l
                if best_l is not None:
                    pair_sum = val_k + best_l_val
                    if pair_sum > best_pair_sum:
                        best_pair_sum = pair_sum
                        best_pair = (k, best_l)
            
            if best_pair is not None:
                total_sum = base_sum + best_pair_sum
                indices = list(combo) + list(best_pair)
                update_best(total_sum, indices)
        
        if elapsed() > DEADLINE:
            return best_indices

    # Step 6: Targeted search - base subsets from first 8 denominators + 1 additional
    for size in range(4, 7):
        for combo in combinations(range(8), size):
            base_sum = Fraction(0)
            for i in combo:
                base_sum += Fraction(1, i + 2)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            best_single = None
            best_single_val = Fraction(0)
            for j in range(8, 47):
                if j in combo:
                    continue
                val = Fraction(1, j + 2)
                if base_sum + val < 1:
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single is not None:
                total_sum = base_sum + best_single_val
                indices = list(combo) + [best_single]
                update_best(total_sum, indices)
        if elapsed() > DEADLINE:
            return best_indices

    # Step 7: Local search - try swapping one element in best_indices with another
    if best_indices:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for i in list(current_set):
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = current_sum - Fraction(1, i + 2) + Fraction(1, j + 2)
                    if new_sum < 1 and new_sum > current_sum:
                        new_set = current_set - {i} | {j}
                        new_indices = sorted(new_set)
                        update_best(new_sum, new_indices)
                        current_set = new_set
                        current_sum = new_sum
                        improved = True
                        break
                if elapsed() > DEADLINE:
                    return best_indices
    
    # Step 8: Try adding one more element to the best solution
    if best_indices and elapsed() < DEADLINE:
        current_set = set(best_indices)
        current_sum = best_sum
        for j in range(47):
            if j in current_set:
                continue
            new_sum = current_sum + Fraction(1, j + 2)
            if new_sum < 1:
                new_set = current_set | {j}
                new_indices = sorted(new_set)
                update_best(new_sum, new_indices)
    
    # Step 9: Try removing one element and adding two
    if best_indices and elapsed() < DEADLINE:
        current_set = set(best_indices)
        current_sum = best_sum
        for i in list(current_set):
            base_sum = current_sum - Fraction(1, i + 2)
            base_set = current_set - {i}
            best_pair_sum = Fraction(0)
            best_pair = None
            for k in range(47):
                if k in base_set:
                    continue
                val_k = Fraction(1, k + 2)
                if base_sum + val_k >= 1:
                    continue
                rem = Fraction(1) - base_sum - val_k
                if rem <= 0:
                    continue
                best_l = None
                best_l_val = Fraction(0)
                for l in range(k + 1, 47):
                    if l in base_set:
                        continue
                    val_l = Fraction(1, l + 2)
                    if base_sum + val_k + val_l < 1:
                        if val_l > best_l_val:
                            best_l_val = val_l
                            best_l = l
                if best_l is not None:
                    pair_sum = val_k + best_l_val
                    if pair_sum > best_pair_sum:
                        best_pair_sum = pair_sum
                        best_pair = (k, best_l)
            if best_pair is not None:
                total_sum = base_sum + best_pair_sum
                indices = sorted(base_set | set(best_pair))
                update_best(total_sum, indices)
            if elapsed() > DEADLINE:
                return best_indices

    return best_indices
# EVOLVE-BLOCK-END

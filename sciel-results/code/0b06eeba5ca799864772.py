"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    float_vals = [1.0 / d for d in denominators]
    
    def is_valid(selected):
        """Check if selected indices form a valid solution (sum strictly below 1)."""
        if not selected:
            return False
        total = Fraction(0)
        for i in selected:
            total += fracs[i]
        return total < 1
    
    def get_deficit(selected):
        """Compute the deficit 1 - sum(1/d_i) for selected indices."""
        total = Fraction(0)
        for i in selected:
            total += fracs[i]
        return 1 - total
    
    # Multiple greedy initializations with different orderings
    best_selected = set()
    best_deficit = Fraction(1)  # worst case: empty set would give deficit 1
    
    # Greedy 1: sort by denominator ascending (largest fractions first)
    order1 = list(range(n))
    selected = set()
    total = Fraction(0)
    for i in order1:
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    if is_valid(selected):
        d = get_deficit(selected)
        if d < best_deficit:
            best_deficit = d
            best_selected = set(selected)
    
    # Greedy 2: sort by denominator descending (smallest fractions first)
    order2 = list(range(n - 1, -1, -1))
    selected = set()
    total = Fraction(0)
    for i in order2:
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    if is_valid(selected):
        d = get_deficit(selected)
        if d < best_deficit:
            best_deficit = d
            best_selected = set(selected)
    
    # Greedy 3: random-ish order (using a simple deterministic permutation)
    import random
    rng = random.Random(42)
    order3 = list(range(n))
    rng.shuffle(order3)
    selected = set()
    total = Fraction(0)
    for i in order3:
        if total + fracs[i] < 1:
            selected.add(i)
            total += fracs[i]
    if is_valid(selected):
        d = get_deficit(selected)
        if d < best_deficit:
            best_deficit = d
            best_selected = set(selected)
    
    # Greedy 4: try to get close to 1 by adding fractions that bring us closest
    selected = set()
    total = Fraction(0)
    available = set(range(n))
    while available:
        best_i = None
        best_new_total = total
        for i in available:
            new_total = total + fracs[i]
            if new_total < 1 and new_total > best_new_total:
                best_new_total = new_total
                best_i = i
        if best_i is None:
            break
        selected.add(best_i)
        total = best_new_total
        available.discard(best_i)
    if is_valid(selected):
        d = get_deficit(selected)
        if d < best_deficit:
            best_deficit = d
            best_selected = set(selected)
    
    # Local search: try adding, removing, swapping elements
    # Use float for fast pruning, Fraction for verification
    selected = set(best_selected)
    total_float = sum(float_vals[i] for i in selected)
    total_frac = sum(fracs[i] for i in selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding single elements
        for i in range(n):
            if time.time() > deadline:
                break
            if i not in selected:
                new_float = total_float + float_vals[i]
                if new_float < 0.999999999 and new_float > total_float:
                    new_selected = selected | {i}
                    if is_valid(new_selected):
                        new_deficit = get_deficit(new_selected)
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = new_selected
                            selected = new_selected
                            total_float = new_float
                            total_frac = sum(fracs[i] for i in selected)
                            improved = True
                            break
        
        if time.time() > deadline:
            break
        
        # Try removing single elements (to make room for better additions)
        for rem in list(selected):
            if time.time() > deadline:
                break
            new_selected = selected - {rem}
            new_float = total_float - float_vals[rem]
            # After removal, try adding elements
            for add in range(n):
                if time.time() > deadline:
                    break
                if add not in new_selected:
                    new_float2 = new_float + float_vals[add]
                    if new_float2 < 0.999999999 and new_float2 > total_float:
                        new_selected2 = new_selected | {add}
                        if is_valid(new_selected2):
                            new_deficit = get_deficit(new_selected2)
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                best_selected = new_selected2
                                selected = new_selected2
                                total_float = new_float2
                                total_frac = sum(fracs[i] for i in selected)
                                improved = True
                                break
                if time.time() > deadline:
                    break
            if improved:
                break
        
        if time.time() > deadline:
            break
        
        # Try 2-opt: remove two, add two
        if len(selected) >= 2:
            rem_list = list(selected)
            for rem1_idx in range(len(rem_list)):
                if time.time() > deadline:
                    break
                rem1 = rem_list[rem1_idx]
                for rem2_idx in range(rem1_idx + 1, len(rem_list)):
                    if time.time() > deadline:
                        break
                    rem2 = rem_list[rem2_idx]
                    rem_float = total_float - float_vals[rem1] - float_vals[rem2]
                    rem_set = selected - {rem1, rem2}
                    
                    for add1 in range(n):
                        if time.time() > deadline:
                            break
                        if add1 not in rem_set:
                            for add2 in range(n):
                                if time.time() > deadline:
                                    break
                                if add2 not in rem_set and add2 != add1:
                                    new_float = rem_float + float_vals[add1] + float_vals[add2]
                                    if new_float < 0.999999999 and new_float > total_float:
                                        new_selected = rem_set | {add1, add2}
                                        if is_valid(new_selected):
                                            new_deficit = get_deficit(new_selected)
                                            if new_deficit < best_deficit:
                                                best_deficit = new_deficit
                                                best_selected = new_selected
                                                selected = new_selected
                                                total_float = new_float
                                                total_frac = sum(fracs[i] for i in selected)
                                                improved = True
                                                break
                                if time.time() > deadline:
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final verification
    if not is_valid(best_selected):
        # Fallback: use the first valid greedy solution
        best_selected = {0}  # Just index 0 (denominator 2) is always valid
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END

from fractions import Fraction
import random
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    best_set = set()
    best_deficit = Fraction(1, 1)
    
    current_set = set()
    current_total = Fraction(0, 1)
    for i in range(n):
        if current_total + recips[i] < 1:
            current_set.add(i)
            current_total += recips[i]
    deficit = 1 - current_total
    if deficit < best_deficit:
        best_deficit = deficit
        best_set = set(current_set)
    
    random.seed(42)
    
    def compute_total(s):
        t = Fraction(0, 1)
        for i in s:
            t += recips[i]
        return t
    
    def local_search(start_set, max_iters=2000):
        current_set = set(start_set)
        current_total = compute_total(current_set)
        if current_total >= 1:
            return None, None
        current_deficit = 1 - current_total
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding
            best_add = -1
            best_add_def = current_deficit
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_def = 1 - new_total
                        if new_def < best_add_def:
                            best_add_def = new_def
                            best_add = i
            if best_add >= 0:
                current_set.add(best_add)
                current_total += recips[best_add]
                current_deficit = best_add_def
                improved = True
                continue
            
            # Try best swap
            best_swap_i = -1
            best_swap_j = -1
            best_swap_def = current_deficit
            for i in list(current_set):
                for j in range(n):
                    if j not in current_set:
                        new_total = current_total - recips[i] + recips[j]
                        if new_total < 1:
                            new_def = 1 - new_total
                            if new_def < best_swap_def:
                                best_swap_def = new_def
                                best_swap_i = i
                                best_swap_j = j
            if best_swap_i >= 0:
                current_set.remove(best_swap_i)
                current_set.add(best_swap_j)
                current_total = current_total - recips[best_swap_i] + recips[best_swap_j]
                current_deficit = best_swap_def
                improved = True
        
        if current_deficit < best_deficit:
            return set(current_set), current_deficit
        return None, None
    
    # Simulated annealing style search
    start_time = time.time()
    max_time = 55.0
    
    # First do several local search restarts
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
        if restart == 0:
            current_set = set(best_set)
        else:
            current_set = set(best_set)
            num_changes = random.randint(1, 5)
            for _ in range(num_changes):
                if current_set:
                    in_list = list(current_set)
                    out_list = [i for i in range(n) if i not in current_set]
                    if in_list and out_list:
                        r = random.choice(in_list)
                        a = random.choice(out_list)
                        current_set.remove(r)
                        current_set.add(a)
        
        new_set, new_def = local_search(current_set)
        if new_set is not None:
            best_set = new_set
            best_deficit = new_def
    
    # Now do more aggressive random restarts with local search
    for restart in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Random perturbation
        current_set = set(best_set)
        num_changes = random.randint(1, 8)
        for _ in range(num_changes):
            if current_set:
                in_list = list(current_set)
                out_list = [i for i in range(n) if i not in current_set]
                if in_list and out_list:
                    r = random.choice(in_list)
                    a = random.choice(out_list)
                    current_set.remove(r)
                    current_set.add(a)
        
        new_set, new_def = local_search(current_set, max_iters=1000)
        if new_set is not None:
            best_set = new_set
            best_deficit = new_def
    
    # Final intensive local search
    if time.time() - start_time < max_time:
        new_set, new_def = local_search(best_set, max_iters=5000)
        if new_set is not None:
            best_set = new_set
            best_deficit = new_def
    
    return sorted(best_set)

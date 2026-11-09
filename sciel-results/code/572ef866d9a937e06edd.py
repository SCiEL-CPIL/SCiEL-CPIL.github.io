import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (evaluator kills at 150s)
    
    N = 51975
    base = N
    
    # Generate divisors of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute index arrays for each (m, r) pair
    # idx_cache[(m, r)] = np.array of indices
    idx_cache = {}
    def get_indices(m, r):
        key = (m, r)
        if key not in idx_cache:
            idx_cache[key] = np.arange(r, N, m)
        return idx_cache[key]
    
    # Coverage counts
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    
    classes = []
    used_moduli = set()
    
    def add_class(m, r):
        nonlocal uncovered
        idx = get_indices(m, r)
        dec = np.sum(counts[idx] == 0)
        counts[idx] += 1
        uncovered -= dec
        classes.append([m, r])
        used_moduli.add(m)
    
    def remove_class(m, r):
        nonlocal uncovered
        idx = get_indices(m, r)
        inc = np.sum(counts[idx] == 1)
        counts[idx] -= 1
        uncovered += inc
        classes.remove([m, r])
        used_moduli.remove(m)
    
    def change_residue(i, new_r):
        """Change residue of class i to new_r. Returns delta in uncovered."""
        m, old_r = classes[i]
        if new_r == old_r:
            return 0
        # Remove old
        idx_old = get_indices(m, old_r)
        inc_old = np.sum(counts[idx_old] == 1)
        counts[idx_old] -= 1
        # Add new
        idx_new = get_indices(m, new_r)
        dec_new = np.sum(counts[idx_new] == 0)
        counts[idx_new] += 1
        classes[i][1] = new_r
        # delta = new_uncovered - old_uncovered = -inc_old + dec_new... wait
        # old_uncovered decreases by dec_new (new coverage) and increases by inc_old (lost coverage)
        # Actually: removing old: uncovered += inc_old
        #           adding new: uncovered -= dec_new
        # So delta = inc_old - dec_new
        delta = inc_old - dec_new
        uncovered += delta
        return delta
    
    # Greedy initialization: try multiple orderings
    def greedy_init(order):
        nonlocal counts, uncovered, classes, used_moduli
        counts = np.zeros(N, dtype=np.int32)
        uncovered = N
        classes = []
        used_moduli = set()
        
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = get_indices(m, r)
                gain = np.sum(counts[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                add_class(m, best_r)
        return uncovered
    
    # Try several greedy orderings
    best_uncovered = N
    best_classes = []
    
    # Ordering 1: ascending
    greedy_init(divisors)
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = [c.copy() for c in classes]
    
    # Ordering 2: descending
    greedy_init(divisors[::-1])
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = [c.copy() for c in classes]
    
    # Ordering 3: random shuffles
    for _ in range(5):
        random_order = divisors.copy()
        random.shuffle(random_order)
        greedy_init(random_order)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c.copy() for c in classes]
    
    # Restore best state
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    classes = []
    used_moduli = set()
    for m, r in best_classes:
        idx = get_indices(m, r)
        counts[idx] += 1
    # Recalculate uncovered
    uncovered = np.sum(counts == 0)
    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)
    
    # Tabu Search / Iterated Local Search
    current_uncovered = uncovered
    best_uncovered = uncovered
    best_classes = [c.copy() for c in classes]
    
    tabu_list = {}  # (m, r) -> iteration when it becomes non-tabu
    tabu_tenure = 20
    
    def local_search():
        """Hill climbing: try all single-residue changes, accept best improvement."""
        nonlocal uncovered, classes, current_uncovered
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_new_r = -1
            
            for i in range(len(classes)):
                m, r = classes[i]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Evaluate change
                    idx_old = get_indices(m, r)
                    idx_new = get_indices(m, new_r)
                    # Compute delta without modifying counts
                    # Removing r: residues with count==1 become uncovered (inc)
                    # Adding new_r: residues with count==0 become covered (dec)
                    # But need to be careful about overlap
                    inc = np.sum(counts[idx_old] == 1)
                    dec = np.sum(counts[idx_new] == 0)
                    # If idx_old and idx_new overlap, need adjustment
                    # For same modulus, different residues: no overlap
                    delta = inc - dec
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_new_r = new_r
            
            if best_i >= 0:
                # Apply the best change
                m, r = classes[best_i]
                idx_old = get_indices(m, r)
                idx_new = get_indices(m, best_new_r)
                counts[idx_old] -= 1
                counts[idx_new] += 1
                classes[best_i][1] = best_new_r
                uncovered += best_delta
                current_uncovered = uncovered
                improved = True
        
        return uncovered
    
    # Iterated Local Search with perturbations
    max_iter = 5000
    iteration = 0
    
    while time.time() < deadline - 5 and iteration < max_iter:
        iteration += 1
        
        # Local search
        current_uncovered = local_search()
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c.copy() for c in classes]
        
        if best_uncovered == 0:
            break
        
        # Perturbation: random changes to escape local optimum
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            if not classes:
                break
            i = random.randint(0, len(classes) - 1)
            m, r = classes[i]
            new_r = random.randint(0, m - 1)
            if new_r != r:
                idx_old = get_indices(m, r)
                idx_new = get_indices(m, new_r)
                counts[idx_old] -= 1
                counts[idx_new] += 1
                classes[i][1] = new_r
        
        # Occasionally try adding/removing a class
        if random.random() < 0.2 and len(classes) > 1:
            # Remove a random class
            i = random.randint(0, len(classes) - 1)
            m, r = classes[i]
            idx = get_indices(m, r)
            counts[idx] -= 1
            uncovered += np.sum(counts[idx] == 0)  # This is wrong, need to recalc
            # Actually, let's just recalculate uncovered
            uncovered = np.sum(counts == 0)
            classes.pop(i)
            used_moduli.remove(m)
        elif random.random() < 0.2:
            # Add a random unused class
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                m = random.choice(unused)
                r = random.randint(0, m - 1)
                idx = get_indices(m, r)
                counts[idx] += 1
                uncovered = np.sum(counts == 0)
                classes.append([m, r])
                used_moduli.add(m)
    
    # Final local search on best solution
    counts = np.zeros(N, dtype=np.int32)
    for m, r in best_classes:
        idx = get_indices(m, r)
        counts[idx] += 1
    uncovered = np.sum(counts == 0)
    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)
    
    # More thorough local search
    for _ in range(10):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_delta = 0
            best_i = -1
            best_new_r = -1
            
            for i in range(len(classes)):
                m, r = classes[i]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    idx_old = get_indices(m, r)
                    idx_new = get_indices(m, new_r)
                    inc = np.sum(counts[idx_old] == 1)
                    dec = np.sum(counts[idx_new] == 0)
                    delta = inc - dec
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_new_r = new_r
            
            if best_i >= 0:
                m, r = classes[best_i]
                idx_old = get_indices(m, r)
                idx_new = get_indices(m, best_new_r)
                counts[idx_old] -= 1
                counts[idx_new] += 1
                classes[best_i][1] = best_new_r
                uncovered += best_delta
                improved = True
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_classes = [c.copy() for c in classes]
    
    return best_classes

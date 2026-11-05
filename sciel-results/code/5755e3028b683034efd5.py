import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    base = 51975
    # Find all divisors > 1 of base
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors
    divisors_sorted = sorted(divisors)
    num_divisors = len(divisors_sorted)
    
    # Precompute for each divisor: the list of residue classes (slices)
    # This allows faster access in local search
    residue_slices = {}
    for m in divisors_sorted:
        residue_slices[m] = [slice(r, base, m) for r in range(m)]
    
    def evaluate_greedy(order):
        """
        Greedily assign residues to moduli in the given order.
        Returns (classes, cover_count, uncovered_count)
        """
        classes = []
        cover_count = np.zeros(base, dtype=np.int32)
        uncovered = base
        
        for m in order:
            best_r = 0
            best_gain = -1
            slices = residue_slices[m]
            
            for r in range(m):
                # Count zeros in slice
                gain = np.count_nonzero(cover_count[slices[r]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            classes.append([m, best_r])
            
            # Update cover_count and uncovered
            cover_count[slices[best_r]] += 1
            uncovered -= best_gain
            
        return classes, cover_count, uncovered

    def local_search_fast(classes, cover_count, uncovered):
        """
        Optimized hill climbing local search with incremental filtering.
        """
        improved = True
        max_passes = 50
        pass_count = 0
        
        while improved and pass_count < max_passes and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            
            # Shuffle class order to help escape local minima
            indices = list(range(len(classes)))
            random.shuffle(indices)
            
            for i in indices:
                m, r_old = classes[i]
                best_r = r_old
                best_uncovered = uncovered
                slices = residue_slices[m]
                
                # Precompute old slice
                old_slice = cover_count[slices[r_old]]
                # Count how many have count == 1 (these will become uncovered if we move)
                lose_base = np.count_nonzero(old_slice == 1)
                
                # Try all other residues with incremental filtering
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    
                    new_slice = cover_count[slices[r_new]]
                    
                    # Quick filter: if gain <= lose_base, this move can't improve
                    # (since delta = lose - gain, and we need delta < 0)
                    gain = np.count_nonzero(new_slice == 0)
                    if gain <= lose_base:
                        continue
                    
                    # Full evaluation
                    lose = lose_base
                    delta = lose - gain
                    new_uncovered = uncovered + delta
                    
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new
                
                if best_r != r_old:
                    # Apply change
                    classes[i][1] = best_r
                    cover_count[slices[r_old]] -= 1
                    cover_count[slices[best_r]] += 1
                    uncovered = best_uncovered
                    improved = True
                    
        return classes, cover_count, uncovered

    def local_search_with_restarts(classes, cover_count, uncovered, num_restarts=3):
        """
        Local search with multiple random restarts to escape local minima.
        """
        best_classes = [c[:] for c in classes]
        best_cover_count = cover_count.copy()
        best_uncovered = uncovered
        
        for restart in range(num_restarts):
            # Create a perturbed version
            perturbed_classes = [c[:] for c in classes]
            perturbed_cover_count = cover_count.copy()
            perturbed_uncovered = uncovered
            
            # Randomly perturb a few classes
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                idx = random.randint(0, len(perturbed_classes) - 1)
                m, r_old = perturbed_classes[idx]
                if m > 1:
                    r_new = random.randint(0, m - 1)
                    if r_new != r_old:
                        slices = residue_slices[m]
                        perturbed_cover_count[slices[r_old]] -= 1
                        perturbed_cover_count[slices[r_new]] += 1
                        perturbed_classes[idx][1] = r_new
                        # Recalculate uncovered for this change
                        old_slice = cover_count[slices[r_old]]
                        new_slice = cover_count[slices[r_new]]
                        # This is approximate; full recalculation happens in local search
                        perturbed_uncovered = np.count_nonzero(perturbed_cover_count == 0)
            
            # Run local search on perturbed version
            ls_classes, ls_cover, ls_uncovered = local_search_fast(
                perturbed_classes, perturbed_cover_count, perturbed_uncovered
            )
            
            if ls_uncovered < best_uncovered:
                best_uncovered = ls_uncovered
                best_classes = [c[:] for c in ls_classes]
                best_cover_count = ls_cover.copy()
        
        return best_classes, best_cover_count, best_uncovered

    best_classes = []
    best_uncovered = base + 1
    best_cover_count = None
    
    # Phase 1: Multiple Greedy Initializations
    # 1. Ascending order
    # 2. Descending order
    # 3. Random shuffles
    
    orders = [
        divisors_sorted,
        sorted(divisors_sorted, reverse=True)
    ]
    
    # Add random orders
    random.seed(42)
    for _ in range(15):
        order = divisors_sorted[:]
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
            
        classes, cover_count, uncovered = evaluate_greedy(order)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c[:] for c in classes]
            best_cover_count = cover_count.copy()
        
        # Phase 2: Local Search on this solution
        if time.time() < deadline - 15:
            classes_ls, cover_count_ls, uncovered_ls = local_search_fast(classes, cover_count, uncovered)
            
            if uncovered_ls < best_uncovered:
                best_uncovered = uncovered_ls
                best_classes = [c[:] for c in classes_ls]
                best_cover_count = cover_count_ls.copy()
                
    # Phase 3: More local search with restarts on the global best
    if best_classes and time.time() < deadline - 10:
        best_classes, best_cover_count, best_uncovered = local_search_with_restarts(
            best_classes, best_cover_count, best_uncovered, num_restarts=5
        )

    if not best_classes:
        # Fallback
        best_classes = [[3, 0]]
        
    return best_classes

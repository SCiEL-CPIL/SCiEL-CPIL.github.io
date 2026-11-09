"""Combined program: efficient coverage tracking + multi-restart local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Generate divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    
    # Precompute residue index arrays for each divisor
    residue_indices = {}
    for m in divisors:
        residue_indices[m] = {}
        for r in range(m):
            residue_indices[m][r] = np.arange(r, base, m)
    
    def score_classes(classes):
        """Count uncovered residues."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            idx = residue_indices[m][r]
            covered[idx] = 1
        return base - int(covered.sum())
    
    def greedy_init(order):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                idx = residue_indices[m][r]
                count = int(np.sum(1 - covered[idx]))
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            idx = residue_indices[m][best_r]
            covered[idx] = 1
        return classes
    
    def local_search(classes, time_budget):
        """Local search with efficient delta computation using coverage counts."""
        classes = [c[:] for c in classes]
        n_classes = len(classes)
        
        # Build coverage count array: how many classes cover each position
        cov_count = np.zeros(base, dtype=np.int16)
        for m, r in classes:
            idx = residue_indices[m][r]
            cov_count[idx] += 1
        uncovered = int(np.sum(cov_count == 0))
        
        start = time.time()
        improved = True
        stagnation = 0
        
        while time.time() - start < time_budget and improved:
            improved = False
            stagnation += 1
            
            # Try changing residue for each class
            idx_order = list(range(n_classes))
            random.shuffle(idx_order)
            
            for i in idx_order:
                if time.time() - start >= time_budget:
                    break
                
                m, r = classes[i]
                old_idx = residue_indices[m][r]
                
                # Find best new residue
                best_r = r
                best_uncovered = uncovered
                
                # Try all residues for this modulus (efficient delta computation)
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_idx = residue_indices[m][new_r]
                    
                    # Compute delta:
                    # Positions that become newly covered: in new_idx where cov_count == 0
                    newly_covered = int(np.sum(cov_count[new_idx] == 0))
                    # Positions that become newly uncovered: in old_idx where cov_count == 1
                    newly_uncovered = int(np.sum(cov_count[old_idx] == 1))
                    
                    delta = newly_uncovered - newly_covered
                    candidate_uncovered = uncovered + delta
                    
                    if candidate_uncovered < best_uncovered:
                        best_uncovered = candidate_uncovered
                        best_r = new_r
                
                if best_r != r:
                    # Apply the change
                    old_idx = residue_indices[m][r]
                    new_idx = residue_indices[m][best_r]
                    cov_count[old_idx] -= 1
                    cov_count[new_idx] += 1
                    classes[i] = [m, best_r]
                    uncovered = best_uncovered
                    improved = True
            
            # If no improvement, try random perturbation
            if not improved:
                if stagnation > 3:
                    break
                # Perturb: randomly change 1-3 residues
                n_perturb = random.randint(1, 3)
                for _ in range(n_perturb):
                    i = random.randrange(n_classes)
                    m, r = classes[i]
                    new_r = random.randint(0, m - 1)
                    if new_r == r:
                        continue
                    old_idx = residue_indices[m][r]
                    new_idx = residue_indices[m][new_r]
                    cov_count[old_idx] -= 1
                    cov_count[new_idx] += 1
                    classes[i] = [m, new_r]
                    uncovered = int(np.sum(cov_count == 0))
    
        return classes, uncovered
    
    # Multiple greedy initializations
    best_score = base
    best_classes = None
    rng = random.Random(42)
    
    # 1. Greedy with descending moduli (larger moduli first cover more)
    classes = greedy_init(divisors[::-1])
    score = score_classes(classes)
    if score < best_score:
        best_score = score
        best_classes = [c[:] for c in classes]
    
    # 2. Greedy with ascending moduli
    if time.time() < deadline - 10:
        classes = greedy_init(divisors)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    # 3. Greedy with random orderings (more trials since greedy is fast)
    for trial in range(20):
        if time.time() >= deadline - 30:
            break
        order = divisors[:]
        rng.shuffle(order)
        classes = greedy_init(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 2
        classes, score = local_search(best_classes, remaining * 0.6)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # More restarts if time remains
    for trial in range(30):
        if time.time() >= deadline - 15:
            break
        order = divisors[:]
        rng.shuffle(order)
        classes = greedy_init(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Quick local search
        remaining = deadline - time.time() - 10
        if remaining > 5 and score <= best_score + 200:
            classes, score = local_search(classes, min(remaining * 0.5, 20))
            if score < best_score:
                best_score = score
                best_classes = classes
    
    # Final local search on best
    remaining = deadline - time.time() - 1
    if remaining > 3:
        classes, score = local_search(best_classes, remaining)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    return best_classes
# EVOLVE-BLOCK-END

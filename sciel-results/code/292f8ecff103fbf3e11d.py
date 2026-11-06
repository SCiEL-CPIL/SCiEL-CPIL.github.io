import time
import random
import numpy as np

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    i = 2
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.append(n)
    return sorted(divs)

def construct():
    base = 51975
    divisors = _get_divisors(base)
    n_divs = len(divisors)
    
    # Precompute indices for each modulus
    # For modulus m, the residues covered by class [m, r] are r, r+m, r+2m, ...
    # We need to efficiently compute the score (number of uncovered residues)
    
    # Strategy: 
    # 1. Greedy initialization
    # 2. Local search with efficient delta computation
    # 3. Simulated annealing for escaping local minima
    # 4. Final polish pass
    
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    # Precompute for each divisor m, the list of indices it covers for each residue r
    # Actually, for efficiency, we'll compute on the fly but use numpy where possible
    
    # Initial assignment: greedy
    covered = np.zeros(base, dtype=np.int32)
    residues = {}
    
    # Sort divisors by size descending (larger moduli cover fewer residues, 
    # so assign them first to avoid conflicts)
    sorted_divs = sorted(divisors, reverse=True)
    
    for m in sorted_divs:
        # Find residue r that maximizes coverage of currently uncovered residues
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = np.arange(r, base, m)
            count = np.sum(covered[idx] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        residues[m] = best_r
        covered[np.arange(best_r, base, m)] += 1
    
    score = int(np.sum(covered == 0))
    best_score = score
    best_residues = residues.copy()
    
    # Local search
    # For each modulus, try changing its residue to see if it improves the score
    # Delta computation: when changing residue from r_old to r_new for modulus m:
    # - Positions that were covered only by this class (count==1) and in r_old but not r_new become uncovered (+1 to score)
    # - Positions that were uncovered (count==0) and in r_new but not r_old become covered (-1 to score)
    
    # Precompute index lists for each divisor
    index_lists = {}
    for m in divisors:
        # For each residue r, the indices are r, r+m, r+2m, ...
        # We'll compute the symmetric difference efficiently
        pass
    
    # More efficient approach: for each m, precompute the sets of indices for each r
    # But this might use too much memory. Instead, compute delta on the fly.
    
    # Local search with simulated annealing
    current_temp = 1000.0
    cooling_rate = 0.999
    
    random.seed(42)
    
    while time.time() - start_time < deadline:
        # Pick a random modulus
        m = random.choice(divisors)
        current_r = residues[m]
        
        # Compute delta for changing to each possible residue
        # For efficiency, only consider a subset of residues if m is large
        if m <= 200:
            candidate_rs = range(m)
        else:
            # Sample some residues
            candidate_rs = [current_r] + [random.randint(0, m-1) for _ in range(50)]
        
        best_r = current_r
        best_delta = 0
        
        for r in candidate_rs:
            if r == current_r:
                continue
            
            # Compute delta
            # Old indices: current_r, current_r+m, ...
            # New indices: r, r+m, ...
            # Symmetric difference
            old_idx = np.arange(current_r, base, m)
            new_idx = np.arange(r, base, m)
            
            # Find positions in old_idx not in new_idx (become uncovered if count was 1)
            # Find positions in new_idx not in old_idx (become covered if count was 0)
            
            # Use set operations for efficiency
            old_set = set(old_idx.tolist())
            new_set = set(new_idx.tolist())
            
            removed = old_set - new_set  # These lose coverage
            added = new_set - old_set    # These gain coverage
            
            # For removed positions: if covered[pos] == 1, they become uncovered (score +1)
            # For added positions: if covered[pos] == 0, they become covered (score -1)
            
            if removed:
                removed_arr = np.array(list(removed))
                delta_removed = int(np.sum(covered[removed_arr] == 1))
            else:
                delta_removed = 0
            
            if added:
                added_arr = np.array(list(added))
                delta_added = -int(np.sum(covered[added_arr] == 0))
            else:
                delta_added = 0
            
            delta = delta_removed + delta_added
            
            if delta < best_delta:
                best_delta = delta
                best_r = r
            elif delta == best_delta and random.random() < 0.1:
                # Occasionally accept equal moves to escape plateaus
                best_r = r
        
        # Apply the best move
        if best_r != current_r:
            old_idx = np.arange(current_r, base, m)
            new_idx = np.arange(best_r, base, m)
            covered[old_idx] -= 1
            covered[new_idx] += 1
            score += best_delta
            residues[m] = best_r
            
            if score < best_score:
                best_score = score
                best_residues = residues.copy()
                # Reset temperature on improvement
                current_temp = max(current_temp * 1.1, 1000.0)
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < 1.0:
            current_temp = 1.0
    
    # Final polish: try all residues for each modulus one more time
    while time.time() - start_time < deadline - 5:
        improved = False
        for m in divisors:
            current_r = residues[m]
            best_r = current_r
            best_delta = 0
            
            for r in range(m):
                if r == current_r:
                    continue
                
                old_idx = np.arange(current_r, base, m)
                new_idx = np.arange(r, base, m)
                
                old_set = set(old_idx.tolist())
                new_set = set(new_idx.tolist())
                
                removed = old_set - new_set
                added = new_set - old_set
                
                if removed:
                    removed_arr = np.array(list(removed))
                    delta_removed = int(np.sum(covered[removed_arr] == 1))
                else:
                    delta_removed = 0
                
                if added:
                    added_arr = np.array(list(added))
                    delta_added = -int(np.sum(covered[added_arr] == 0))
                else:
                    delta_added = 0
                
                delta = delta_removed + delta_added
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r
            
            if best_r != current_r:
                old_idx = np.arange(current_r, base, m)
                new_idx = np.arange(best_r, base, m)
                covered[old_idx] -= 1
                covered[new_idx] += 1
                score += best_delta
                residues[m] = best_r
                improved = True
                
                if score < best_score:
                    best_score = score
                    best_residues = residues.copy()
        
        if not improved:
            break
    
    return [[m, best_residues[m]] for m in divisors]

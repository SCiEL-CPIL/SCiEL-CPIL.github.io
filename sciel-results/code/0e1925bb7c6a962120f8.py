import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    # Precompute for each pair (i,j) the set of residues r in [0, m_i) 
    # that are "forbidden" given a residue s in [0, m_j)
    # r is forbidden iff (r - s) % gcd(m_i, m_j) == 0
    # We'll compute this on the fly but cache the gcd values
    
    def is_valid_assignment(assignment, idx, r):
        """Check if assigning residue r to modulus idx is valid."""
        m_i = all_moduli[idx]
        for j, s in assignment.items():
            g = gcd_mat[idx][j]
            if (r - s) % g == 0:
                return False
        return True
    
    def count_valid_residues(assignment, idx):
        """Count how many valid residues exist for modulus idx given current assignment."""
        m_i = all_moduli[idx]
        count = 0
        for r in range(m_i):
            valid = True
            for j, s in assignment.items():
                g = gcd_mat[idx][j]
                if (r - s) % g == 0:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    def get_valid_residues(assignment, idx):
        """Get all valid residues for modulus idx given current assignment."""
        m_i = all_moduli[idx]
        valid_rs = []
        for r in range(m_i):
            valid = True
            for j, s in assignment.items():
                g = gcd_mat[idx][j]
                if (r - s) % g == 0:
                    valid = False
                    break
            if valid:
                valid_rs.append(r)
        return valid_rs
    
    # Greedy solver with MRV heuristic
    def greedy_mrv(order=None, deterministic=True, seed=None):
        """Greedy with MRV: at each step, pick the unassigned modulus with fewest valid residues."""
        if seed is not None:
            random.seed(seed)
        
        assignment = {}
        unassigned = set(range(n))
        
        # If order is provided, use it as a preference but still apply MRV
        # Actually, let's do pure MRV for the greedy phase
        
        while unassigned:
            if time.time() > deadline:
                break
            
            # Find the modulus with minimum valid residues (MRV)
            best_idx = -1
            best_count = float('inf')
            best_valid_rs = None
            
            # If we have a preferred order, we can bias towards it
            candidates = list(unassigned)
            if order is not None:
                # Sort candidates by their position in the preferred order
                order_map = {idx: pos for pos, idx in enumerate(order)}
                candidates.sort(key=lambda x: order_map.get(x, float('inf')))
            
            for idx in candidates:
                m_i = all_moduli[idx]
                valid_rs = []
                for r in range(m_i):
                    valid = True
                    for j, s in assignment.items():
                        g = gcd_mat[idx][j]
                        if (r - s) % g == 0:
                            valid = False
                            break
                    if valid:
                        valid_rs.append(r)
                
                if not valid_rs:
                    return assignment  # Dead end
                
                if len(valid_rs) < best_count:
                    best_count = len(valid_rs)
                    best_idx = idx
                    best_valid_rs = valid_rs
                    if best_count == 1:
                        break
            
            if best_idx == -1:
                break
            
            # Pick a residue
            if deterministic:
                # Pick the residue that minimizes future conflicts
                best_r = best_valid_rs[0]
                best_score = float('inf')
                for r in best_valid_rs:
                    score = 0
                    for k in unassigned:
                        if k == best_idx:
                            continue
                        g = gcd_mat[best_idx][k]
                        score += all_moduli[k] // g
                    if score < best_score:
                        best_score = score
                        best_r = r
                r = best_r
            else:
                r = random.choice(best_valid_rs)
            
            assignment[best_idx] = r
            unassigned.remove(best_idx)
        
        return assignment
    
    # Backtracking solver with MRV and forward checking
    def backtrack(assignment, unassigned, depth=0):
        """Backtracking with MRV heuristic."""
        if time.time() > deadline:
            return assignment
        
        if not unassigned:
            return assignment  # Complete solution found
        
        # MRV: pick the unassigned modulus with fewest valid residues
        best_idx = -1
        best_valid_rs = None
        best_count = float('inf')
        
        for idx in unassigned:
            m_i = all_moduli[idx]
            valid_rs = []
            for r in range(m_i):
                valid = True
                for j, s in assignment.items():
                    g = gcd_mat[idx][j]
                    if (r - s) % g == 0:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if not valid_rs:
                return assignment  # Dead end
            
            if len(valid_rs) < best_count:
                best_count = len(valid_rs)
                best_idx = idx
                best_valid_rs = valid_rs
                if best_count == 1:
                    break
        
        if best_idx == -1:
            return assignment
        
        unassigned.remove(best_idx)
        
        # Try each valid residue
        for r in best_valid_rs:
            if time.time() > deadline:
                break
            
            assignment[best_idx] = r
            result = backtrack(assignment, unassigned, depth + 1)
            
            # If we found a complete solution or improved, keep it
            if len(result) > len(assignment) - 1:
                # Check if result is better
                if len(result) >= len(assignment):
                    del assignment[best_idx]
                    unassigned.add(best_idx)
                    return result
            
            del assignment[best_idx]
        
        unassigned.add(best_idx)
        return assignment
    
    # Local search: try to add more classes to a partial solution
    def local_search(assignment):
        """Try to add unassigned moduli to the assignment."""
        assignment = dict(assignment)
        unassigned = set(range(n)) - set(assignment.keys())
        
        improved = True
        while improved:
            improved = False
            for idx in list(unassigned):
                m_i = all_moduli[idx]
                valid_rs = []
                for r in range(m_i):
                    valid = True
                    for j, s in assignment.items():
                        g = gcd_mat[idx][j]
                        if (r - s) % g == 0:
                            valid = False
                            break
                    if valid:
                        valid_rs.append(r)
                
                if valid_rs:
                    # Pick the best residue
                    best_r = valid_rs[0]
                    best_score = float('inf')
                    for r in valid_rs:
                        score = 0
                        for k in unassigned:
                            if k == idx:
                                continue
                            g = gcd_mat[idx][k]
                            score += all_moduli[k] // g
                        if score < best_score:
                            best_score = score
                            best_r = r
                    assignment[idx] = best_r
                    unassigned.remove(idx)
                    improved = True
        
        return assignment
    
    # Try multiple strategies
    best_assignment = {}
    
    # Strategy 1: Greedy with MRV, multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        result = greedy_mrv(deterministic=False, seed=seed)
        if len(result) > len(best_assignment):
            best_assignment = result
    
    # Strategy 2: Greedy with MRV, deterministic
    result = greedy_mrv(deterministic=True)
    if len(result) > len(best_assignment):
        best_assignment = result
    
    # Strategy 3: Local search from greedy solution
    if best_assignment:
        improved = local_search(best_assignment)
        if len(improved) > len(best_assignment):
            best_assignment = improved
    
    # Strategy 4: Backtracking from a smaller initial set
    # Start with a subset and try to extend
    if len(best_assignment) < n:
        # Try backtracking from the current best
        unassigned = set(range(n)) - set(best_assignment.keys())
        if unassigned:
            result = backtrack(dict(best_assignment), unassigned)
            if len(result) > len(best_assignment):
                best_assignment = result
    
    # Strategy 5: Try different initial orderings for greedy
    # Primes first
    def is_prime(m):
        if m < 2:
            return False
        for d in range(2, int(m**0.5) + 1):
            if m % d == 0:
                return False
        return True
    
    primes = [i for i in range(n) if is_prime(all_moduli[i])]
    composites = [i for i in range(n) if not is_prime(all_moduli[i])]
    
    # Sort composites by number of divisors
    def num_divisors(m):
        count = 0
        for d in range(1, int(m**0.5) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    
    composites.sort(key=lambda i: (num_divisors(all_moduli[i]), all_moduli[i]))
    priority_order = primes + composites
    
    result = greedy_mrv(order=priority_order, deterministic=True)
    if len(result) > len(best_assignment):
        best_assignment = result
    
    # Try reverse order
    reverse_order = list(reversed(priority_order))
    result = greedy_mrv(order=reverse_order, deterministic=True)
    if len(result) > len(best_assignment):
        best_assignment = result
    
    # Natural order
    result = greedy_mrv(order=list(range(n)), deterministic=True)
    if len(result) > len(best_assignment):
        best_assignment = result
    
    # More randomized restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < 40:
        for restart in range(100):
            if time.time() > deadline:
                break
            result = greedy_mrv(deterministic=False, seed=restart * 7 + 3)
            if len(result) > len(best_assignment):
                best_assignment = result
                # Try local search from this new best
                improved = local_search(best_assignment)
                if len(improved) > len(best_assignment):
                    best_assignment = improved
    
    # Final local search
    if best_assignment:
        improved = local_search(best_assignment)
        if len(improved) > len(best_assignment):
            best_assignment = improved
    
    # Convert to result format
    if best_assignment:
        result = [[best_assignment[idx], all_moduli[idx]] for idx in best_assignment]
    else:
        result = [[0, 2]]
    
    return result

# EVOLVE-BLOCK-END

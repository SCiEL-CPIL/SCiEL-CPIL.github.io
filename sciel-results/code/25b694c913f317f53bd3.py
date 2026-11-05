import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute gcd matrix
    # Index 0 corresponds to m=20, index 1 to m=21, etc.
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
            
    def get_score(selected):
        return sum(1.0 / m for m, r in selected)
        
    def is_compatible_idx(idx_m, r, selected_idx_res):
        """
        selected_idx_res is a list of (idx_prime, r_prime)
        """
        m = moduli[idx_m]
        for idx_prime, r_prime in selected_idx_res:
            g = gcd_mat[idx_m][idx_prime]
            if r % g == r_prime % g:
                return False
        return True
        
    def greedy_fill(seed):
        random.seed(seed)
        selected_idx_res = []
        
        for idx_m in range(n_mods):
            m = moduli[idx_m]
            
            # Find a valid residue
            # To make it random, we can try residues in a random order
            # Or just pick a random r and check. If fail, pick another.
            # Efficient way: collect valids? No, expensive.
            # Just iterate 0..m-1 and pick the first valid? Deterministic.
            # To get diversity, we can shuffle the list of residues?
            # m is small (max 200). Shuffling is cheap.
            
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if is_compatible_idx(idx_m, r, selected_idx_res):
                    selected_idx_res.append((idx_m, r))
                    found = True
                    break
                    
            # If not found, skip
            
        # Convert back to (m, r)
        result = []
        for idx_m, r in selected_idx_res:
            result.append([moduli[idx_m], r])
        return result

    best_solution = []
    best_score = -1
    
    # Run restarts
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        sol = greedy_fill(i + 1)
        score = get_score(sol)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
    if not best_solution:
        return [[20 * j, j - 1] for j in range(1, 11)]
        
    return best_solution

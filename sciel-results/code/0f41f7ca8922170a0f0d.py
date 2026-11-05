import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    moduli = list(range(20, 201))
    
    # Precompute gcds
    size = 181
    gcd_cache = [[0]*size for _ in range(size)]
    for i in range(size):
        m1 = i + 20
        for j in range(size):
            m2 = j + 20
            gcd_cache[i][j] = math.gcd(m1, m2)
            
    def get_score(selected):
        return sum(1.0/m for m, _ in selected)

    def try_add_missing(sol):
        """
        Greedily add missing moduli to sol.
        Returns True if any modulus was added.
        """
        added_any = False
        # Iterate moduli in ascending order to prioritize high value
        for m_new in moduli:
            # Check if m_new is already in sol
            if any(m == m_new for m, _ in sol):
                continue
                
            # Determine forbidden residues for m_new
            # forbidden[r] = True if r is not allowed
            forbidden = [False] * m_new
            
            for m, r in sol:
                g = gcd_cache[m-20][m_new-20]
                # Mark r, r+g, r+2g... as forbidden
                for k in range(r, m_new, g):
                    forbidden[k] = True
                    
            # Find a valid residue
            valid_r = -1
            for r in range(m_new):
                if not forbidden[r]:
                    valid_r = r
                    break
            
            if valid_r != -1:
                sol.append((m_new, valid_r))
                added_any = True
                
        return added_any

    # 1. Greedy Initialization
    initial_sol = []
    try_add_missing(initial_sol)
    
    best_solution = [tuple(x) for x in initial_sol]
    best_score = get_score(best_solution)
    
    # 2. Local Search
    current_solution = [tuple(x) for x in best_solution]
    current_score = best_score
    
    # Helper to update best
    def update_best(sol, score):
        nonlocal best_solution, best_score
        if score > best_score:
            best_score = score
            best_solution = [tuple(x) for x in sol]

    # Local Search Loop
    # We will use a simple hill climbing with random perturbations.
    # Moves:
    # 1. Change residue of a random modulus, then re-pack.
    # 2. Remove a random modulus, then re-pack.
    
    no_improve_count = 0
    max_no_improve = 1000 # Reset perturbation strength?
    
    while time.time() < deadline - 5:
        
        # Move 1: Residue Change + Repack
        if current_solution:
            work_sol = [tuple(x) for x in current_solution]
            idx = random.randint(0, len(work_sol)-1)
            m_k, r_k = work_sol[idx]
            
            # Try to find a different valid residue for m_k
            # We can just pick a random residue and check if it's valid against others?
            # Or better: Remove m_k, re-pack, and see if m_k comes back with a different r?
            # No, re-pack is greedy.
            
            # Let's try: Remove m_k, then try to add it back with a specific r?
            # Actually, just removing and re-adding via try_add_missing might give the same r.
            # To force change, we can try all residues? Too slow.
            
            # Alternative: Pick a random residue r_new for m_k.
            # Check if (m_k, r_new) is compatible with rest.
            rest = [x for i, x in enumerate(work_sol) if i != idx]
            
            # Use the forbidden array logic to check compatibility
            # Actually, just check against rest directly is O(N).
            # Let's do a quick check.
            compatible = True
            for m, r in rest:
                g = gcd_cache[m-20][m_k-20]
                if r_new % g == r % g:
                    compatible = False
                    break
            
            if compatible and r_new != r_k:
                work_sol[idx] = (m_k, r_new)
                # Now try to add missing
                if try_add_missing(work_sol):
                    new_score = get_score(work_sol)
                    if new_score > current_score:
                        current_solution = work_sol
                        current_score = new_score
                        update_best(current_solution, current_score)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        # Move 2: Removal + Repack
        if random.random() < 0.2 and len(current_solution) > 10:
            work_sol = [tuple(x) for x in current_solution]
            rem_idx = random.randint(0, len(work_sol)-1)
            work_sol.pop(rem_idx)
            try_add_missing(work_sol)
            new_score = get_score(work_sol)
            if new_score > current_score:
                current_solution = work_sol
                current_score = new_score
                update_best(current_solution, current_score)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        # If stuck for a while, do a random restart?
        # Or just continue.
        
    return [[m, r] for m, r in best_solution]

if __name__ == "__main__":
    res = construct()
    print(res)
    score = sum(1.0/m for m, r in res)
    print(f"Score: {score}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
    print(f"Valid: {valid}")

import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Map modulus value to index for quick lookup
    m_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute "constraint count" for each modulus:
    # Number of other moduli that share a factor > 1
    constraint_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_tab[i][j] > 1:
                constraint_count[i] += 1

    # Orderings to try
    # 1. Most constrained first (highest constraint_count)
    order_constrained = sorted(range(n), key=lambda i: (-constraint_count[i], moduli[i]))
    # 2. Largest modulus first
    order_desc = sorted(range(n), key=lambda i: -moduli[i])
    # 3. Smallest modulus first
    order_asc = sorted(range(n), key=lambda i: moduli[i])
    # 4. Random orders generated in loop

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_greedy(order, rng):
        selected = []
        for idx in order:
            m = moduli[idx]
            # Try residues in random order to break symmetry and escape local minima
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                if check_residue(idx, r, selected):
                    selected.append((idx, r))
                    break
        return selected

    def local_search(selected, rng, deadline):
        """
        Local search: Try to add unselected moduli by swapping residues of existing ones.
        Returns improved list.
        """
        best = list(selected)
        best_len = len(best)
        selected_idx = set(idx for (idx, _) in best)
        
        # Limit iterations to avoid spending too much time per restart
        max_iters = 500
        
        for _ in range(max_iters):
            if time.time() > deadline - 10: # Keep 10s buffer
                break
                
            improved = False
            
            # Identify candidates to add
            candidates = [idx for idx in range(n) if idx not in selected_idx]
            if not candidates:
                break
            
            rng.shuffle(candidates)
            
            # Try to add one candidate
            for new_idx in candidates[:20]: # Limit candidates to check per iter
                m_new = moduli[new_idx]
                
                # First, try adding with any residue without changing existing
                found_direct = False
                for r_new in range(m_new):
                    if check_residue(new_idx, r_new, best):
                        best.append((new_idx, r_new))
                        selected_idx.add(new_idx)
                        best_len += 1
                        improved = True
                        found_direct = True
                        break
                
                if found_direct:
                    continue # Next candidate
            
                # If direct add failed, try swapping one existing residue
                # We pick a random existing element to modify
                # To be efficient, we only try to modify elements that conflict with new_idx
                # Actually, we can just try to modify a random element and see if it helps
                # A more targeted approach: Find elements that block new_idx.
                
                # For simplicity and speed, we try to change one random existing residue
                # and then re-check if new_idx can be added.
                
                if len(best) == 0:
                    continue
                    
                pos = rng.randint(0, len(best) - 1)
                old_idx, old_r = best[pos]
                old_m = moduli[old_idx]
                
                # Try changing old_r to a new value
                # We don't need to try all, just a few random ones
                for _ in range(10):
                    new_r_old = rng.randint(0, old_m - 1)
                    if new_r_old == old_r:
                        continue
                    
                    # Check if new_r_old is compatible with all OTHERS (excluding pos)
                    ok_old = True
                    for qpos in range(len(best)):
                        if qpos == pos:
                            continue
                        other_idx, other_r = best[qpos]
                        g = gcd_tab[old_idx][other_idx]
                        if (new_r_old - other_r) % g == 0:
                            ok_old = False
                            break
                    
                    if not ok_old:
                        continue
                    
                    # If ok, temporarily update and try to add new_idx
                    best[pos] = (old_idx, new_r_old)
                    
                    found_new = False
                    for r_new in range(m_new):
                        if check_residue(new_idx, r_new, best):
                            best.append((new_idx, r_new))
                            selected_idx.add(new_idx)
                            best_len += 1
                            improved = True
                            found_new = True
                            break
                    
                    if not found_new:
                        # Revert
                        best[pos] = (old_idx, old_r)
                    else:
                        break # Break out of residue loop for this swap attempt
                
                if improved:
                    break # Break out of candidate loop, restart local search loop
            
            if not improved:
                # Optional: Try removing one element and re-adding others?
                # This is complex. For now, if no improvement, we might be stuck.
                # We can break or continue with different random choices.
                # Let's break to save time for next restart if we've tried enough.
                if rng.random() < 0.1:
                    break

        return best

    best_solution = []
    best_len = 0
    
    rng = random.Random(42)
    
    # Phase 1: Deterministic orderings
    for order in [order_constrained, order_desc, order_asc]:
        res = try_greedy(order, rng)
        if len(res) > best_len:
            best_solution = res
            best_len = len(res)
    
    # Phase 2: Random restarts with local search
    restart_count = 0
    max_restarts = 10000
    
    while time.time() < deadline - 15: # Keep 15s buffer
        restart_count += 1
        
        # Generate a random order
        # Sometimes mix constrained and random
        if restart_count % 5 == 0:
            order = list(order_constrained)
            rng.shuffle(order)
        else:
            order = list(range(n))
            rng.shuffle(order)
        
        res = try_greedy(order, rng)
        
        # Apply local search
        res = local_search(res, rng, deadline)
        
        if len(res) > best_len:
            best_solution = res
            best_len = len(res)
            
            # If we reached a very high number, we might be close to optimal
            # Max possible is 127 (all moduli). 
            # If we get > 100, it's good.
            
    # Final polish: Try to add any remaining moduli to the best solution found
    if time.time() < deadline - 2:
        selected_idx = set(idx for (idx, _) in best_solution)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue(idx, r, best_solution):
                        best_solution.append((idx, r))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    return [[r, moduli[idx]] for (idx, r) in best_solution]

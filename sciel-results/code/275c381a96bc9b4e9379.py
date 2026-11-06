import time
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (5s buffer)
    
    # 1. Problem Setup
    # Moduli range [20, 200]
    # We need to select a set of pairs (m, r) such that:
    # - All m are distinct
    # - 0 <= r < m
    # - For any two pairs (m1, r1) and (m2, r2), r1 != r2 mod gcd(m1, m2)
    # Maximize sum(1/m)
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcds
    gcd_cache = [[0]*201 for _ in range(201)]
    for m1 in all_moduli:
        for m2 in all_moduli:
            gcd_cache[m1][m2] = gcd(m1, m2)
            
    # 2. Candidate Generation
    # We cannot consider all (m, r) pairs (sum m ~ 20k nodes) directly in a naive MWIS 
    # because the graph is dense and branch and bound might explode.
    # However, we can prune candidates.
    # For a fixed m, some residues are "better" than others?
    # Actually, the weight depends only on m.
    # The constraint is pairwise.
    
    # Strategy:
    # We will perform a Branch and Bound search over the moduli.
    # State: 
    #   - `selected`: list of (m, r) chosen so far
    #   - `next_modulus_index`: index in the sorted list of moduli we are considering
    #   - `current_score`: sum of 1/m for selected
    
    # To make B&B effective, we need a good upper bound.
    # Upper Bound = current_score + sum(1/m for remaining moduli that are "potentially" compatible).
    # A simple bound is sum(1/m for all remaining moduli). This is loose.
    # A tighter bound: For each remaining modulus m, check if there EXISTS a residue r 
    # such that (m, r) is compatible with ALL selected (m', r').
    # If no such r exists, m cannot be added, so its contribution to the bound is 0.
    # If such r exists, we add 1/m to the bound.
    
    # To speed up the "existence check", we can precompute compatibility or use bitmasks?
    # Since m <= 200, the number of residues is small.
    # Checking existence for one m against K selected moduli takes O(K * m) time.
    # With K ~ 50 and m ~ 200, that's 10k ops per check.
    # If we have 180 moduli, and depth is ~50, this might be too slow for pure Python B&B 
    # if the branching factor is high.
    
    # Alternative: 
    # We can order the moduli by weight (1/m) descending, i.e., m ascending.
    # This is a standard heuristic for B&B.
    
    # Let's refine the Upper Bound calculation:
    # `can_add(m, selected)`: returns True if there is any r in [0, m) such that 
    # r % gcd(m, m_sel) != r_sel % gcd(m, m_sel) for all (m_sel, r_sel) in selected.
    
    def can_add(m, selected):
        # Check if there is a valid residue for m given current selection
        # We can do this by tracking "forbidden" residues modulo each gcd?
        # No, the condition is r != r_sel mod g.
        # This means r cannot be in the set {r_sel mod g + k*g | k=0..m/g-1} for each sel?
        # Actually, r % g == r_sel % g is the conflict.
        # So for each selected (m_sel, r_sel), r is forbidden if r % gcd(m, m_sel) == r_sel % gcd(m, m_sel).
        
        # Let g = gcd(m, m_sel).
        # The forbidden residues for m are those congruent to r_sel modulo g.
        # There are m/g such residues in [0, m).
        # We need to check if the union of these forbidden sets covers all [0, m).
        
        # If m is prime and selected moduli are coprime to m, g=1, forbidden set is {r_sel % 1} = {0}.
        # Wait, if g=1, r % 1 == 0 always. r_sel % 1 == 0 always.
        # So if gcd(m, m_sel)=1, they are ALWAYS compatible?
        # Let's re-read the constraint: "r_1 != r_2 modulo gcd(m_1, m_2)".
        # If gcd=1, r1 % 1 == 0 and r2 % 1 == 0. 0 != 0 is FALSE.
        # So if gcd=1, they are ALWAYS INCOMPATIBLE?
        
        # Let's check the task description carefully.
        # "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))"
        # If gcd(m1, m2) = 1, then r1 % 1 == 0 and r2 % 1 == 0.
        # The condition requires 0 != 0, which is false.
        # Therefore, two moduli with gcd 1 CANNOT both be in the set?
        
        # Let's verify with an example.
        # m1=20, m2=21. gcd=1.
        # Any r1 in [0,19], r2 in [0,20].
        # r1 % 1 == 0. r2 % 1 == 0.
        # Condition: 0 != 0. False.
        # So yes, coprime moduli are incompatible.
        
        # This changes the problem significantly!
        # We can only select moduli that share a common factor > 1 with each other?
        # No, it's pairwise. 
        # If I select m1 and m2, they must have gcd > 1.
        # If I select m1, m2, m3, all pairs must have gcd > 1.
        # This implies the set of moduli must form a clique in the "gcd > 1" graph?
        # Actually, it's stronger. Even if gcd > 1, the residues must be distinct modulo that gcd.
        
        # Let's re-evaluate the "coprime" implication.
        # If gcd(m1, m2) = 1, they are incompatible.
        # So we can only pick moduli from a set where every pair has gcd > 1.
        # This restricts the choice of moduli heavily.
        # For example, we can't pick 20 and 21. We can't pick 20 and 23 (23 is prime).
        # We must pick moduli that are "linked" by shared factors.
        
        # This suggests a different structural approach:
        # Pick a "core" set of prime factors. All selected moduli must be divisible by at least one of these?
        # No, pairwise.
        # Example: 20 (2^2*5), 25 (5^2), 10 (2*5).
        # gcd(20,25)=5. gcd(20,10)=10. gcd(25,10)=5. All > 1.
        # So {20, 25, 10} is a valid candidate set for moduli (before residue check).
        
        # What about {20, 22, 25}?
        # gcd(20,22)=2. gcd(20,25)=5. gcd(22,25)=1.
        # 22 and 25 are coprime, so they can't be together.
        
        # So the problem decomposes into finding a large clique in the graph where edges exist if gcd(u,v) > 1, 
        # AND then assigning residues such that r_i != r_j mod gcd(u,v).
        
        # Since we want to maximize sum(1/m), we prefer small m.
        # Small m are more likely to be composite and share factors.
        
        # Let's implement a B&B that respects the gcd > 1 constraint first.
        
        # Precompute compatibility matrix for moduli (just gcd > 1)
        # compat_moduli[i][j] = (gcd(m_i, m_j) > 1)
        
        # We will search for a subset of moduli that forms a clique in this graph.
        # Then for each such clique, we check if a valid residue assignment exists.
        # Actually, we can integrate the residue check into the B&B.
        
        # Revised B&B State:
        # - `selected_moduli`: list of moduli chosen so far
        # - `selected_residues`: dict m -> r
        # - `next_index`: index in sorted moduli list to consider
        # - `current_score`: sum 1/m
        
        # Upper Bound:
        # Sum of 1/m for all remaining moduli m such that:
        # 1. gcd(m, m_sel) > 1 for all m_sel in selected_moduli
        # 2. There exists a residue r for m compatible with all selected (m_sel, r_sel)
        
        # To make this fast, we can precompute for each m, the list of "conflicting" moduli?
        # No, the residue compatibility depends on the specific residues chosen.
        
        # However, the "gcd > 1" filter is static.
        # Let's precompute `potential_neighbors[m]` = list of moduli m' > m such that gcd(m, m') > 1.
        
        # We sort moduli by m ascending (highest weight first).
        # sorted_moduli = [20, 21, ..., 200]
        
        # B&B Algorithm:
        # def dfs(idx, selected_moduli, selected_residues, current_score):
        #   if time expired: return
        #   if idx == len(sorted_moduli):
        #       update global best
        #       return
        #   
        #   # Calculate Upper Bound
        #   # Iterate from idx to end
        #   # For each m, check if it's compatible with ALL selected_moduli (gcd > 1)
        #   # AND if there is a residue r compatible with all selected_residues.
        #   # If yes, add 1/m to bound.
        #   
        #   # If current_score + bound <= best_score: return (prune)
        
        #   # Option 1: Skip sorted_moduli[idx]
        #   dfs(idx + 1, ...)
        
        #   # Option 2: Take sorted_moduli[idx]
        #   # Try to find a valid residue r for m = sorted_moduli[idx]
        #   # If found, dfs(idx + 1, selected + m, residues + {m: r}, score + 1/m)
        
        # This might be slow if the bound is loose.
        # Let's try to optimize the "residue existence" check.
        # For a fixed m and set of selected (m_s, r_s), 
        # r is invalid if r % gcd(m, m_s) == r_s % gcd(m, m_s) for ANY s.
        # This is a constraint satisfaction problem on r.
        # Since m <= 200, we can just iterate r from 0 to m-1 and check.
        # O(m * |selected|). |selected| is at most ~20-30 (due to gcd constraints).
        # 200 * 30 = 6000 ops.
        # If we do this for ~100 remaining moduli in the bound calculation, that's 600k ops per node.
        # Too slow for deep B&B in Python.
        
        # Optimization:
        # We can precompute for each m, and for each possible "state" of constraints?
        # No, constraints are dynamic.
        
        # Alternative:
        # Use a simpler bound: Sum of 1/m for remaining moduli that satisfy gcd > 1 with all selected.
        # Ignore the residue constraint for the bound (it's an overestimate).
        # This bound is cheaper to compute (just gcd checks).
        # Then, when we actually "Take" a modulus, we do the residue check.
        # If no residue exists, we prune that branch.
        
        # Let's try this.
        
        sorted_moduli = sorted(all_moduli)
        n_moduli = len(sorted_moduli)
        
        # Precompute gcd > 1 matrix
        # g_ok[i][j] = gcd(sorted_moduli[i], sorted_moduli[j]) > 1
        g_ok = [[False]*n_moduli for _ in range(n_moduli)]
        for i in range(n_moduli):
            mi = sorted_moduli[i]
            for j in range(i+1, n_moduli):
                mj = sorted_moduli[j]
                if gcd(mi, mj) > 1:
                    g_ok[i][j] = True
                    g_ok[j][i] = True
        
        best_score = 0.0
        best_solution = []
        
        # To speed up bound calculation, we can maintain a bitmask or list of "active" moduli?
        # Or just loop.
        
        def is_residue_valid(m, r, selected_residues, selected_moduli_list):
            for m_sel, r_sel in zip(selected_moduli_list, selected_residues):
                g = gcd_cache[m][m_sel]
                if r % g == r_sel % g:
                    return False
            return True
            
        def find_valid_residue(m, selected_moduli_list, selected_residues):
            for r in range(m):
                if is_residue_valid(m, r, selected_moduli_list, selected_residues):
                    return r
            return None
            
        def dfs(idx, selected_moduli_list, selected_residues, current_score):
            nonlocal best_score, best_solution
            
            if time.time() > deadline:
                return
                
            if idx == n_moduli:
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(zip(selected_moduli_list, selected_residues))
                return
                
            # Calculate Upper Bound
            # Sum of 1/m for j >= idx such that g_ok[idx..n-1][j] is True for all selected?
            # Wait, the new modulus must be compatible with ALL selected moduli.
            # So for a candidate modulus at index j, we need g_ok[j][k] to be True for all k in selected_indices.
            
            bound = 0.0
            # We can precompute for each index j, a bitmask of selected indices it's compatible with?
            # No, selected is dynamic.
            
            # Simple loop for bound
            # To optimize, we can skip j if it's not compatible with the LAST added modulus?
            # No, it must be compatible with ALL.
            
            # Let's just compute the bound.
            # If the bound is too expensive, we can use a looser bound:
            # Sum of 1/m for all j >= idx.
            # This is very loose but fast.
            # Let's try the "gcd compatible" bound first. If it's too slow, we switch.
            
            # Optimization: Precompute prefix sums of 1/m?
            # sum_1_m[j] = sum(1/m for m in sorted_moduli[j:])
            
            sum_1_m_suffix = [0.0] * (n_moduli + 1)
            for i in range(n_moduli - 1, -1, -1):
                sum_1_m_suffix[i] = sum_1_m_suffix[i+1] + 1.0 / sorted_moduli[i]
            
            # If we use the loose bound:
            # ub = current_score + sum_1_m_suffix[idx]
            # This doesn't prune much.
            
            # Let's use the gcd-compatible bound.
            # For each j from idx to n-1:
            #   Check if sorted_moduli[j] is compatible (gcd>1) with all in selected_moduli_list.
            #   If yes, add 1/m.
            
            # To make this fast, we can maintain a set of "blocked" moduli?
            # No, let's just loop. 
            # Max 180 moduli. 180 * 30 (avg selected) = 5400 gcd checks.
            # gcd is fast. 5400 * 1000 nodes = 5.4M ops. Should be ok.
            
            for j in range(idx, n_moduli):
                m_j = sorted_moduli[j]
                compatible = True
                for m_sel in selected_moduli_list:
                    if not g_ok[sorted_moduli.index(m_sel)][j]: # Index lookup is slow
                        compatible = False
                        break
                if compatible:
                    bound += 1.0 / m_j
            
            if current_score + bound <= best_score:
                return
                
            m_cur = sorted_moduli[idx]
            
            # Option 1: Skip m_cur
            dfs(idx + 1, selected_moduli_list, selected_residues, current_score)
            
            # Option 2: Take m_cur
            # Check if there is a valid residue
            r_val = find_valid_residue(m_cur, selected_moduli_list, selected_residues)
            if r_val is not None:
                new_moduli = selected_moduli_list + [m_cur]
                new_residues = selected_residues + [r_val]
                dfs(idx + 1, new_moduli, new_residues, current_score + 1.0/m_cur)
                
        # The index lookup `sorted_moduli.index(m_sel)` is O(N). 
        # Let's pass indices instead of values to speed up g_ok access.
        
        best_score = 0.0
        best_solution = []
        
        def dfs_idx(idx, selected_indices, selected_residues, current_score):
            nonlocal best_score, best_solution
            
            if time.time() > deadline:
                return
                
            if idx == n_moduli:
                if current_score > best_score:
                    best_score = current_score
                    # Reconstruct solution
                    sol = []
                    for k, r in zip(selected_indices, selected_residues):
                        sol.append((sorted_moduli[k], r))
                    best_solution = sol
                return
                
            # Calculate Upper Bound
            bound = 0.0
            for j in range(idx, n_moduli):
                compatible = True
                for k in selected_indices:
                    if not g_ok[k][j]:
                        compatible = False
                        break
                if compatible:
                    bound += 1.0 / sorted_moduli[j]
                    
            if current_score + bound <= best_score:
                return
                
            # Option 1: Skip
            dfs_idx(idx + 1, selected_indices, selected_residues, current_score)
            
            # Option 2: Take
            m_cur = sorted_moduli[idx]
            # Check residue validity
            # We need to check against selected_moduli values
            # We can precompute the list of selected moduli values to avoid mapping back and forth
            # Or just use the indices to get values.
            
            # To speed up residue check, we can pass the list of (m, r) for selected.
            # But that creates copies.
            # Let's stick to indices and map when checking residue.
            
            valid_r = None
            for r in range(m_cur):
                ok = True
                for k in selected_indices:
                    m_sel = sorted_moduli[k]
                    r_sel = selected_residues[selected_indices.index(k)] # This is O(N^2) and wrong logic
                    # We need to align indices and residues.
                    # selected_indices is a list of indices.
                    # selected_residues is a list of residues corresponding to those indices.
                    # So for the i-th element in selected_indices, the residue is selected_residues[i].
                    pass
                
            # Let's refactor to pass lists of (m, r) for the residue check, 
            # and indices for the g_ok check.
            
        # This is getting messy. Let's simplify.
        # Use lists of moduli and residues for the current path.
        # Use a set of indices for fast g_ok lookup?
        # g_ok is a matrix. Access g_ok[i][j] is O(1).
        # If we pass `selected_indices` as a list, we can iterate it.
        
        # Let's try a cleaner implementation.
        
        def dfs_clean(idx, sel_mods, sel_res, score):
            nonlocal best_score, best_solution
            
            if time.time() > deadline:
                return
                
            if idx == n_moduli:
                if score > best_score:
                    best_score = score
                    best_solution = list(zip(sel_mods, sel_res))
                return
                
            # Bound
            bound = 0.0
            # Precompute which indices in sel_mods correspond to which g_ok rows?
            # We don't have indices in sel_mods.
            # Let's maintain `sel_indices` as well.
            
            pass
        
        # Okay, I will implement the version with `sel_indices` and `sel_res` and `sel_mods` (for residue check).
        
        def dfs_optimized(idx, sel_indices, sel_mods, sel_res, score):
            nonlocal best_score, best_solution
            
            if time.time() > deadline:
                return
                
            if idx == n_moduli:
                if score > best_score:
                    best_score = score
                    best_solution = list(zip(sel_mods, sel_res))
                return
                
            # Calculate Upper Bound
            bound = 0.0
            for j in range(idx, n_moduli):
                compatible = True
                for k in sel_indices:
                    if not g_ok[k][j]:
                        compatible = False
                        break
                if compatible:
                    bound += 1.0 / sorted_moduli[j]
                    
            if score + bound <= best_score:
                return
                
            # Skip
            dfs_optimized(idx + 1, sel_indices, sel_mods, sel_res, score)
            
            # Take
            m_cur = sorted_moduli[idx]
            # Find valid residue
            found_r = None
            for r in range(m_cur):
                valid = True
                for i in range(len(sel_mods)):
                    m_sel = sel_mods[i]
                    r_sel = sel_res[i]
                    g = gcd_cache[m_cur][m_sel]
                    if r % g == r_sel % g:
                        valid = False
                        break
                if valid:
                    found_r = r
                    break
                    
            if found_r is not None:
                dfs_optimized(idx + 1, 
                              sel_indices + [idx], 
                              sel_mods + [m_cur], 
                              sel_res + [found_r], 
                              score + 1.0/m_cur)
                              
        dfs_optimized(0, [], [], [], 0.0)
        
        if not best_solution:
            # Fallback if B&B failed to find anything (unlikely)
            return [[20, 0]]
            
        return [[m, r] for m, r in best_solution]

if __name__ == "__main__":
    res = construct()
    print(f"Selected {len(res)} moduli")
    print(f"Score: {sum(1.0/m for m, r in res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if g == 1:
                print(f"Invalid (gcd=1): {res[i]} and {res[j]}")
                valid = False
            elif r1 % g == r2 % g:
                print(f"Invalid (residue): {res[i]} and {res[j]}")
                valid = False
    if valid:
        print("Valid")

import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # Hard deadline: 115 seconds to be safe against the 120s limit
    deadline = start_time + 115.0
    
    # We want to maximize n.
    # Known lower bound is 34.
    # We will try to find a solution for n=35 first.
    # If that fails, we fall back to n=34.
    
    def solve_hypergraph(n, time_limit):
        """
        Attempts to find a valid 2-coloring for n vertices using WalkSAT.
        Returns a list of colors (0 for Red, 1 for Blue) or None if failed.
        """
        if n < 4:
            # Trivial cases
            triples = list(combinations(range(n), 3))
            return [0] * len(triples)
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Clauses:
        # Red K4 constraint: Not all 4 edges are Red (0).
        # Clause: (e1==1) OR (e2==1) OR (e3==1) OR (e4==1)
        # Blue K5 constraint: Not all 10 edges are Blue (1).
        # Clause: (e1==0) OR ... OR (e10==0)
        
        # We represent a clause as a list of (variable_index, required_value)
        # But for WalkSAT, we just need to know if the clause is satisfied.
        # A clause is satisfied if at least one literal is true.
        # Literal (var, val) is true if colors[var] == val.
        
        clauses = []
        # Map variable to list of clause indices it appears in (for fast update)
        var_to_clauses = [[] for _ in range(num_triples)]
        
        # Add Red K4 clauses
        for k4 in combinations(range(n), 4):
            # The 4 edges in the K4
            edges = []
            for i in range(4):
                # Remove k4[i]
                t = tuple(sorted([v for j, v in enumerate(k4) if j != i]))
                idx = triple_to_idx[t]
                edges.append(idx)
            # Clause: at least one edge must be Blue (1)
            clause = [(idx, 1) for idx in edges]
            clause_id = len(clauses)
            clauses.append(clause)
            for idx, val in clause:
                var_to_clauses[idx].append(clause_id)
                
        # Add Blue K5 clauses
        for k5 in combinations(range(n), 5):
            edges = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                edges.append(idx)
            # Clause: at least one edge must be Red (0)
            clause = [(idx, 0) for idx in edges]
            clause_id = len(clauses)
            clauses.append(clause)
            for idx, val in clause:
                var_to_clauses[idx].append(clause_id)
                
        num_clauses = len(clauses)
        
        # Initialize colors randomly
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Initial clause satisfaction status
        # satisfied[i] = True if clause i is satisfied
        satisfied = np.zeros(num_clauses, dtype=bool)
        # num_satisfied[i] = number of true literals in clause i
        num_satisfied = np.zeros(num_clauses, dtype=np.int32)
        
        for i, clause in enumerate(clauses):
            count = 0
            for idx, val in clause:
                if colors[idx] == val:
                    count += 1
            num_satisfied[i] = count
            if count > 0:
                satisfied[i] = True
                
        unsatisfied_clauses = np.where(~satisfied)[0]
        
        if len(unsatisfied_clauses) == 0:
            return colors.tolist()
            
        # WalkSAT Parameters
        # p is the probability to pick a variable from the clause (good heuristic)
        # vs random variable (to escape local minima)
        # Standard WalkSAT uses a specific clause, then picks a var in it.
        # We can add a random flip probability to escape.
        
        run_deadline = time.time() + time_limit
        
        # To speed up, we keep a list of unsatisfied clause indices
        # We update this list incrementally.
        
        while time.time() < run_deadline:
            if len(unsatisfied_clauses) == 0:
                return colors.tolist()
                
            # Pick a random unsatisfied clause
            clause_idx = unsatisfied_clauses[random.randint(0, len(unsatisfied_clauses) - 1)]
            clause = clauses[clause_idx]
            
            # Pick a variable to flip
            # 80% chance: pick a random variable from the clause
            # 20% chance: pick a random variable from the whole set (to escape)
            if random.random() < 0.8:
                var_idx, req_val = clause[random.randint(0, len(clause) - 1)]
            else:
                var_idx = random.randint(0, num_triples - 1)
                
            # Flip the variable
            old_val = colors[var_idx]
            new_val = 1 - old_val
            colors[var_idx] = new_val
            
            # Update clause satisfaction for all clauses containing var_idx
            # We need to update num_satisfied and satisfied status
            # And update the unsatisfied_clauses list
            
            # We will rebuild the unsatisfied list only if it gets too messy?
            # No, incremental is better. But maintaining a dynamic list with removals is tricky in Python.
            # Alternative: Keep a boolean array `is_unsat` and a list `unsat_list`.
            # When a clause becomes satisfied, remove from list.
            # When a clause becomes unsatisfied, add to list.
            
            # For simplicity and speed in Python, we can just scan the affected clauses 
            # and update a global count, and periodically rebuild the list?
            # Or, we can just keep the list and filter it occasionally?
            # Let's try to update the list directly.
            
            new_unsat_list = []
            # This is O(N_clauses) if we rebuild.
            # Better: Iterate through affected clauses, update their status.
            # Then, if the status changed, adjust the list.
            
            # Since we only flip 1 var, only clauses in var_to_clauses[var_idx] are affected.
            # Let's collect indices of clauses that changed status.
            
            # Optimization: 
            # 1. Update num_satisfied for affected clauses.
            # 2. Check if satisfied status changed.
            # 3. If a clause became satisfied, remove it from unsatisfied_clauses.
            # 4. If a clause became unsatisfied, add it to unsatisfied_clauses.
            
            # To make removal from list fast, we can use a set for unsatisfied clauses?
            # Set membership is O(1). But picking random from set is O(1) in Python 3.7+? 
            # random.choice(list(s)) is O(N). 
            # Let's stick to list, but we accept that removal is O(N) in worst case.
            # Given N_clauses ~ 100k for n=35, O(N) removal is slow if done every step.
            
            # Alternative Strategy: 
            # Don't maintain a perfect list. 
            # Keep a list of *potentially* unsatisfied clauses.
            # When picking, check if it's actually unsatisfied. If yes, process. If no, skip.
            # This is "Lazy" evaluation.
            
            # Let's implement the Lazy approach for speed.
            # unsat_candidates list.
            
            # Initially, unsat_candidates is the list of all unsatisfied clauses.
            # When we flip a var, we don't update the list immediately.
            # We just update the `num_satisfied` array.
            # When we pick a candidate, we check `num_satisfied[cand] == 0`.
            # If it is > 0, it was satisfied by the flip (or earlier flips). We discard it.
            # If it is 0, we process it.
            
            # This avoids list mutation overhead.
            
            # We need to ensure we don't run out of candidates.
            # If len(unsat_candidates) is small, we might need to refill from all clauses?
            # Or just keep the list and filter it if it gets too many "stale" entries.
            
            # Let's refine the loop:
            
            # We already picked clause_idx from the list.
            # But we didn't check if it's still unsatisfied!
            # Wait, the code above picked from `unsatisfied_clauses` which was built at start.
            # We need to check if `num_satisfied[clause_idx] == 0`.
            
            # Let's restructure the inner loop to be Lazy.
            
            # Actually, the code above *already* flipped the variable based on a clause 
            # that *might* be stale. This is fine for WalkSAT.
            # The flip might not help that specific clause if it's already satisfied, 
            # but it might help others.
            
            # However, we must update `num_satisfied` for the affected clauses.
            
            for c_idx in var_to_clauses[var_idx]:
                # Determine how this flip affects clause c_idx
                # We need to know the required value for var_idx in clause c_idx
                # This is slow to look up. 
                # Precompute: clause_literal_req[c_idx][var_idx] = required value?
                # Memory heavy.
                
                # Alternative: Just re-evaluate the clause?
                # Clause size is 4 or 10. Re-evaluating is O(10).
                # This is very fast.
                
                old_sat = num_satisfied[c_idx]
                # Re-evaluate
                new_sat = 0
                for idx, val in clauses[c_idx]:
                    if colors[idx] == val:
                        new_sat += 1
                num_satisfied[c_idx] = new_sat
                
                # If it became satisfied, we don't need to do anything specific to the list 
                # if we are using the lazy approach where we just check `num_satisfied` 
                # when we pick a clause.
                
                # But we need to know if we should add it to the candidate list?
                # No, the candidate list is static initially, or we just use all clauses?
                # If we use all clauses, picking is O(1) but we waste time on satisfied ones.
                # Let's maintain a list of *currently* unsatisfied clauses, but update it lazily.
                
                # If old_sat == 0 and new_sat > 0: It became satisfied.
                # If old_sat > 0 and new_sat == 0: It became unsatisfied.
                
                # We will maintain `unsat_list`.
                # If it became unsatisfied, append to unsat_list.
                # If it became satisfied, we don't remove it immediately (lazy).
                
                if old_sat == 0 and new_sat > 0:
                    pass # It's in the list, but now satisfied. Lazy skip later.
                elif old_sat > 0 and new_sat == 0:
                    unsat_list.append(c_idx)
                    
            # End of update
            
            # Periodically clean the unsat_list to remove satisfied clauses
            # If the list gets too big with stale entries, clean it.
            if len(unsat_list) > 10000:
                unsat_list = [i for i in unsat_list if num_satisfied[i] == 0]
                if len(unsat_list) == 0:
                    return colors.tolist()
            
            # Pick next clause
            if len(unsat_list) == 0:
                # Rebuild from scratch if empty (should mean all satisfied, but check)
                # Check if all are satisfied
                if np.all(num_satisfied > 0):
                    return colors.tolist()
                # Else, some are unsatisfied but not in list? 
                # This can happen if we didn't add them? 
                # We add when they become unsatisfied. Initially we added all unsatisfied.
                # So if list is empty, it should be solved.
                return colors.tolist()
                
            clause_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Check if it's actually unsatisfied (Lazy check)
            if num_satisfied[clause_idx] > 0:
                continue # Stale entry, skip flip? Or flip anyway?
                # Better to skip to focus on real violations.
                # But we already flipped a var in the previous iteration?
                # No, the flip happens *after* picking.
                # So we pick, check if valid, if not, pick another?
                # To keep it simple: 
                # 1. Pick candidate.
                # 2. If satisfied, discard and pick another (loop).
                # 3. If unsatisfied, pick var, flip, update.
            
            # Let's restructure the main loop to handle this properly.
            
        return None # Timeout

    # Refactored WalkSAT with Lazy List
    def solve_walksat(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        clauses = []
        var_to_clauses = [[] for _ in range(num_triples)]
        
        # Red K4: At least one Blue (1)
        for k4 in combinations(range(n), 4):
            edges = []
            for i in range(4):
                t = tuple(sorted([v for j, v in enumerate(k4) if j != i]))
                idx = triple_to_idx[t]
                edges.append(idx)
            clause = [(idx, 1) for idx in edges]
            clause_id = len(clauses)
            clauses.append(clause)
            for idx, val in clause:
                var_to_clauses[idx].append(clause_id)
                
        # Blue K5: At least one Red (0)
        for k5 in combinations(range(n), 5):
            edges = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                edges.append(idx)
            clause = [(idx, 0) for idx in edges]
            clause_id = len(clauses)
            clauses.append(clause)
            for idx, val in clause:
                var_to_clauses[idx].append(clause_id)
                
        num_clauses = len(clauses)
        
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        num_satisfied = np.zeros(num_clauses, dtype=np.int32)
        
        unsat_list = []
        for i, clause in enumerate(clauses):
            count = 0
            for idx, val in clause:
                if colors[idx] == val:
                    count += 1
            num_satisfied[i] = count
            if count == 0:
                unsat_list.append(i)
                
        if len(unsat_list) == 0:
            return colors.tolist()
            
        run_deadline = time.time() + time_limit
        last_clean = time.time()
        
        while time.time() < run_deadline:
            if len(unsat_list) == 0:
                return colors.tolist()
                
            # Pick a clause
            clause_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Lazy check
            if num_satisfied[clause_idx] > 0:
                # It's satisfied now. Remove from list? 
                # We can't remove easily. Just continue to next iteration.
                # To avoid infinite loop on stale entries, we should clean up.
                # But if most are stale, we waste time.
                # Let's clean up if we hit too many stale ones?
                # For now, just continue.
                continue
                
            # Clause is unsatisfied. Pick a variable.
            clause = clauses[clause_idx]
            # 80% from clause, 20% random
            if random.random() < 0.8:
                var_idx, _ = clause[random.randint(0, len(clause) - 1)]
            else:
                var_idx = random.randint(0, num_triples - 1)
                
            # Flip
            colors[var_idx] = 1 - colors[var_idx]
            
            # Update affected clauses
            for c_idx in var_to_clauses[var_idx]:
                old_sat = num_satisfied[c_idx]
                # Re-evaluate
                new_sat = 0
                for idx, val in clauses[c_idx]:
                    if colors[idx] == val:
                        new_sat += 1
                num_satisfied[c_idx] = new_sat
                
                if old_sat == 0 and new_sat > 0:
                    # Became satisfied. (It's in unsat_list, but will be skipped)
                    pass
                elif old_sat > 0 and new_sat == 0:
                    # Became unsatisfied. Add to list.
                    unsat_list.append(c_idx)
                    
            # Clean up if list is too large or periodically
            if time.time() - last_clean > 1.0:
                unsat_list = [i for i in unsat_list if num_satisfied[i] == 0]
                last_clean = time.time()
                if len(unsat_list) == 0:
                    return colors.tolist()
                    
        return None

    # Try n=35 first
    # 100 seconds for n=35
    if time.time() < start_time + 100.0:
        sol_35 = solve_walksat(35, 95.0)
        if sol_35 is not None:
            return {'n': 35, 'colors': sol_35}
            
    # Fallback to n=34
    # Should be very fast
    if time.time() < start_time + 110.0:
        sol_34 = solve_walksat(34, 5.0)
        if sol_34 is not None:
            return {'n': 34, 'colors': sol_34}
            
    # If even 34 fails (unlikely), try random 9
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Result: n={res['n']}, len(colors)={len(res['colors'])}")

import random
import math
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin before 330s kill
    
    # Precompute clause literal arrays as int8: +1 for positive, -1 for negative
    # var_indices: (m, 3) array of variable indices (0-based)
    # signs: (m, 3) array of +1 or -1
    var_indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var_indices[i, j] = abs(lit) - 1
            signs[i, j] = 1 if lit > 0 else -1
    
    # Build variable to clause incidence
    # For each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = var_indices[i, j]
            var_to_clauses[v].append((i, signs[i, j]))
    
    # Convert to numpy arrays for faster access
    # For each variable, store clause indices and signs as flat arrays
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        if var_to_clauses[v]:
            idxs = [c[0] for c in var_to_clauses[v]]
            sns = [c[1] for c in var_to_clauses[v]]
            var_clause_idx.append(np.array(idxs, dtype=np.int32))
            var_clause_sign.append(np.array(sns, dtype=np.int8))
        else:
            var_clause_idx.append(np.array([], dtype=np.int32))
            var_clause_sign.append(np.array([], dtype=np.int8))
    
    # Phase saving: remember last value of each variable
    phase = [False] * n
    
    def evaluate(assignment):
        """Compute number of unsatisfied clauses. assignment is np.array of int8: 1=True, -1=False"""
        # satisfied[i,j] = (assignment[var_indices[i,j]] * signs[i,j]) > 0
        # A clause is unsatisfied if no literal is satisfied
        # satisfied_count[i] = number of satisfied literals in clause i
        # unsat_count = sum(satisfied_count == 0)
        vals = assignment[var_indices]  # (m, 3)
        sat = (vals * signs) > 0  # (m, 3)
        sat_count = sat.sum(axis=1)  # (m,)
        return int((sat_count == 0).sum())
    
    def compute_sat_count(assignment):
        """Compute satisfied literal count per clause"""
        vals = assignment[var_indices]
        sat = (vals * signs) > 0
        return sat.sum(axis=1)
    
    # Luby restart sequence
    def luby(k):
        i = 0
        while (1 << (i + 1)) - 1 < k:
            i += 1
        while True:
            if (1 << i) == k:
                return 1
            if k <= (1 << i) - 1:
                return luby(k - (1 << (i - 1)))
            k -= (1 << i) - 1
            i -= 1
    
    # Main search
    best_energy = float('inf')
    best_assignment = None
    restart_num = 0
    luby_counter = 0
    
    # Parameters
    flip_prob = 0.5  # probability to pick random var in unsat clause
    weight_decay = 0.9999  # clause weight decay
    
    while time.time() < deadline:
        restart_num += 1
        luby_counter += 1
        max_steps = luby(luby_counter) * 100  # scale Luby by 100
        
        # Initialize random assignment
        assignment = np.random.choice([-1, 1], size=n).astype(np.int8)
        
        # Update phase
        for v in range(n):
            phase[v] = (assignment[v] == 1)
        
        sat_count = compute_sat_count(assignment)
        energy = int((sat_count == 0).sum())
        
        if energy == 0:
            return [a == 1 for a in assignment]
        
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment.copy()
        
        # Clause weights for weighted WalkSAT
        weights = np.ones(m, dtype=np.float64)
        
        step = 0
        while step < max_steps and time.time() < deadline:
            # Find unsatisfied clauses
            unsat_mask = (sat_count == 0)
            unsat_idx = np.where(unsat_mask)[0]
            
            if len(unsat_idx) == 0:
                return [a == 1 for a in assignment]
            
            energy = len(unsat_idx)
            
            if energy < best_energy:
                best_energy = energy
                best_assignment = assignment.copy()
            
            # Pick a clause: with prob 0.7 pick weighted random from unsat, else random
            if random.random() < 0.7:
                # Weighted pick
                w = weights[unsat_idx]
                w = w / w.sum()
                c_idx = unsat_idx[random.choices(range(len(unsat_idx)), weights=w, k=1)[0]]
            else:
                c_idx = unsat_idx[random.randint(0, len(unsat_idx) - 1)]
            
            # Pick variable from this clause
            if random.random() < flip_prob:
                # Random variable in clause
                j = random.randint(0, 2)
            else:
                # Pick variable that appears most often in other unsat clauses
                # Simplified: pick the variable in this clause with highest weight contribution
                best_j = 0
                best_score = -1
                for j in range(3):
                    v = var_indices[c_idx, j]
                    # Count how many unsat clauses contain v
                    score = 0
                    for c2 in var_to_clauses[v]:
                        if sat_count[c2[0]] == 0:
                            score += weights[c2[0]]
                    if score > best_score:
                        best_score = score
                        best_j = j
                j = best_j
            
            var = var_indices[c_idx, j]
            
            # Compute delta E for flipping var
            # For each clause containing var, check if flip changes satisfaction
            c_idxs = var_clause_idx[var]
            c_sns = var_clause_sign[var]
            
            if len(c_idxs) == 0:
                # Variable doesn't appear in any clause (shouldn't happen)
                step += 1
                continue
            
            # Current satisfaction of the literal for var in each clause
            # sat[i] = (assignment[var] * c_sns[i]) > 0
            current_sat = (assignment[var] * c_sns) > 0  # bool array
            
            # After flip: new_sat = !current_sat
            # For clauses where current_sat=True: flip makes it unsat
            #   If sat_count was 1, it becomes 0 -> energy +1
            # For clauses where current_sat=False: flip makes it sat
            #   If sat_count was 0, it becomes 1 -> energy -1
            
            old_counts = sat_count[c_idxs]
            
            # Clauses that will become unsat (were satisfied, now unsat)
            becomes_unsat = current_sat & (old_counts == 1)
            # Clauses that will become sat (were unsat, now sat)
            becomes_sat = (~current_sat) & (old_counts == 0)
            
            delta = int(becomes_unsat.sum()) - int(becomes_sat.sum())
            
            # Accept if delta <= 0 or with probability exp(-delta/T)
            # Use a simple temperature schedule
            T = max(0.1, 1.0 - 0.0001 * step)
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Flip the variable
                assignment[var] = -assignment[var]
                phase[var] = (assignment[var] == 1)
                
                # Update sat_count
                # For each clause containing var:
                #   if the literal was satisfied, it's now unsat: sat_count -= 1
                #   if the literal was unsat, it's now sat: sat_count += 1
                # current_sat tells us old state
                sat_count[c_idxs] = np.where(current_sat, old_counts - 1, old_counts + 1)
                
                # Update weights: increase weight of unsat clauses
                unsat_now = (sat_count == 0)
                weights[unsat_now] *= 1.01
                weights[~unsat_now] *= weight_decay
                # Normalize occasionally to prevent overflow
                if step % 1000 == 0:
                    weights /= weights.sum() * m
            
            step += 1
        
        # End of restart
        # Decay weights back
        weights *= 0.5
    
    # Return best found
    if best_assignment is not None:
        return [a == 1 for a in best_assignment]
    else:
        return [False] * n

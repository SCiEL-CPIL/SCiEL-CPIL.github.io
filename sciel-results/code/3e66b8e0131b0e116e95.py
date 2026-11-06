# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin

    # Convert clauses to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build data structures
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # === PREPROCESSING: Pure Literal Elimination + Unit Propagation ===
    fixed = {}
    remaining_vars = set(range(n))
    remaining_clauses = list(clauses_0based)
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        
        pos_count = {}
        neg_count = {}
        for clause in remaining_clauses:
            for var, sign in clause:
                if var not in remaining_vars:
                    continue
                if sign > 0:
                    pos_count[var] = pos_count.get(var, 0) + 1
                else:
                    neg_count[var] = neg_count.get(var, 0) + 1
        
        for var in list(remaining_vars):
            if var not in pos_count and var not in neg_count:
                continue
            if var in pos_count and var not in neg_count:
                fixed[var] = True
                remaining_vars.discard(var)
                changed = True
            elif var not in pos_count and var in neg_count:
                fixed[var] = False
                remaining_vars.discard(var)
                changed = True
        
        if changed:
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
            continue
        
        unit_clauses = []
        for clause in remaining_clauses:
            unassigned = [(var, sign) for var, sign in clause if var in remaining_vars]
            if len(unassigned) == 1:
                unit_clauses.append(unassigned[0])
        
        if unit_clauses:
            for var, sign in unit_clauses:
                if var in remaining_vars:
                    fixed[var] = (sign > 0)
                    remaining_vars.discard(var)
            changed = True
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
    
    # === STOCHASTIC LOCAL CONSISTENCY (AC-3) PREPROCESSING ===
    # Randomly assign variables to enforce arc consistency, pruning search space
    ac_start = time.time()
    ac_budget = min(5.0, max(0.5, (deadline - ac_start) * 0.05))  # 5% of time, max 5s
    rng_ac = random.Random(999)
    
    # Build clause-to-var index for efficient AC
    clause_vars_list = []
    for clause in remaining_clauses:
        vs = [v for v, s in clause]
        clause_vars_list.append(vs)
    
    # AC-3 style propagation with random variable selection
    ac_changed = True
    ac_iter = 0
    while ac_changed and time.time() < ac_start + ac_budget:
        ac_changed = False
        ac_iter += 1
        
        # Try to find a variable that can be fixed based on arc consistency
        # A variable v can be fixed if all clauses containing v are satisfied
        # by some other literal when v takes a particular value
        
        # Sample a few random variables to check
        sample_size = min(50, len(remaining_vars))
        if sample_size == 0:
            break
        sample = rng_ac.sample(list(remaining_vars), sample_size)
        
        for var in sample:
            # Check if var can be set to True
            can_be_true = True
            can_be_false = True
            
            for (ci, sign) in var_clauses[var]:
                if ci >= len(remaining_clauses):
                    continue
                clause = remaining_clauses[ci]
                if not clause:
                    continue
                    
                # Check if clause is satisfiable without var
                # i.e., is there another literal in the clause that can be true?
                # For simplicity, check if any other unassigned var exists
                other_unassigned = [(v, s) for v, s in clause if v != var and v in remaining_vars]
                
                if not other_unassigned:
                    # Clause depends only on var (and fixed vars)
                    # If var is the only unassigned, it must satisfy this clause
                    if sign > 0:
                        can_be_false = False  # var must be True
                    else:
                        can_be_true = False  # var must be False
                    continue
                
                # If there are other unassigned vars, we can't determine yet
                # unless the clause is already satisfied by a fixed var
                is_satisfied_by_fixed = False
                for v, s in clause:
                    if v in fixed:
                        if fixed[v] == (s > 0):
                            is_satisfied_by_fixed = True
                            break
                if is_satisfied_by_fixed:
                    continue  # Clause already satisfied
            
            if can_be_true and not can_be_false:
                # var must be False
                fixed[var] = False
                remaining_vars.discard(var)
                ac_changed = True
            elif can_be_false and not can_be_true:
                # var must be True
                fixed[var] = True
                remaining_vars.discard(var)
                ac_changed = True
            
            if ac_changed:
                # Simplify clauses
                new_clauses = []
                for clause in remaining_clauses:
                    sat = False
                    simplified = []
                    for var2, sign2 in clause:
                        if var2 in fixed:
                            if fixed[var2] == (sign2 > 0):
                                sat = True
                                break
                        else:
                            simplified.append((var2, sign2))
                    if not sat:
                        if len(simplified) > 0:
                            new_clauses.append(simplified)
                remaining_clauses = new_clauses
                break  # Restart loop after change
    
    # Re-run unit propagation after AC
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        
        pos_count = {}
        neg_count = {}
        for clause in remaining_clauses:
            for var, sign in clause:
                if var not in remaining_vars:
                    continue
                if sign > 0:
                    pos_count[var] = pos_count.get(var, 0) + 1
                else:
                    neg_count[var] = neg_count.get(var, 0) + 1
        
        for var in list(remaining_vars):
            if var not in pos_count and var not in neg_count:
                continue
            if var in pos_count and var not in neg_count:
                fixed[var] = True
                remaining_vars.discard(var)
                changed = True
            elif var not in pos_count and var in neg_count:
                fixed[var] = False
                remaining_vars.discard(var)
                changed = True
        
        if changed:
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
            continue
        
        unit_clauses = []
        for clause in remaining_clauses:
            unassigned = [(var, sign) for var, sign in clause if var in remaining_vars]
            if len(unassigned) == 1:
                unit_clauses.append(unassigned[0])
        
        if unit_clauses:
            for var, sign in unit_clauses:
                if var in remaining_vars:
                    fixed[var] = (sign > 0)
                    remaining_vars.discard(var)
            changed = True
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
    
    n_reduced = len(remaining_vars)
    m_reduced = len(remaining_clauses)
    
    if n_reduced == 0:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            else:
                full_assignment[var] = False
        return full_assignment

    # Map reduced variables to indices
    reduced_var_list = list(remaining_vars)
    var_to_reduced_idx = {var: i for i, var in enumerate(reduced_var_list)}
    
    # Convert remaining clauses to use reduced indices
    reduced_clauses = []
    for clause in remaining_clauses:
        reduced_clause = []
        for var, sign in clause:
            if var in remaining_vars:
                reduced_clause.append((var_to_reduced_idx[var], sign))
        if reduced_clause:
            reduced_clauses.append(reduced_clause)
    
    n_r = n_reduced
    m_r = len(reduced_clauses)
    
    # Build var_clauses for reduced instance
    reduced_var_clauses = [[] for _ in range(n_r)]
    for ci, clause in enumerate(reduced_clauses):
        for var, sign in clause:
            reduced_var_clauses[var].append((ci, sign))

    def _run_walksat_v2(rng, max_time, walk_prob=0.3):
        """
        Optimized WalkSAT using boolean mask for unsat clauses.
        """
        if m_r == 0:
            return [True] * n_r, True
            
        assignment = np.array([rng.random() < 0.5 for _ in range(n_r)], dtype=bool)
        
        clause_vars = []
        clause_signs = []
        for clause in reduced_clauses:
            vs, ss = zip(*clause) if clause else ((), ())
            clause_vars.append(np.array(vs, dtype=np.int32))
            clause_signs.append(np.array(ss, dtype=np.int32))
        
        sat_count = np.zeros(m_r, dtype=np.int32)
        unsat_mask = np.zeros(m_r, dtype=bool)
        unsat_count = 0
        
        for i in range(m_r):
            cv = clause_vars[i]
            cs = clause_signs[i]
            match = assignment[cv] == (cs > 0)
            cnt = np.sum(match)
            sat_count[i] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_mask[i] = True
        
        if unsat_count == 0:
            return assignment.tolist(), True

        best_assignment = assignment.copy()
        best_unsat = unsat_count
        
        current_time = time.time()
        while time.time() < current_time + max_time:
            if unsat_count == 0:
                return assignment.tolist(), True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
                
            # Pick an unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist(), True
            ci = rng.choice(unsat_indices)
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            if rng.random() < walk_prob:
                idx = rng.randrange(len(cv))
                var_to_flip = cv[idx]
            else:
                best_score = -float('inf')
                var_to_flip = cv[0]
                
                for k in range(len(cv)):
                    var = cv[k]
                    sign = cs[k]
                    
                    score = 0
                    for (other_ci, other_sign) in reduced_var_clauses[var]:
                        if other_ci == ci:
                            continue
                            
                        is_true = (assignment[var] == (other_sign > 0))
                        
                        if is_true:
                            if sat_count[other_ci] == 1:
                                score -= 1
                        else:
                            if sat_count[other_ci] == 0:
                                score += 1
                            
                    if score > best_score:
                        best_score = score
                        var_to_flip = var
            
            # Perform Flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update sat_count and unsat_mask
            for (other_ci, other_sign) in reduced_var_clauses[var_to_flip]:
                was_true = (old_val == (other_sign > 0))
                
                if was_true:
                    sat_count[other_ci] -= 1
                    if sat_count[other_ci] == 0:
                        if not unsat_mask[other_ci]:
                            unsat_mask[other_ci] = True
                            unsat_count += 1
                else:
                    sat_count[other_ci] += 1
                    if sat_count[other_ci] == 1:
                        if unsat_mask[other_ci]:
                            unsat_mask[other_ci] = False
                            unsat_count -= 1
            
        return best_assignment.tolist(), False

    def _run_sa(rng, max_time):
        if m_r == 0:
            return [True] * n_r, True
            
        assignment = np.array([rng.random() < 0.5 for _ in range(n_r)], dtype=bool)
        
        clause_vars = []
        clause_signs = []
        for clause in reduced_clauses:
            vs, ss = zip(*clause) if clause else ((), ())
            clause_vars.append(np.array(vs, dtype=np.int32))
            clause_signs.append(np.array(ss, dtype=np.int32))
        
        sat_count = np.zeros(m_r, dtype=np.int32)
        unsat_mask = np.zeros(m_r, dtype=bool)
        unsat_count = 0
        
        for i in range(m_r):
            cv = clause_vars[i]
            cs = clause_signs[i]
            match = assignment[cv] == (cs > 0)
            cnt = np.sum(match)
            sat_count[i] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_mask[i] = True
        
        if unsat_count == 0:
            return assignment.tolist(), True

        best_assignment = assignment.copy()
        best_unsat = unsat_count
        
        T0 = 1.0
        T_end = 0.005
        
        current_time = time.time()
        
        while time.time() < current_time + max_time:
            if unsat_count == 0:
                return assignment.tolist(), True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick random variable
            var = rng.randrange(n_r)
            old_val = assignment[var]
            assignment[var] = not old_val
            
            delta = 0
            for (other_ci, other_sign) in reduced_var_clauses[var]:
                was_true = (old_val == (other_sign > 0))
                
                if was_true:
                    sat_count[other_ci] -= 1
                    if sat_count[other_ci] == 0:
                        if not unsat_mask[other_ci]:
                            unsat_mask[other_ci] = True
                            unsat_count += 1
                            delta += 1
                else:
                    sat_count[other_ci] += 1
                    if sat_count[other_ci] == 1:
                        if unsat_mask[other_ci]:
                            unsat_mask[other_ci] = False
                            unsat_count -= 1
                            delta -= 1
            
            new_unsat = unsat_count
            
            if new_unsat < unsat_count:
                pass # Accept
            else:
                if delta > 0:
                    elapsed = time.time() - current_time
                    T = T0 * (T_end / T0) ** (elapsed / max_time)
                    if rng.random() < np.exp(-delta / max(T, 1e-10)):
                        pass # Accept
                    else:
                        # Reject: Revert
                        assignment[var] = old_val
                        for (other_ci, other_sign) in reduced_var_clauses[var]:
                            was_true = (old_val == (other_sign > 0))
                            if was_true:
                                sat_count[other_ci] += 1
                                if sat_count[other_ci] == 1:
                                    unsat_mask[other_ci] = False
                                    unsat_count -= 1
                            else:
                                sat_count[other_ci] -= 1
                                if sat_count[other_ci] == 0:
                                    unsat_mask[other_ci] = True
                                    unsat_count += 1
                else:
                    # Reject: Revert
                    assignment[var] = old_val
                    for (other_ci, other_sign) in reduced_var_clauses[var]:
                        was_true = (old_val == (other_sign > 0))
                        if was_true:
                            sat_count[other_ci] += 1
                            if sat_count[other_ci] == 1:
                                unsat_mask[other_ci] = False
                                unsat_count -= 1
                        else:
                            sat_count[other_ci] -= 1
                            if sat_count[other_ci] == 0:
                                unsat_mask[other_ci] = True
                                unsat_count += 1
            
        return best_assignment.tolist(), False

    # Run Solvers
    rng = random.Random(42)
    
    # Allocate time: 60% to WalkSAT, 40% to Simulated Annealing
    total_time = deadline - time.time()
    if total_time <= 0:
        return [False] * n
        
    walk_time = total_time * 0.6
    sa_time = total_time * 0.4
    
    result, solved = _run_walksat_v2(rng, walk_time, walk_prob=0.3)
    
    if solved:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assignment[var] = result[var_to_reduced_idx[var]]
            else:
                full_assignment[var] = False
        return full_assignment

    # If not solved, try Simulated Annealing
    result2, solved2 = _run_sa(random.Random(123), sa_time)
    
    if solved2:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assignment[var] = result2[var_to_reduced_idx[var]]
            else:
                full_assignment[var] = False
        return full_assignment

    # Return best result found
    full_assignment = [False] * n
    for var in range(n):
        if var in fixed:
            full_assignment[var] = fixed[var]
        elif var in var_to_reduced_idx:
            full_assignment[var] = result[var_to_reduced_idx[var]]
        else:
            full_assignment[var] = False
    return full_assignment

# EVOLVE-BLOCK-END

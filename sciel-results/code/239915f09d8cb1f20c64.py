"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 325.0  # Leave margin for overhead (evaluator kills at 330s)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess clauses into numpy arrays for fast evaluation
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Precompute sign_bool for faster satisfaction check
    sign_bool = (clause_data == 1)  # (m, 3) bool array
    
    # Build variable-to-clauses mapping for incremental updates
    var_clause_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clause_pos[v].append((i, j))
    
    def compute_satisfaction(assignment):
        """Compute boolean array of satisfied clauses from scratch."""
        vals = assignment[clause_vars]  # (m, 3) bools
        literal_true = (vals == sign_bool)  # (m, 3)
        return np.any(literal_true, axis=1)  # (m,) bools
    
    def polish(assignment, sat, max_steps=50000):
        """High-precision polish: greedy flips targeting minimum unsat, then random-walk escape."""
        rng = random.Random(99999)
        best_unsat = int(np.sum(~sat))
        best_assign = assignment.copy()
        
        # Phase 1: Greedy descent - always flip variable that reduces unsat most
        for step in range(max_steps // 2):
            if time.time() - start_time > timeout - 3:
                break
            if best_unsat == 0:
                return assignment.tolist(), sat
            
            # Find best single flip
            best_delta = 0
            best_var = -1
            for v in range(n):
                if not var_clause_pos[v]:
                    continue
                becomes_sat = 0
                becomes_unsat = 0
                for ci, j in var_clause_pos[v]:
                    if not sat[ci]:
                        becomes_sat += 1
                    else:
                        sole = True
                        for k in range(3):
                            if k == j:
                                continue
                            v2 = clause_vars[ci, k]
                            if v2 == v:
                                continue
                            if assignment[v2] == sign_bool[ci, k]:
                                sole = False
                                break
                        if sole:
                            becomes_unsat += 1
                delta = becomes_sat - becomes_unsat
                if delta > best_delta:
                    best_delta = delta
                    best_var = v
            
            if best_var < 0 or best_delta <= 0:
                break
            
            assignment[best_var] = not assignment[best_var]
            for ci, j in var_clause_pos[best_var]:
                was_sat = sat[ci]
                is_now_sat = False
                for k in range(3):
                    if assignment[clause_vars[ci, k]] == sign_bool[ci, k]:
                        is_now_sat = True
                        break
                sat[ci] = is_now_sat
                if was_sat and not is_now_sat:
                    best_unsat += 1
                elif not was_sat and is_now_sat:
                    best_unsat -= 1
            
            if best_unsat < int(np.sum(~sat)):
                pass  # already tracked
            current_unsat = int(np.sum(~sat))
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()
        
        # Phase 2: Random walk with bias toward improving moves
        for step in range(max_steps // 2):
            if time.time() - start_time > timeout - 3:
                break
            current_unsat = int(np.sum(~sat))
            if current_unsat == 0:
                return assignment.tolist(), sat
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()
            
            # Pick random unsatisfied clause
            unsat_idx = np.where(~sat)[0]
            if len(unsat_idx) == 0:
                return assignment.tolist(), sat
            ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])
            
            # Score each variable in this clause
            clause_v = clause_vars[ci]
            deltas = []
            for v in clause_v:
                becomes_sat = 0
                becomes_unsat = 0
                for ci2, j2 in var_clause_pos[v]:
                    if not sat[ci2]:
                        becomes_sat += 1
                    else:
                        sole = True
                        for k in range(3):
                            if k == j2:
                                continue
                            v2 = clause_vars[ci2, k]
                            if v2 == v:
                                continue
                            if assignment[v2] == sign_bool[ci2, k]:
                                sole = False
                                break
                        if sole:
                            becomes_unsat += 1
                deltas.append(becomes_sat - becomes_unsat)
            
            # 60% greedy, 40% random among top-2
            max_d = max(deltas)
            if rng.random() < 0.6:
                var_to_flip = clause_v[int(np.argmax(deltas))]
            else:
                # Sample from top-2 deltas
                sorted_idx = np.argsort(deltas)[::-1]
                if len(sorted_idx) >= 2 and rng.random() < 0.5:
                    var_to_flip = clause_v[int(sorted_idx[1])]
                else:
                    var_to_flip = clause_v[int(sorted_idx[0])]
            
            assignment[var_to_flip] = not assignment[var_to_flip]
            for ci2, j2 in var_clause_pos[var_to_flip]:
                was_sat = sat[ci2]
                is_now_sat = False
                for k in range(3):
                    if assignment[clause_vars[ci2, k]] == sign_bool[ci2, k]:
                        is_now_sat = True
                        break
                sat[ci2] = is_now_sat
        
        current_unsat = int(np.sum(~sat))
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assign = assignment.copy()
        
        if best_unsat == 0:
            return best_assign.tolist(), sat
        return None, sat
    
    def walk_sat(restart_assignment, max_flips, seed):
        """Run WalkSAT with clause-weight tracking and efficient unsat tracking."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Initialize satisfaction state
        sat = compute_satisfaction(assignment)
        unsat_count = int(np.sum(~sat))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        # Track unsat indices efficiently
        unsat_indices = np.where(~sat)[0]
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        # Clause weights for weighted clause selection
        clause_weights = np.ones(m, dtype=np.float64)
        
        # Track number of times each unsat clause has been picked (for adaptive weighting)
        clause_pick_count = np.zeros(m, dtype=np.int32)
        
        for step in range(max_flips):
            if time.time() - start_time > timeout - 5:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Track best state
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause with weighting
            if len(unsat_indices) > 0:
                # Weighted selection: clauses picked less often get higher weight
                local_weights = clause_weights[unsat_indices]
                # Boost under-picked clauses
                local_weights = local_weights * (1.0 + 0.1 * clause_pick_count[unsat_indices])
                local_weights = local_weights / local_weights.sum()
                clause_idx = int(unsat_indices[rng.choices(range(len(unsat_indices)), weights=local_weights)[0]])
                clause_pick_count[clause_idx] += 1
            else:
                return assignment.tolist()
            
            # Choose variable to flip using score-based strategy
            clause_v = clause_vars[clause_idx]
            deltas = [0, 0, 0]
            
            for pos, v in enumerate(clause_v):
                becomes_sat = 0
                becomes_unsat = 0
                
                for ci, j in var_clause_pos[v]:
                    if not sat[ci]:
                        becomes_sat += 1
                    else:
                        sole = True
                        for k in range(3):
                            if k == j:
                                continue
                            v2 = clause_vars[ci, k]
                            if v2 == v:
                                continue
                            if assignment[v2] == sign_bool[ci, k]:
                                sole = False
                                break
                        if sole:
                            becomes_unsat += 1
                
                deltas[pos] = becomes_sat - becomes_unsat
            
            # Adaptive temperature: decrease over time for more greedy behavior
            temp = max(0.5, 3.0 - 0.05 * (step / max(1, max_flips / 10)))
            exp_deltas = np.exp(np.array(deltas, dtype=np.float64) / temp)
            probs = exp_deltas / exp_deltas.sum()
            
            # Greedy probability increases as we get closer to solution
            greedy_prob = min(0.9, 0.5 + 0.1 * (1 - unsat_count / max(1, m)))
            if rng.random() < greedy_prob:
                var_to_flip = clause_v[int(np.argmax(deltas))]
            else:
                var_to_flip = clause_v[rng.choices([0, 1, 2], weights=probs)[0]]
            
            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Incrementally update satisfaction for affected clauses
            changed = False
            for ci, j in var_clause_pos[var_to_flip]:
                was_sat = sat[ci]
                is_now_sat = False
                for k in range(3):
                    if assignment[clause_vars[ci, k]] == sign_bool[ci, k]:
                        is_now_sat = True
                        break
                sat[ci] = is_now_sat
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    changed = True
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    changed = True
            
            # Rebuild unsat_indices only when needed
            if changed:
                unsat_indices = np.where(~sat)[0]
        
        # Final check
        if unsat_count == 0:
            return assignment.tolist()
        
        # If close to solution, try polish
        if unsat_count <= 5:
            result, _ = polish(assignment, sat)
            if result is not None:
                return result
        
        return None
    
    # Main search loop with restarts and phase saving
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    phase_assignment = None
    best_unsat_ever = m + 1
    best_phase_sat = None
    
    # Initial phase: try a few quick random walks to find a good starting point
    for _ in range(8):
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        sat = compute_satisfaction(initial)
        unsat = int(np.sum(~sat))
        if unsat < best_unsat_ever:
            best_unsat_ever = unsat
            phase_assignment = initial.copy()
            best_phase_sat = sat.copy()
    
    # Try quick WalkSAT from best initial
    if phase_assignment is not None:
        quick_result = walk_sat(phase_assignment, 20000, seed=42)
        if quick_result is not None:
            return quick_result
    
    while time.time() - start_time < timeout - 15:
        restart_count += 1
        
        # Use phase assignment with adaptive perturbation
        if phase_assignment is not None and restart_count > 1:
            initial = phase_assignment.copy()
            # Perturbation scales with restart count but caps out
            num_flips = max(1, int(n * (0.05 + 0.02 * min(restart_count, 25))))
            num_flips = min(num_flips, n)
            flip_indices = rng.sample(range(n), num_flips)
            for idx in flip_indices:
                initial[idx] = not initial[idx]
        else:
            initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Adaptive flip budget: more flips for harder instances
        base_flips = 15000
        adaptive_factor = 1.0 + 0.15 * min(restart_count, 20)
        max_flips = int(base_flips * adaptive_factor)
        
        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
        
        # Update phase assignment if we found something better
        # (walk_sat returns None on failure, but we can track best internally)
        # For now, keep the initial phase assignment
    
    if best_result is not None:
        return best_result
    
    # If we didn't find a solution, try one final polish from the best known state
    if phase_assignment is not None:
        final_sat = compute_satisfaction(phase_assignment)
        final_result, _ = polish(phase_assignment, final_sat, max_steps=100000)
        if final_result is not None:
            return final_result
        return phase_assignment.tolist()
    
    # Fallback: return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END

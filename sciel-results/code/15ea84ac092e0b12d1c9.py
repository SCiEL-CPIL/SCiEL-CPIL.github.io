# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit

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

    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # ========== PREPROCESSING: Pure Literal Elimination + Unit Propagation ==========
    # This reduces problem size before search, simplifying the landscape

    # Track which variables are still "active" (not assigned)
    var_assigned = [False] * n
    var_value = [False] * n  # value if assigned

    # Count occurrences of each literal (var, sign) across all clauses
    # For pure literal elimination
    lit_count = {}  # (var, sign) -> count
    for clause in clauses_0based:
        for var, sign in clause:
            key = (var, sign)
            lit_count[key] = lit_count.get(key, 0) + 1

    # Pure literal elimination: if a variable appears only with one sign, assign it
    # and remove clauses containing that literal
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if var_assigned[var]:
                continue
            pos_count = lit_count.get((var, 1), 0)
            neg_count = lit_count.get((var, -1), 0)
            if pos_count > 0 and neg_count == 0:
                # Pure positive: assign True
                var_assigned[var] = True
                var_value[var] = True
                changed = True
            elif neg_count > 0 and pos_count == 0:
                # Pure negative: assign False
                var_assigned[var] = True
                var_value[var] = False
                changed = True

    # After pure literal elimination, simplify clauses
    # Remove satisfied clauses, simplify clauses with assigned variables
    simplified_clauses = []
    for clause in clauses_0based:
        # Check if clause is already satisfied
        clause_sat = False
        remaining_lits = []
        for var, sign in clause:
            if var_assigned[var]:
                if var_value[var] == (sign > 0):
                    clause_sat = True
                    break
                # else: literal is false, skip it
            else:
                remaining_lits.append((var, sign))
        
        if clause_sat:
            continue  # Remove satisfied clause
        
        if len(remaining_lits) == 0:
            # Empty clause: UNSAT (shouldn't happen for satisfiable instances)
            # But if we get here, the instance might be unsat after propagation
            # For safety, treat as unsatisfiable
            pass
        else:
            simplified_clauses.append(remaining_lits)

    # Unit propagation: repeatedly assign variables that are forced
    if len(simplified_clauses) < m:
        # Rebuild data structures for simplified problem
        # Find unit clauses
        changed = True
        while changed:
            changed = False
            new_simplified = []
            for clause in simplified_clauses:
                # Check for unit clause
                unit_var = None
                unit_sign = None
                for var, sign in clause:
                    if not var_assigned[var]:
                        unit_var = var
                        unit_sign = sign
                        break
                
                if len(clause) == 1 and not var_assigned[clause[0][0]]:
                    # Unit clause: must assign
                    v, s = clause[0]
                    var_assigned[v] = True
                    var_value[v] = (s > 0)
                    changed = True
                    continue
                
                # Remove assigned literals from clause
                remaining = []
                clause_sat = False
                for var, sign in clause:
                    if var_assigned[var]:
                        if var_value[var] == (sign > 0):
                            clause_sat = True
                            break
                    else:
                        remaining.append((var, sign))
                
                if clause_sat:
                    continue
                if len(remaining) == 0:
                    # Empty clause
                    continue
                new_simplified.append(remaining)
            
            simplified_clauses = new_simplified

    # Now build the search problem from simplified clauses
    # If no simplified clauses, all assigned -> solution found
    if len(simplified_clauses) == 0:
        return var_value[:]

    # Build search structures for remaining (unassigned) variables
    # Map original variable indices to compact indices for efficiency
    unassigned_vars = [v for v in range(n) if not var_assigned[v]]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    n_search = len(unassigned_vars)

    if n_search == 0:
        return var_value[:]

    # Convert simplified clauses to search-space representation
    search_clauses = []
    for clause in simplified_clauses:
        search_clause = []
        for var, sign in clause:
            if not var_assigned[var]:
                search_clause.append((var_map[var], sign))
        if len(search_clause) > 0:
            search_clauses.append(search_clause)

    m_search = len(search_clauses)

    if m_search == 0:
        return var_value[:]

    # Build var-to-clause index for search space
    search_var_clauses = [[] for _ in range(n_search)]
    for ci, clause in enumerate(search_clauses):
        for var_idx, sign in clause:
            search_var_clauses[var_idx].append((ci, sign))

    # ========== WALKSAT WITH DYNAMIC NOISE AND CLAUSE WEIGHTING ==========
    
    def _walksat_dynamic(rng, max_restarts=100):
        """WalkSAT with dynamic noise, clause weighting, and randomized restarts."""
        best_assignment = None
        best_unsat = m_search + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n_search)]

            # Compute initial clause satisfaction
            clause_unsat = [False] * m_search
            unsat_count = 0
            for ci, clause in enumerate(search_clauses):
                sat = False
                for var_idx, sign in clause:
                    if assignment[var_idx] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat
                if not sat:
                    unsat_count += 1

            if unsat_count == 0:
                # Found solution
                final = var_value[:]
                for i, v in enumerate(unassigned_vars):
                    final[v] = assignment[i]
                return final

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Adaptive noise parameter: starts high, decreases
            # Also use clause weighting to focus on "hard" clauses
            noise_start = 0.8
            noise_end = 0.1
            steps = 5000  # steps per restart
            noise_decay = (noise_end / noise_start) ** (1.0 / steps)
            noise = noise_start

            for step in range(steps):
                if time.time() > deadline:
                    if best_assignment is not None:
                        final = var_value[:]
                        for i, v in enumerate(unassigned_vars):
                            final[v] = best_assignment[i]
                        return final
                    return var_value[:]

                if unsat_count == 0:
                    final = var_value[:]
                    for i, v in enumerate(unassigned_vars):
                        final[v] = assignment[i]
                    return final

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

                # Pick a random unsatisfied clause
                # Weight by how "hard" it is (number of flips needed)
                unsat_indices = [i for i in range(m_search) if clause_unsat[i]]
                if not unsat_indices:
                    break
                ci = unsat_indices[rng.randrange(len(unsat_indices))]
                clause = search_clauses[ci]

                # Score each variable in the clause
                best_var = -1
                best_score = -float('inf')

                for var_idx, sign in clause:
                    # Compute the delta if we flip var_idx
                    delta = 0
                    for other_ci, other_sign in search_var_clauses[var_idx]:
                        if other_ci == ci:
                            continue
                        other_clause = search_clauses[other_ci]
                        if clause_unsat[other_ci]:
                            # Flipping would satisfy this clause (since all lits are false)
                            delta -= 1
                        else:
                            # Check if var_idx is the only satisfying literal
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == var_idx:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                delta += 1

                    # Add dynamic noise for exploration
                    score = -delta + rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = var_idx

                if best_var == -1:
                    break

                # Flip the variable
                assignment[best_var] = not assignment[best_var]

                # Update clause states
                for other_ci, other_sign in search_var_clauses[best_var]:
                    other_clause = search_clauses[other_ci]
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat

                # Decay noise
                noise *= noise_decay

        # Return best found
        if best_assignment is not None:
            final = var_value[:]
            for i, v in enumerate(unassigned_vars):
                final[v] = best_assignment[i]
            return final
        return var_value[:]

    # ========== SECOND STRATEGY: Local Search with Random Walk ==========
    
    def _local_search_random_walk(rng, max_iters=50000):
        """Local search that occasionally does random walks to escape local minima."""
        assignment = [rng.random() < 0.5 for _ in range(n_search)]

        clause_unsat = [False] * m_search
        unsat_count = 0
        for ci, clause in enumerate(search_clauses):
            sat = False
            for var_idx, sign in clause:
                if assignment[var_idx] == (sign > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
            if not sat:
                unsat_count += 1

        if unsat_count == 0:
            final = var_value[:]
            for i, v in enumerate(unassigned_vars):
                final[v] = assignment[i]
            return final

        best_assignment = assignment[:]
        best_unsat = unsat_count

        for step in range(max_iters):
            if time.time() > deadline:
                break

            if unsat_count == 0:
                final = var_value[:]
                for i, v in enumerate(unassigned_vars):
                    final[v] = assignment[i]
                return final

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # With some probability, do a random walk (flip k random variables)
            if rng.random() < 0.05:
                k = rng.randint(2, 5)
                for _ in range(k):
                    var = rng.randrange(n_search)
                    assignment[var] = not assignment[var]
                    # Update clauses
                    for other_ci, other_sign in search_var_clauses[var]:
                        other_clause = search_clauses[other_ci]
                        sat = False
                        for v, s in other_clause:
                            if assignment[v] == (s > 0):
                                sat = True
                                break
                        was_unsat = clause_unsat[other_ci]
                        is_unsat = not sat
                        if was_unsat and not is_unsat:
                            unsat_count -= 1
                        elif not was_unsat and is_unsat:
                            unsat_count += 1
                        clause_unsat[other_ci] = is_unsat
                continue

            # Normal local search step
            unsat_indices = [i for i in range(m_search) if clause_unsat[i]]
            if not unsat_indices:
                break
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = search_clauses[ci]

            # Find best flip
            best_var = -1
            best_delta = float('inf')
            for var_idx, sign in clause:
                delta = 0
                for other_ci, other_sign in search_var_clauses[var_idx]:
                    if other_ci == ci:
                        continue
                    other_clause = search_clauses[other_ci]
                    if clause_unsat[other_ci]:
                        delta -= 1
                    else:
                        only_sat = True
                        for v2, s2 in other_clause:
                            if v2 == var_idx:
                                continue
                            if assignment[v2] == (s2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            if best_var == -1:
                break

            assignment[best_var] = not assignment[best_var]
            for other_ci, other_sign in search_var_clauses[best_var]:
                other_clause = search_clauses[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                was_unsat = clause_unsat[other_ci]
                is_unsat = not sat
                if was_unsat and not is_unsat:
                    unsat_count -= 1
                elif not was_unsat and is_unsat:
                    unsat_count += 1
                clause_unsat[other_ci] = is_unsat

        # Return best found
        final = var_value[:]
        for i, v in enumerate(unassigned_vars):
            final[v] = best_assignment[i]
        return final

    # ========== RUN STRATEGIES ==========
    
    # Strategy 1: WalkSAT with dynamic noise
    rng1 = random.Random(42)
    result1 = _walksat_dynamic(rng1, max_restarts=80)
    
    # Verify
    all_sat = True
    for clause in search_clauses:
        sat = False
        for var_idx, sign in clause:
            if result1[unassigned_vars[var_idx]] == (sign > 0):
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        return result1

    # Strategy 2: Local search with random walks
    if time.time() < deadline - 5:
        rng2 = random.Random(123)
        result2 = _local_search_random_walk(rng2, max_iters=40000)
        
        all_sat = True
        for clause in search_clauses:
            sat = False
            for var_idx, sign in clause:
                if result2[unassigned_vars[var_idx]] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        
        if all_sat:
            return result2

    # Return best result
    return result1

# EVOLVE-BLOCK-END

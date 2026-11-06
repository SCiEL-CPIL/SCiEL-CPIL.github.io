"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Deadline guard: 330s limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause indices for faster access
    # clauses are lists of signed literals
    # We need to check satisfaction efficiently
    
    def check_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(clauses, assignment):
        count = 0
        for c in clauses:
            if not check_satisfied(c, assignment):
                count += 1
        return count

    best_assignment = None
    min_unsat = float('inf')
    
    # WalkSAT with restarts
    max_steps_per_run = 5000
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        # Phase saving: start with previous best or random
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb the best assignment slightly
            assignment = list(best_assignment)
            for _ in range(n // 5):
                idx = rng.randint(0, n - 1)
                assignment[idx] = not assignment[idx]
        
        # Walk
        for step in range(max_steps_per_run):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                if not check_satisfied(c, assignment):
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                # Solved
                return assignment
            
            # Track best so far
            current_unsat = len(unsat_clauses)
            if current_unsat < min_unsat:
                min_unsat = current_unsat
                best_assignment = list(assignment)
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Heuristic Flip: Pick the literal in the clause that satisfies the most
            # other unsatisfied clauses (or breaks the fewest).
            best_literal = None
            best_score = -1
            for lit in clause:
                var_idx = abs(lit) - 1
                # Simulate flip
                # Count how many unsat clauses this flip would satisfy
                # A clause is satisfied if any of its literals matches the new assignment
                # Since the clause is currently unsat, all its literals are currently false.
                # If we flip var_idx, it becomes true.
                # So this clause becomes satisfied.
                # We need to check impact on OTHER unsat clauses.
                # Actually, simpler: just count frequency of this variable in unsat_clauses?
                # No, that's not quite right.
                
                # Efficient heuristic:
                # If we flip this variable, this clause is satisfied.
                # How many other unsat clauses contain this variable?
                # If they contain the variable with the same sign, they become satisfied.
                # If they contain the variable with the opposite sign, they remain unsatisfied 
                # (unless another literal in them is true, but they are unsat, so all are false).
                # Wait, if a clause is unsat, all its literals are false.
                # If I flip var v to True:
                # Any clause containing +v becomes True (satisfied).
                # Any clause containing -v remains False (unless it has other literals, 
                # but since it was unsat, all were false, so -v was false -> v was true. 
                # Now v is false. So -v becomes true. So clauses with -v become satisfied too!)
                
                # Correction:
                # Current state: Clause C is unsat. All literals in C are false.
                # Let's flip variable $v$ (part of C).
                # New state of $v$: inverted.
                # Clause C: contains $v$ or $\neg v$. Since C was unsat, the literal in C was false.
                # If C contains $+v$, then $v$ was False. Flip makes $v$ True. $+v$ becomes True. C satisfied.
                # If C contains $-v$, then $v$ was True. Flip makes $v$ False. $-v$ becomes True. C satisfied.
                # So C is always satisfied.
                
                # What about other unsat clauses $C'$?
                # If $C'$ contains $+v$:
                #   If $v$ was False, $+v$ was False. Flip makes $v$ True. $+v$ becomes True. $C'$ satisfied.
                #   If $v$ was True, $+v$ was True. But $C'$ is unsat, so this case is impossible.
                #   So if $C'$ contains $+v$, it MUST be that $v$ was False (so $+v$ false). Flip satisfies $C'$.
                # If $C'$ contains $-v$:
                #   If $v$ was True, $-v$ was False. Flip makes $v$ False. $-v$ becomes True. $C'$ satisfied.
                #   If $v$ was False, $-v$ was True. But $C'$ is unsat, so this case is impossible.
                
                # Conclusion: Flipping $v$ satisfies ALL unsat clauses that contain $v$ (in either polarity)!
                # This is because for an unsat clause, the literal involving $v$ must have been evaluating to False.
                # Flipping $v$ makes that literal True.
                
                # So the heuristic is: Pick the variable in the current unsat clause that appears 
                # in the most OTHER unsat clauses.
                
                count = 0
                for c2 in unsat_clauses:
                    if c2 is clause: continue
                    for l2 in c2:
                        if abs(l2) == abs(lit):
                            count += 1
                            break
                            
                if count > best_score:
                    best_score = count
                    best_literal = lit
                    
            if best_literal is None:
                best_literal = rng.choice(clause)
                
            assignment[abs(best_literal) - 1] = not assignment[abs(best_literal) - 1]
            
        if time.time() > deadline:
            break

    # If we didn't find a solution, return the best assignment found (which might not be satisfying, 
    # but the function contract says "Return a satisfying assignment...". 
    # If we return a non-satisfying one, it scores 0 for that instance.
    # However, if we are out of time, we have no choice.
    # If best_assignment is None, return a random one.
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
# EVOLVE-BLOCK-END

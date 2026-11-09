            clause = clauses[ci]
            
            if not clause:
                # Handle empty clause (UNSAT case, though task says SAT)
                return assignment
            
            # Select literal: prefer flipping literals that would satisfy the clause
            # (i.e., literals currently false in the clause)
            false_literals = []
    
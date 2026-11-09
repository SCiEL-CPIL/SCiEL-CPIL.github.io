    def is_satisfied(assignment, clause_idx):
        # We need to check if any literal in the clause is true.
        # Pos literals: var is True
        # Neg literals: var is False
        
        # Create a mask of True variables
        # This is expensive if done from scratch every time.
        # Instead, we can maintain a `true_mask` and `false_mask`?
        # Or just use the assignment list.
        
        # For n=50, bit operations are fast.
        # Let's precompute nothing, just check.
        pm = pos_masks[clause_idx]
        nm = neg_masks[clause_idx]
        
        # Check pos
        for v in range(n):
            if pm & (1 << v):
                if assignment[v]:
                    return True
        # Check neg
        for v in range(n):
            if nm & (1 << v):
                if not assignment[v]:
                    return True
        return False

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    
    def _check_add_A(x, A, S_A, D_B):
        """Check if x can be added to A given current A, Sums of A, and Diffs of B."""
        # Sidon check for A
        if (x + x) in S_A:
            return False
        for a in A:
            if (x + a) in S_A:
                return False
        
        # Difference disjointness check
        for a in A:
            d1 = x - a
            d2 = a - x
            if d1 in D_B or d2 in D_B:
                return False
        return True

    def _check_add_B(x, B, S_B, D_A):
        """Check if x can be added to B given current B, Sums of B, and Diffs of A."""
        # Sidon check for B
        if (x + x) in S_B:
            return False
        for b in B:
            if (x + b) in S_B:
                return False
        
        # Difference disjointness check
        for b in B:
            d1 = x - b
            d2 = b - x
            if d1 in D_A or d2 in D_A:
                return False
        return True

    def _beam_search(width, deadline):
        """
        Beam search over elements 1..256.
        State: (A, B, S_A, S_B, D_A, D_B)
        """
        # Initial state: empty sets
        # A, B are lists. S_A, S_B, D_A, D_B are sets.
        initial_state = ([], [], set(), set(), set(), set())
        beam = [initial_state]
        
        for x in range(1, 257):
            if time.time() > deadline:
                break
            
            new_beam = []
            for A, B, S_A, S_B, D_A, D_B in beam:
                # Option 1: Skip x
                new_beam.append((A, B, S_A, S_B, D_A, D_B))
                
                # Option 2: Add x to A
                if _check_add_A(x, A, S_A, D_B):
                    new_A = A + [x]
                    new_S_A = S_A | {x + x}
                    new_D_A = D_A
                    # Update sums and diffs
                    for a in A:
                        new_S_A.add(x + a)
                        new_D_A.add(x - a)
                        new_D_A.add(a - x)
                    new_beam.append((new_A, B, new_S_A, S_B, new_D_A, D_B))
                    
                # Option 3: Add x to B
                if _check_add_B(x, B, S_B, D_A):
                    new_B = B + [x]
                    new_S_B = S_B | {x + x}
                    new_D_B = D_B
                    for b in B:
                        new_S_B.add(x + b)
                        new_D_B.add(x - b)
                        new_D_B.add(b - x)
                    new_beam.append((A, new_B, S_A, new_S_B, D_A, new_D_B))
            
            # Prune beam
            # Sort by size descending
            new_beam.sort(key=lambda s: len(s[0]) + len(s[1]), reverse=True)
            beam = new_beam[:width]
            
            # Optional: Early termination if no improvement? 
            # But we don't know the max yet.
            
        # Return best state from final beam
        if not beam:
            return ([], [])
        
        best_state = beam[0]
        return (best_state[0], best_state[1])

    # Run beam search with different widths
    # Start with a smaller width for quick result, then larger
    best_A, best_B = ([], [])
    best_size = 0
    
    # Try width 50 first
    A, B = _beam_search(50, deadline)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A, best_B = A, B
        
    # If time permits, try width 100
    if time.time() < deadline - 5.0:
        A, B = _beam_search(100, deadline)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A, best_B = A, B
            
    # If time permits, try width 200
    if time.time() < deadline - 10.0:
        A, B = _beam_search(200, deadline)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A, best_B = A, B

    # Ensure nonempty
    if not best_A and not best_B:
        best_A, best_B = [1], [2]
    elif not best_A:
        best_A = [1]
    elif not best_B:
        best_B = [1]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END

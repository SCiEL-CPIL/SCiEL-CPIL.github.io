"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: keep a safety margin before the 60s evaluator limit.
    deadline = time.time() + 55.0
    
    # Greedy baseline (same as seed) to have a valid fallback.
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    best_seq = greedy()
    best_len = len(best_seq)
    
    # Backtracking CSP solver with forward checking.
    # State: current sequence, set of all consecutive block sums so far.
    # We try to extend the sequence by picking the next integer > last.
    # Pruning: if adding x creates a duplicate sum, skip x.
    
    def backtrack(seq, sums, start_val, depth):
        nonlocal best_seq, best_len
        
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(best_seq)
        
        if len(seq) == 256:
            return True
        
        # If we've used up all values, stop.
        if start_val > 256:
            return False
        
        # Time check
        if time.time() > deadline:
            return False
        
        last = seq[-1] if seq else 0
        
        # Try candidates in increasing order, but also try some heuristics
        # to avoid getting stuck in bad branches.
        # We'll try a mix: first increasing, but if that fails quickly,
        # we could try other orders. For simplicity, start with increasing.
        
        # Optimization: if we can't possibly beat best_len, prune.
        # Max possible length = current len + (256 - last)
        if len(seq) + (256 - last) <= best_len:
            return False
        
        for x in range(last + 1, 257):
            # Check if adding x creates any duplicate consecutive block sum.
            # New sums are: x, x + seq[-1], x + seq[-1] + seq[-2], ...
            # i.e., x + suffix_sums of seq.
            
            # Compute new sums
            new_sums = []
            s = x
            for a in reversed(seq):
                s += a
                new_sums.append(s)
            
            # Check for collisions
            valid = True
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also, we need to ensure that the new sums don't collide with each other
            # (they are all distinct since seq is strictly increasing and positive)
            # Actually, the new sums are x, x+a_n, x+a_n+a_{n-1}, ...
            # Since a_i > 0, these are strictly increasing, so no internal collision.
            
            # Recurse
            sums.update(new_sums)
            if backtrack(seq + [x], sums, x, depth + 1):
                sums.difference_update(new_sums)
                return True
            
            sums.difference_update(new_sums)
        
        return False
    
    # Run backtracking with a time limit
    # Start with empty sequence
    backtrack([], set(), 0, 0)
    
    return best_seq
# EVOLVE-BLOCK-END

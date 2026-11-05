"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # 1. Run Greedy to get a good initial lower bound
    def run_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products

    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)

    # Helper to check validity and get new products set
    def check_validity(seq, prods, x):
        """
        Checks if x can be appended to seq given existing products in prods.
        Returns (is_valid, new_prods_set)
        """
        tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            tails.append(prod)
        
        # Check collision
        for t in tails:
            if t in prods:
                return False, None
        
        # If valid, compute new set
        # Optimization: only copy if necessary, but we need a new set object for the beam state
        new_prods = set(prods)
        new_prods.update(tails)
        return True, new_prods

    # 2. Beam Search
    # State: seq_tuple -> prods_set
    # We start with the empty state
    beam = {(): set()}
    
    # Beam width: needs to be large enough to explore diverse subsets
    # 300-500 is a good balance for 127 steps
    W = 400

    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            # The state (seq, prods) is carried over
            # If multiple paths lead to the same seq, prods are identical (deterministic).
            # We just need to ensure it's in next_beam.
            if seq not in next_beam:
                next_beam[seq] = prods
            
            # Option 2: Include x
            # Check if x can be appended
            is_valid, new_prods = check_validity(seq, prods, x)
            
            if is_valid:
                new_seq = seq + (x,)
                # new_seq is unique for this step given seq and x
                # But could new_seq exist in next_beam from a different path?
                # No, because x is fixed for this iteration. 
                # However, if we had duplicate seqs in beam (we don't, we dedupe), 
                # or if two different previous seqs led to the same new_seq? 
                # No, new_seq ends with x, and the prefix is seq. 
                # So new_seq is unique per (seq, x).
                # But wait, could next_beam already contain new_seq from a "Skip" of a previous iteration?
                # No, "Skip" keeps the old seq. "Include" creates a new seq ending in x.
                # Since x is unique in this loop, any seq ending in x is new to this loop's next_beam 
                # unless it was already there? No, next_beam is built fresh each x.
                # Actually, next_beam accumulates states for the *next* step.
                # If we skip x, we add seq.
                # If we include x, we add seq+(x,).
                # These are distinct sets of keys unless seq == seq' + (x,) which is impossible since x > all elements in seq.
                
                next_beam[new_seq] = new_prods
        
        # Prune beam
        # Sort by length desc, then by sum of elements asc (smaller sum is better for future flexibility? 
        # Or maybe larger sum? 
        # Usually, smaller elements leave more room for larger numbers later? 
        # Or does having smaller products allow more distinct products?
        # Let's stick with the heuristic from inspiration 1: -len, then sum asc.
        # Rationale: Shorter sequences with smaller numbers might be "easier" to extend? 
        # Or perhaps we want to prioritize sequences that are already long.
        # The primary goal is length. So -len is primary.
        # Secondary: If lengths are equal, which is better?
        # A sequence with smaller numbers has smaller products, potentially less likely to collide 
        # with future large products? Or is it the other way around?
        # Let's try -len, then sum asc.
        
        # To avoid sorting the entire dict which might be large (up to W*2 = 800 items), 
        # and W=400, sorting 800 items is fast.
        
        sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
        
        # Keep top W
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}
        
        # Update best
        if sorted_seqs:
            top_seq = sorted_seqs[0]
            if len(top_seq) > best_len:
                best_len = len(top_seq)
                best_seq = list(top_seq)
                best_prods = next_beam[top_seq]

    return list(best_seq)
# EVOLVE-BLOCK-END

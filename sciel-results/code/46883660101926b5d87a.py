import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Beam width
    W = 100
    
    # State: (sequence_tuple, product_set)
    # Start with just [2]
    # We know 2 is the smallest, so it's a good start.
    # Actually, we should consider if starting with 3 is better? 
    # Unlikely. 2 introduces product 2. 3 introduces 3. 
    # Any sequence starting with 3 can be prefixed with 2? 
    # If S = [3, 5], prods {3, 5, 15}. 
    # If we do [2, 3, 5], prods {2, 3, 5, 6, 15, 30}. 
    # New prods: 2, 6, 30. None in {3,5,15}. So [2,3,5] is valid.
    # So including 2 is always safe and adds length. 
    # Thus, optimal sequence must start with 2.
    
    beam = [((2,), {2})]
    
    for v in range(3, 129):
        if time.time() > deadline:
            break
            
        new_candidates = []
        
        for seq, prods in beam:
            # Option 1: Skip v
            # We add the current state to candidates. 
            # Note: Multiple sequences in the beam might be identical? 
            # No, we deduplicate by sequence tuple.
            new_candidates.append((seq, prods))
            
            # Option 2: Include v
            # Check if v can be appended to seq
            # New products ending at v
            new_prods = []
            prod = v
            for a in reversed(seq):
                prod *= a
                new_prods.append(prod)
            new_prods_set = set(new_prods)
            
            if not prods.intersection(new_prods_set):
                new_seq = seq + (v,)
                new_prods_total = prods.union(new_prods_set)
                new_candidates.append((new_seq, new_prods_total))
        
        # Deduplicate by sequence
        # Since sequences are tuples, we can use a dict
        best_by_seq = {}
        for seq, prods in new_candidates:
            # If we have seen this sequence before, it's the same prods.
            # We just need to keep one instance.
            if seq not in best_by_seq:
                best_by_seq[seq] = prods
        
        candidates = list(best_by_seq.items())
        
        # Sort candidates
        # Primary: Length (descending)
        # Secondary: Last element (ascending) - smaller last element allows more future numbers
        # Tertiary: Maybe sum of elements? Smaller sum might be better?
        # Let's stick to Length, then Last Element.
        candidates.sort(key=lambda x: (-len(x[0]), x[0][-1]))
        
        beam = candidates[:W]
        
        # Optional: Early termination if no sequences can be extended?
        # If max length in beam is L, and we are at v, and v is close to 128, 
        # we might not gain much. But let's run to the end or time out.
        
    # Return the longest sequence
    if not beam:
        return []
        
    # Find the one with max length
    best_seq = max(beam, key=lambda x: len(x[0]))
    return list(best_seq[0])


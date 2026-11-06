import time

def construct():
    start_time = time.time()
    # Use 85 seconds to be safe against the 90s limit
    deadline = start_time + 85.0
    
    # State: (sequence_tuple, product_set)
    # Initialize with sequences starting from small numbers
    # Starting with 2 is the most promising for maximizing length
    # We include a few others for diversity in the initial beam
    beam = []
    for x in range(2, 10):
        beam.append(((x,), {x}))
        
    best_seq = max(beam, key=lambda s: len(s[0]))[0]
    best_len = len(best_seq)
    
    max_beam_width = 500
    
    while time.time() < deadline:
        next_beam = []
        
        for seq, prods in beam:
            last_val = seq[-1]
            
            # Optimization: If last_val is close to 128, few options
            # We can skip if last_val == 128
            
            for x in range(last_val + 1, 129):
                # Check if x itself is a duplicate product
                if x in prods:
                    continue
                
                # Check if any x * p is a duplicate product
                # We also need to build the new set
                # To save time, we can check collisions first, then build set?
                # Or build set and check membership?
                
                # Let's check collisions first to avoid set creation if invalid
                # Collision if x in prods (checked)
                # Collision if (x * p) in prods for any p in prods
                
                valid = True
                for p in prods:
                    if (x * p) in prods:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Valid extension. Build new state.
                new_prods = set(prods)
                new_prods.add(x)
                for p in prods:
                    new_prods.add(x * p)
                    
                new_seq = seq + (x,)
                next_beam.append((new_seq, new_prods))
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        
        if not next_beam:
            break
            
        # Sort by length (descending), then by last element (ascending)
        # This keeps long sequences and those with smaller last elements (more room to grow)
        next_beam.sort(key=lambda item: (-len(item[0]), item[0][-1]))
        
        # Trim to beam width
        beam = next_beam[:max_beam_width]
        
        # If the best sequence in the beam is not improving and the beam is "stable"
        # (i.e., we are just cycling through the same states or no new long states found),
        # we might want to break early? 
        # But it's hard to detect "stable" easily. 
        # The time limit handles this.
        
    return list(best_seq)

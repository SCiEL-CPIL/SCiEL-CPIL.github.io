import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    # Beam search
    # State: (sequence, set_of_products)
    # We start with all singletons
    beam = []
    for x in range(2, 129):
        beam.append(([x], {x}))
    
    best_seq = []
    best_len = 0
    
    # We will iterate by "depth" or just keep expanding the beam
    # Since the sequence must be strictly increasing, the length is bounded by 127.
    # But realistically it will be much shorter.
    
    # To make it efficient, we can group by last element? 
    # No, just expanding the beam is fine for small K.
    
    K = 500  # Beam width
    current_depth = 1
    
    while beam and time.time() < deadline:
        next_beam = []
        
        for seq, prods in beam:
            last = seq[-1]
            # Try extending with y > last
            for y in range(last + 1, 129):
                # Calculate new products
                # New products are:
                # 1. y
                # 2. y * seq[-1]
                # 3. y * seq[-1] * seq[-2]
                # ...
                # We can compute these incrementally.
                new_prods = set(prods)
                p = y
                # Check singleton y
                if p in new_prods:
                    continue
                new_prods.add(p)
                
                valid = True
                for a in reversed(seq):
                    p *= a
                    if p in new_prods:
                        valid = False
                        break
                    new_prods.add(p)
                
                if valid:
                    new_seq = seq + [y]
                    next_beam.append((new_seq, new_prods))
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not next_beam:
            break
        
        # Sort next_beam to keep the best candidates
        # Heuristic: Longer is better. If same length, smaller last element is better (more room).
        # Also, maybe sequences with smaller numbers are better?
        # Let's sort by length desc, then last element asc.
        next_beam.sort(key=lambda x: (-len(x[0]), x[0][-1]))
        
        beam = next_beam[:K]
        current_depth += 1
        
        # Optional: Early termination if we've reached the max possible length?
        # Max length is 127, but that's unlikely.
        
    return best_seq

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    
    # Verify
    prods = set()
    for i in range(len(res)):
        p = 1
        for j in range(i, len(res)):
            p *= res[j]
            if p in prods:
                print(f"Collision at {i}, {j}: {p}")
                sys.exit(1)
            prods.add(p)
    print("Valid")

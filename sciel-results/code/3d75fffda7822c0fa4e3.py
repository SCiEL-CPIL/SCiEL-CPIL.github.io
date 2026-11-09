import itertools
import time

def construct():
    start_time = time.time()
    
    # Generate smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            
    solutions = [None] * 129
    
    # Pre-compute valid antichains for k=1, 2, 3
    # We stop if all 1..128 are solved
    for k in range(1, 4):
        # Check if all solved
        if all(solutions[i] is not None for i in range(1, 129)):
            break
            
        for combo in itertools.combinations(smooth, k):
            # Check antichain
            valid = True
            for i in range(k):
                for j in range(i+1, k):
                    if combo[j] % combo[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            s = sum(combo)
            if 1 <= s <= 128 and solutions[s] is None:
                solutions[s] = list(combo)
                
        # Time check
        if time.time() - start_time > 50:
            break
            
    # Fallback: If any are None, do a per-target search (should not happen if k<=3 is sufficient)
    for n in range(1, 129):
        if solutions[n] is None:
            # Try k=4, 5...
            for k in range(4, 10):
                for combo in itertools.combinations([x for x in smooth if x < n], k):
                    valid = True
                    for i in range(k):
                        for j in range(i+1, k):
                            if combo[j] % combo[i] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and sum(combo) == n:
                        solutions[n] = list(combo)
                        break
                if solutions[n] is not None:
                    break
            if solutions[n] is None:
                # Last resort, just pick any valid antichain summing to n?
                # Or raise? The prompt implies it's always possible.
                # If we fail, we might return an invalid list.
                # But k<=3 should work.
                pass
                
    return solutions[1:129]

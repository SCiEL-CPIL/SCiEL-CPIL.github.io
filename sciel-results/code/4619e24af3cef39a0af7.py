"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 256
    
    # Start with the proven interval [171, 256]
    # For a in [171,256], b,c in [171,256] with a<b<=c:
    # b+c >= 172+172 = 344, b+c <= 256+256 = 512
    # Multiples of a >= 171 in [344, 512]: 2a is in [342, 512]
    # 2a >= 342, but b+c >= 344, so 2a could equal b+c if 2a >= 344
    # Wait, let me recheck: a < b <= c, so b >= a+1, c >= b >= a+1
    # b+c >= (a+1) + (a+1) = 2a+2
    # So b+c >= 2a+2, and 2a < b+c always.
    # Next multiple is 3a. Is 3a <= 512? For a <= 170, yes. For a >= 171, 3a >= 513 > 512.
    # So for a >= 171, 3a > 512 >= b+c, so no multiple of a in [2a+2, 512].
    # Thus [171, 256] is feasible.
    
    base_set = list(range(171, n + 1))
    selected = set(base_set)
    
    # Try to add numbers from 1 to 170
    def is_feasible(candidate, sel_set):
        """Check if adding candidate to sel_set maintains the property."""
        # For all a<b<=c in (sel_set ∪ {candidate}), a ∤ (b+c)
        # We need to check new triples involving candidate
        # Case 1: candidate is 'a' (smallest)
        # For all b,c in sel_set with b > candidate, c >= b: candidate ∤ (b+c)
        for b in sel_set:
            if b <= candidate:
                continue
            for c in sel_set:
                if c < b:
                    continue
                if (b + c) % candidate == 0:
                    return False
        # Case 2: candidate is 'b' or 'c'
        # For all a in sel_set with a < candidate, and b,c where one is candidate
        # Sub-case 2a: a < candidate <= b, a ∤ (candidate + b) for b in sel_set, b >= candidate
        for a in sel_set:
            if a >= candidate:
                continue
            for b in sel_set:
                if b < candidate:
                    continue
                # candidate <= b, check a ∤ (candidate + b)
                if (candidate + b) % a == 0:
                    return False
            # Sub-case 2b: a < b <= candidate, a ∤ (b + candidate)
            # This is the same as above when b <= candidate
            for b in sel_set:
                if b >= candidate:
                    continue
                # a < b < candidate, check a ∤ (b + candidate)
                if (b + candidate) % a == 0:
                    return False
        # Sub-case 2c: candidate = b = c, a < candidate, a ∤ (2*candidate)
        for a in sel_set:
            if a >= candidate:
                continue
            if (2 * candidate) % a == 0:
                return False
        return True
    
    # Greedily try to add numbers from small to large
    for num in range(1, 171):
        if time.time() > deadline:
            break
        if is_feasible(num, selected):
            selected.add(num)
    
    result = sorted(selected)
    return result

# EVOLVE-BLOCK-END

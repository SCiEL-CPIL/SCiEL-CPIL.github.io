"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Generate all 5-smooth numbers <= 128
    smooth = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(i)
    
    smooth_set = set(smooth)
    answers = []
    
    for n in range(1, 129):
        # Check 1 term
        if n in smooth_set:
            answers.append([n])
            continue
            
        # Check 2 terms
        found = None
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b in smooth_set and b > a:
                # Check antichain: a does not divide b and b does not divide a
                # Since b > a, b cannot divide a. So just check a | b
                if b % a != 0:
                    found = [a, b]
                    break
        if found:
            answers.append(found)
            continue
            
        # Check 3 terms
        found = None
        for a in smooth:
            if a >= n:
                break
            for b in smooth:
                if b <= a:
                    continue
                if a + b >= n:
                    break
                c = n - a - b
                if c in smooth_set and c > b:
                    # Check pairwise divisibility
                    # a, b, c are distinct and sorted a < b < c
                    # c cannot divide a or b.
                    # b cannot divide a.
                    # Check: a|b, a|c, b|c
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        found = [a, b, c]
                        break
            if found:
                break
        if found:
            answers.append(found)
            continue
            
        # Check 4 terms (just in case, though likely not needed for 1..128)
        found = None
        # Simple backtracking for 4 terms
        # This is getting complex, but given the previous checks, 
        # it's highly unlikely any n in 1..128 needs 4 terms.
        # Let's add a fallback search for any k up to 5 just to be safe.
        
        def find_antichain(n, k, start_idx, current_terms):
            if len(current_terms) == k:
                if sum(current_terms) == n:
                    # Verify antichain
                    for i in range(len(current_terms)):
                        for j in range(i+1, len(current_terms)):
                            x, y = current_terms[i], current_terms[j]
                            if x % y == 0 or y % x == 0:
                                return None
                    return current_terms[:]
                return None
            if sum(current_terms) > n:
                return None
            
            # Pruning: remaining sum needed
            rem = n - sum(current_terms)
            terms_left = k - len(current_terms)
            if rem < terms_left: # Min term is 1
                return None
            
            for idx in range(start_idx, len(smooth)):
                term = smooth[idx]
                if term > rem:
                    break
                # Check if term is compatible with all in current_terms
                compatible = True
                for t in current_terms:
                    if term % t == 0 or t % term == 0:
                        compatible = False
                        break
                if compatible:
                    res = find_antichain(n, k, idx + 1, current_terms + [term])
                    if res:
                        return res
            return None

        for k in range(4, 6):
            res = find_antichain(n, k, 0, [])
            if res:
                answers.append(res)
                break
                
        if not found:
            # If still not found, there might be an issue, but contract says all targets require valid reps.
            # We assume 4 or 5 terms will find it.
            pass

    return answers

# EVOLVE-BLOCK-END

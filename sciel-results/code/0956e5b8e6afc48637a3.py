from math import gcd

def construct():
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        valid_rs = []
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                valid_rs.append(r)
        
        if not valid_rs:
            continue
            
        # If only one valid r, pick it
        if len(valid_rs) == 1:
            selected.append((m, valid_rs[0]))
            continue
            
        # Otherwise, pick the r that maximizes the number of valid residues for the next modulus m+1
        # Or simply pick the r that minimizes the sum of blocked residues for next few moduli?
        # Let's try: pick r that allows the most valid residues for m+1
        
        best_r = valid_rs[0]
        max_options = -1
        
        next_m = m + 1
        if next_m <= 200:
            for r in valid_rs:
                # Count how many r_next in [0, next_m) are valid given selected + (m, r)
                options = 0
                for r_next in range(next_m):
                    valid_next = True
                    # Check against existing selected
                    for (m_i, r_i) in selected:
                        g = gcd(next_m, m_i)
                        if r_next % g == r_i % g:
                            valid_next = False
                            break
                    if not valid_next:
                        continue
                    # Check against new (m, r)
                    g = gcd(next_m, m)
                    if r_next % g == r % g:
                        valid_next = False
                    if valid_next:
                        options += 1
                if options > max_options:
                    max_options = options
                    best_r = r
        else:
            # If m is 200, just pick any
            best_r = valid_rs[0]

        selected.append((m, best_r))

    return [[m, r] for (m, r) in selected]

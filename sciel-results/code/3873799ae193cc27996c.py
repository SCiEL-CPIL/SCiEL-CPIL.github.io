"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    base = 51975
    # Find all odd divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()  # ascending: smaller moduli cover more
    
    covered = bytearray(base)
    classes = []
    
    # Greedy: for each modulus (smallest first), pick residue covering most uncovered
    for m in divisors:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for x in range(r, base, m):
                if covered[x] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        for x in range(best_r, base, m):
            covered[x] = 1
    
    # Single-modulus local improvement
    improved = True
    while improved:
        improved = False
        for i in range(len(classes)):
            m_i, r_i = classes[i]
            # Find best residue for this modulus given current coverage from others
            # Temporarily remove this class's coverage
            saved = bytearray(covered)
            for x in range(r_i, base, m_i):
                covered[x] = 0
            best_r = r_i
            best_count = -1
            for r in range(m_i):
                count = 0
                for x in range(r, base, m_i):
                    if covered[x] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            # Restore
            covered = saved
            if best_r != r_i:
                classes[i][1] = best_r
                # Update covered
                for x in range(r_i, base, m_i):
                    covered[x] = 0
                for x in range(best_r, base, m_i):
                    covered[x] = 1
                improved = True
    
    return classes
# EVOLVE-BLOCK-END

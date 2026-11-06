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
    n_smooth = len(smooth)
    
    def is_antichain(terms):
        """Check if all pairs in terms form an antichain under divisibility."""
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                x, y = terms[i], terms[j]
                if x > y:
                    x, y = y, x
                if y % x == 0:
                    return False
        return True
    
    answers = []
    
    for n in range(1, 129):
        # Try increasing number of terms, find minimum
        found = None
        
        # k=1
        if n in smooth_set:
            answers.append([n])
            continue
            
        # k=2
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            b = n - a
            if b in smooth_set and b > a:
                if b % a != 0:
                    found = [a, b]
                    break
        if found:
            answers.append(found)
            continue
            
        # k=3
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i+1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c in smooth_set and c > b:
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        found = [a, b, c]
                        break
            if found:
                break
        if found:
            answers.append(found)
            continue
            
        # k=4,5,6... use backtracking
        for k in range(4, 31):
            # Backtracking search
            def backtrack(rem, start_idx, current):
                if len(current) == k:
                    if rem == 0:
                        return current[:]
                    return None
                if rem <= 0:
                    return None
                if k - len(current) > rem:  # min term is 1
                    return None
                
                for idx in range(start_idx, n_smooth):
                    term = smooth[idx]
                    if term > rem:
                        break
                    # Check compatibility
                    ok = True
                    for t in current:
                        if term % t == 0 or t % term == 0:
                            ok = False
                            break
                    if ok:
                        result = backtrack(rem - term, idx + 1, current + [term])
                        if result is not None:
                            return result
                return None
            
            result = backtrack(n, 0, [])
            if result is not None:
                found = result
                break
        
        if found:
            answers.append(found)
        else:
            # Fallback: should not happen per problem statement
            answers.append([n])
    
    return answers

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')
    max_depth = 5

    def dfs(rem, last_d, current_max, path):
        nonlocal best_solution, best_max_denom
        
        if rem == 0:
            if current_max < best_max_denom:
                best_max_denom = current_max
                best_solution = list(path)
            return
        
        if rem < 0:
            return

        if len(path) >= max_depth:
            return

        # Check if rem is a unit fraction
        if rem.numerator == 1:
            d = rem.denominator
            if d > last_d:
                new_max = max(current_max, d)
                if new_max < best_max_denom:
                    best_max_denom = new_max
                    best_solution = list(path) + [d]
            return

        # Min denominator
        d_min = max(last_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Max denominator
        # We want new_max < best_max_denom
        # new_max = max(current_max, d)
        # So d < best_max_denom and current_max < best_max_denom
        d_max = min(best_max_denom - 1, 100000) # Cap to avoid huge ranges if best is inf
        
        if d_min > d_max:
            return

        remaining = max_depth - len(path) - 1
        
        for d in range(d_min, d_max + 1):
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            
            new_max = max(current_max, d)
            if new_max >= best_max_denom:
                break

            # Pruning: lower bound on max denom for the rest
            if remaining > 0:
                # The sum of 'remaining' unit fractions with denominators > d is at most remaining / (d+1)
                # But we need to sum to new_rem.
                # If new_rem > remaining / (d+1), it's impossible? No.
                # If new_rem < 1/(d+1) + ...
                # The smallest sum of 'remaining' distinct unit fractions > d is 1/(d+1) + 1/(d+2) + ...
                # If new_rem < that sum, impossible.
                # The largest denominator in 'remaining' terms summing to new_rem is at least remaining / new_rem.
                # So if max(new_max, remaining / new_rem) >= best_max_denom, prune.
                lb = remaining / float(new_rem)
                if max(new_max, lb) >= best_max_denom:
                    continue

            path.append(d)
            dfs(new_rem, d, new_max, path)
            path.pop()

    dfs(target, 0, 0, [])
    
    if best_solution:
        return best_solution
    
    # Fallback
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END

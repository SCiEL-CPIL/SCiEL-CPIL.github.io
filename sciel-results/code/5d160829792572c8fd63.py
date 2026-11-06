import time
from math import gcd

def construct():
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    # Max denominator: 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    target_num = 5
    target_den = 31
    
    deadline = time.time() + 85.0

    def search_k_terms(k, limit_max):
        """
        Finds a solution of exactly k terms where all denominators are < limit_max.
        Returns the solution with the smallest maximum denominator if found.
        """
        best_local = None
        best_local_max = limit_max
        
        def dfs(rem_num, rem_den, min_d, current, terms_left):
            nonlocal best_local, best_local_max
            
            if time.time() > deadline:
                return False
                
            if rem_num == 0:
                if terms_left == 0 and current:
                    m = max(current)
                    if m < best_local_max:
                        best_local_max = m
                        best_local = current[:]
                        return True
                return False
                
            if terms_left == 0:
                return False
                
            # If we have 1 term left, it must be exactly rem
            if terms_left == 1:
                if rem_num == 1:
                    d = rem_den
                    if min_d <= d < limit_max:
                        current.append(d)
                        m = max(current)
                        if m < best_local_max:
                            best_local_max = m
                            best_local = current[:]
                        current.pop()
                        return True
                return False

            d_start = max(min_d, (rem_den + rem_num - 1) // rem_num)
            
            # If terms_left == 2, solve directly
            if terms_left == 2:
                # 1/d + 1/e = rem_num/rem_den
                # e = d * rem_den / (rem_num * d - rem_den)
                # d must satisfy: d > rem_den/rem_num and e > d
                # e = d*rem_den / (rem_num*d - rem_den)
                # For e > d: d*rem_den > d*(rem_num*d - rem_den)
                # rem_den > rem_num*d - rem_den => 2*rem_den > rem_num*d
                # d < 2*rem_den/rem_num
                d_end = min(limit_max - 1, best_local_max - 1, (2 * rem_den - 1) // rem_num)
                
                for d in range(d_start, d_end + 1):
                    x = rem_num * d - rem_den
                    if x <= 0:
                        continue
                    if (d * rem_den) % x != 0:
                        continue
                    e = (d * rem_den) // x
                    if e <= d:
                        continue
                    if e >= best_local_max:
                        continue
                    current.append(d)
                    current.append(e)
                    m = max(current)
                    if m < best_local_max:
                        best_local_max = m
                        best_local = current[:]
                    current.pop()
                    current.pop()
                    
                return best_local is not None

            # For terms_left > 2, iterate d
            for d in range(d_start, min(limit_max, best_local_max)):
                if time.time() > deadline:
                    return False
                    
                num_rem = rem_num * d - rem_den
                if num_rem <= 0:
                    continue
                
                den_rem = rem_den * d
                
                # Simplify
                g = gcd(num_rem, den_rem)
                if g > 1:
                    num_rem //= g
                    den_rem //= g
                
                # Pruning: upper bound on d from sum constraint
                # We need 1/d + 1/(d+1) + ... + 1/(d+terms_left-1) >= rem_num/rem_den
                # Approximate: terms_left/d >= rem_num/rem_den => d <= terms_left*rem_den/rem_num
                d_bound = (terms_left * rem_den) // rem_num
                if d > d_bound:
                    break

                # Pruning: min possible sum of remaining terms
                # Max sum of (terms_left-1) terms with denominators >= d+1 is
                # (terms_left-1)/(d+1) (when all are d+1, but they must be distinct)
                # Actually, the maximum sum with distinct denominators >= d+1 is
                # 1/(d+1) + 1/(d+2) + ... + 1/(d+terms_left-1)
                # For a quick upper bound: (terms_left-1)/(d+1)
                # We need: num_rem/den_rem <= (terms_left-1)/(d+1)
                # num_rem*(d+1) <= (terms_left-1)*den_rem
                if num_rem * (d + 1) > (terms_left - 1) * den_rem:
                    continue

                # Also need minimum: even with the smallest possible denominators,
                # we can't reach the target. The minimum sum of (terms_left-1) distinct
                # denominators >= d+1 is approximately 1/(d+1) + 1/(d+2) + ...
                # But a simpler lower bound: each term <= 1/(d+1), so sum <= (terms_left-1)/(d+1)
                # This is the same as above.

                # Prune if d >= best_local_max
                if d >= best_local_max:
                    break

                current.append(d)
                dfs(num_rem, den_rem, d + 1, current, terms_left - 1)
                current.pop()
                
                # If we found a better solution, best_local_max decreased,
                # so the loop bound min(limit_max, best_local_max) will handle it

            return best_local is not None

        dfs(target_num, target_den, 1, [], k)
        
        if best_local:
            return best_local
        return None

    # Search for 2-term solutions
    sol = search_k_terms(2, best_max)
    if sol:
        return sol

    # Search for 3-term solutions
    sol = search_k_terms(3, best_max)
    if sol:
        return sol

    # Search for 4-term solutions
    sol = search_k_terms(4, best_max)
    if sol:
        return sol

    # Search for 5-term solutions
    sol = search_k_terms(5, best_max)
    if sol:
        return sol

    # Search for 6-term solutions
    sol = search_k_terms(6, best_max)
    if sol:
        return sol

    # Search for 7-term solutions
    sol = search_k_terms(7, best_max)
    if sol:
        return sol

    # Search for 8-term solutions
    sol = search_k_terms(8, best_max)
    if sol:
        return sol

    # If no better solution found, return the known one
    return best_solution

if __name__ == "__main__":
    print(construct())

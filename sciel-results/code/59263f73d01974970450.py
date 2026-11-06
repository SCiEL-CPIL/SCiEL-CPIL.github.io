"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from heapq import heappush, heappop

def construct():
    deadline = time.time() + 55
    target_num = 5
    target_den = 31
    
    # Known fallback solution
    best_max = 248
    best_solution = [8, 31, 248]
    
    # Dijkstra-like search: state = (num, den) of remaining fraction (reduced)
    # Edge: subtract 1/k for k >= current_min_den
    # Cost: maximize the denominators used (we minimize the max)
    # We search for expansions with max denominator < best_max
    
    def search(max_den_limit):
        """Find Egyptian fraction of target_num/target_den with all denominators <= max_den_limit.
        Returns list of denominators or None."""
        # State: (num, den) reduced fraction remaining
        # We build up the expansion, tracking denominators used
        # Use BFS with pruning
        
        start = (target_num, target_den)
        if start[0] > start[1]:
            # target > 1, need to handle but 5/31 < 1 so ok
            pass
        
        # BFS: (num, den, denominators_used)
        # Prune: if num/den > sum of 1/k for k in [min_k, max_den_limit] with enough terms, prune
        # Actually, let's use a smarter approach:
        # For remaining fraction num/den, the next denominator k must satisfy:
        #   1/k <= num/den  =>  k >= den/num
        #   and we need num/den - 1/k to be representable with denominators in [k, max_den_limit]
        
        # Use iterative deepening on max denominator
        # Actually, let's do a direct search for small max_den values
        
        visited = set()
        # Priority queue: (max_den_so_far, num, den, path)
        # We want to find ANY solution, so BFS by depth (number of terms)
        # But we want to minimize max denominator, so let's try increasing max_den
        
        for max_k in range(7, max_den_limit + 1):
            # Try to find expansion of target_num/target_den using denominators in [1, max_k]
            # with distinct denominators, at least 7 (since 5/31 > 1/7)
            # Actually denominators must be >= ceil(den/num) for the first term
            
            min_first = (target_den + target_num - 1) // target_num  # ceil(den/num)
            
            # BFS over (num, den, next_min_k, path)
            queue = [(target_num, target_den, min_first, [])]
            seen = set()
            
            while queue:
                num, den, next_min, path = queue.pop(0)
                
                if num == 0:
                    return path
                
                if num < 0:
                    continue
                
                # Reduce fraction
                g = gcd(num, den)
                num_r, den_r = num // g, den // g
                
                state = (num_r, den_r, next_min)
                if state in seen:
                    continue
                seen.add(state)
                
                # If num_r/den_r == 0, done (handled above)
                if num_r == 0:
                    return path
                
                # The next denominator k must satisfy:
                # 1/k <= num_r/den_r  =>  k >= ceil(den_r/num_r)
                # and k >= next_min (for ordering/distinctness)
                # and k <= max_k
                k_min = max(next_min, (den_r + num_r - 1) // num_r)
                
                # Also, we need num_r/den_r - 1/k >= 0
                # and the remaining fraction must be expressible
                # Upper bound on k: we need enough "room" for remaining terms
                # For now, just iterate k from k_min to max_k
                
                for k in range(k_min, max_k + 1):
                    # New remaining: num_r/den_r - 1/k = (num_r*k - den_r) / (den_r*k)
                    new_num = num_r * k - den_r
                    new_den = den_r * k
                    if new_num < 0:
                        break  # k too large, 1/k > num_r/den_r
                    # new_num == 0 means we're done
                    if new_num == 0:
                        return path + [k]
                    # Prune: if new_num/new_den > 1/(k+1) + 1/(k+2) + ... (max possible with remaining terms)
                    # For simplicity, just continue BFS
                    queue.append((new_num, new_den, k + 1, path + [k]))
            
            # If we exhausted BFS for this max_k without finding solution, try next max_k
            # But this is very slow. Let's be smarter.
        
        return None
    
    # The BFS above is too slow. Let's use a more targeted approach.
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # Enumerate c from 7 upward, and for each c, solve 1/a + 1/b = 5/31 - 1/c
    
    for c in range(7, best_max):
        if time.time() > deadline:
            break
        # 1/a + 1/b = 5/31 - 1/c = (5c - 31)/(31c)
        p = 5 * c - 31
        q = 31 * c
        if p <= 0:
            continue
        g = gcd(p, q)
        p_r, q_r = p // g, q // g
        
        # Solve 1/a + 1/b = p_r/q_r: (p_r*a - q_r)(p_r*b - q_r) = q_r^2
        q2 = q_r * q_r
        # Find divisors u of q2 such that u <= q2 and (u+q_r) % p_r == 0 and (q2/u + q_r) % p_r == 0
        # u = p_r*a - q_r, so a = (u + q_r) // p_r
        # We need a >= 7, a <= b
        
        # Enumerate divisors of q2
        # q2 can be large, so let's be smart
        # u must divide q2, and u = p_r*a - q_r for some integer a >= 7
        # So u >= p_r*7 - q_r and u <= p_r*b - q_r where b can be up to... let's say c (for ordering a<=b<=c)
        
        u_min = p_r * 7 - q_r
        if u_min < 1:
            u_min = 1
        u_max = p_r * c - q_r  # since b <= c
        if u_max <= 0:
            continue
        
        # Find divisors of q2 in [u_min, u_max]
        # Iterate over possible u values that divide q2
        # Instead of iterating all u, iterate over divisors of q2
        # But q2 can be huge. Let's iterate u from u_min to u_max and check divisibility.
        # This is O(c) per c, which is O(c^2) total. For c up to 248, that's ~60k iterations. Fine.
        
        for u in range(u_min, u_max + 1):
            if q2 % u != 0:
                continue
            v = q2 // u
            if (u + q_r) % p_r != 0 or (v + q_r) % p_r != 0:
                continue
            a = (u + q_r) // p_r
            b = (v + q_r) // p_r
            if a < 7 or a > b or b > c:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == Fraction(5, 31):
                best_max = c
                best_solution = [a, b, c]
                break
        if best_max == c:
            break
    
    # Try 4 terms if we found a 3-term solution
    for d in range(7, best_max):
        if time.time() > deadline:
            break
        # 1/a + 1/b + 1/c = 5/31 - 1/d = (5d-31)/(31d)
        P = 5 * d - 31
        Q = 31 * d
        if P <= 0:
            continue
        g = gcd(P, Q)
        P_r, Q_r = P // g, Q // g
        
        # Now find 1/a + 1/b + 1/c = P_r/Q_r with 7 <= a <= b <= c <= d
        # Enumerate c, then solve 1/a + 1/b = P_r/Q_r - 1/c
        
        c_min = max(7, (Q_r + P_r - 1) // P_r)  # ceil(Q_r/P_r)
        for c in range(c_min, d + 1):
            if time.time() > deadline:
                break
            p2 = P_r * c - Q_r
            if p2 <= 0:
                continue
            q2 = Q_r * c
            g2 = gcd(p2, q2)
            p2_r, q2_r = p2 // g2, q2 // g2
            
            # Solve 1/a + 1/b = p2_r/q2_r with 7 <= a <= b <= c
            q2_2 = q2_r * q2_r
            u_min = p2_r * 7 - q2_r
            if u_min < 1:
                u_min = 1
            u_max = p2_r * c - q2_r
            if u_max <= 0:
                continue
            
            for u in range(u_min, u_max + 1):
                if q2_2 % u != 0:
                    continue
                v = q2_2 // u
                if (u + q2_r) % p2_r != 0 or (v + q2_r) % p2_r != 0:
                    continue
                a = (u + q2_r) // p2_r
                b = (v + q2_r) // p2_r
                if a < 7 or a > b or b > c:
                    continue
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == Fraction(5, 31):
                    best_max = d
                    best_solution = [a, b, c, d]
                    break
            if best_max == d:
                break
        if best_max == d:
            break
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != Fraction(5, 31):
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END

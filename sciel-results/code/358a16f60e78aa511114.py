import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 50.0 # Safety margin
    
    # Moduli 2 to 128
    # We hypothesize the max number of classes is achieved by using all even moduli.
    # Let's try to assign residues to all even moduli.
    
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # Sort moduli by number of divisors or just value?
    # Smaller moduli have fewer constraints? No, they have smaller gcds with others?
    # Actually, moduli with many factors have larger gcds with more numbers, making it harder to be disjoint?
    # Or easier?
    # If gcd is large, the condition r != r_i mod g is stricter?
    # No, r != r_i mod g means r - r_i is not a multiple of g.
    # If g is large, there are fewer multiples of g in the range [0, m-1].
    # So it's EASIER to satisfy if g is large?
    # Let's just try a greedy order.
    
    # We will try multiple random orders and residue assignments.
    best_solution = []
    
    def is_compatible(r, m, current_pairs):
        for r_i, m_i in current_pairs:
            g = __import__('math').gcd(m, m_i)
            if (r - r_i) % g == 0:
                return False
        return True

    def solve_greedy(order):
        pairs = []
        for m in order:
            found = False
            # Try residues 0 to m-1
            # Optimization: If we fail, we can break.
            for r in range(m):
                if is_compatible(r, m, pairs):
                    pairs.append((r, m))
                    found = True
                    break
            if not found:
                return None
        return pairs

    # Try all even moduli
    # Randomize order to escape local minima?
    # Actually, if a solution exists for the set, a greedy approach might fail if the order is bad.
    # But for this specific structure (all even), a solution likely exists.
    
    # Let's try a specific constructive assignment first.
    # If m = 2^k * u, u odd.
    # Try r = 2^k - 1? No, that failed for powers of 2.
    
    # Let's use randomized search with the even moduli.
    random.seed(42)
    
    for _ in range(100):
        if time.time() - start_time > max_time:
            break
            
        order = moduli[:]
        random.shuffle(order)
        sol = solve_greedy(order)
        if sol:
            # Convert to list of lists [r, m]
            return [[r, m] for r, m in sol]
            
    # If greedy failed, try a smaller set?
    # Or just return the best found so far?
    # The problem asks to maximize. If we can't find 64, we should try to find as many as possible.
    
    # Fallback: Powers of 2 with correct residues?
    # We know powers of 2 can be made disjoint.
    # Let's construct a valid set of powers of 2.
    # m_i = 2^i.
    # We need r_i != r_j mod 2^min(i,j).
    # Let's build it greedily for powers of 2.
    p2_moduli = [2**i for i in range(1, 8)] # 2 to 128
    p2_sol = []
    for m in p2_moduli:
        found = False
        for r in range(m):
            if is_compatible(r, m, p2_sol):
                p2_sol.append((r, m))
                found = True
                break
        if not found:
            break
            
    if p2_sol:
        return [[r, m] for r, m in p2_sol]
        
    # Absolute fallback: single class
    return [[0, 2]]

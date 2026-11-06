import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    moduli = list(range(2, 129))

    # Precompute gcds
    gcd_val = [[0] * 129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcd_val[i][j] = math.gcd(i, j)
            gcd_val[j][i] = gcd_val[i][j]

    # Precompute for each m, for each possible g (1..128), the list of residues r in [0,m) 
    # that are congruent to specific values mod g. Actually, let's think differently.
    
    # For a given m and a constraint "r != a (mod g)", the forbidden residues are those 
    # where r % g == a % g. The number of such residues in [0, m) is either floor(m/g) or ceil(m/g).
    
    # Let's precompute: for each m and each g (1..m), for each residue class a mod g,
    # how many r in [0,m) satisfy r % g == a.
    
    # Actually, a simpler approach: for each m, precompute an array residue_mod_g[m][g][a] 
    # = list of r in [0,m) such that r % g == a. But this might be too much memory.
    
    # Alternative: just compute on the fly but more efficiently.
    
    def try_add(m, current_set):
        """
        current_set: list of (r, m) tuples.
        Returns a valid residue r for m if possible, else None.
        """
        if not current_set:
            return 0

        # For each existing (r_j, m_j), we have constraint: r != r_j (mod gcd(m, m_j))
        # Let's collect all forbidden residue classes.
        
        # For efficiency, let's track which residues are forbidden.
        # A residue r is forbidden if for some (r_j, m_j), (r - r_j) % gcd(m, m_j) == 0.
        
        # We can use a boolean array of size m.
        forbidden = [False] * m
        
        for r_j, m_j in current_set:
            g = gcd_val[m][m_j]
            if g == 1:
                # No constraint since gcd=1 means always disjoint
                continue
            # r must not satisfy r % g == r_j % g
            target = r_j % g
            # Mark all r in [0, m) where r % g == target as forbidden
            for r in range(target, m, g):
                if not forbidden[r]:
                    forbidden[r] = True
        
        # Find the first unforbidden residue
        for r in range(m):
            if not forbidden[r]:
                return r
        return None

    def greedy(order):
        current_set = []
        for m in order:
            r = try_add(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    best_set = []
    best_size = 0

    def count_divisors(n):
        cnt = 0
        for i in range(2, n + 1):
            if n % i == 0:
                cnt += 1
        return cnt

    heuristic_orders = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: -count_divisors(x)),
        sorted(moduli, key=lambda x: count_divisors(x)),
        sorted(moduli, key=lambda x: math.gcd(x, 128)),  # More sharing with 128 first?
        sorted(moduli, key=lambda x: -math.gcd(x, 128)),
    ]

    # Run heuristic orders first
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts with the improved try_add
    def try_add_random(m, current_set, rng):
        if not current_set:
            return 0

        forbidden = [False] * m
        
        for r_j, m_j in current_set:
            g = gcd_val[m][m_j]
            if g == 1:
                continue
            target = r_j % g
            for r in range(target, m, g):
                if not forbidden[r]:
                    forbidden[r] = True
        
        valid_rs = [r for r in range(m) if not forbidden[r]]
        if valid_rs:
            return rng.choice(valid_rs)
        return None

    def greedy_random(order, rng):
        current_set = []
        for m in order:
            r = try_add_random(m, current_set, rng)
            if r is not None:
                current_set.append((r, m))
        return current_set

    # Random restarts
    num_restarts = 20000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random_order = moduli.copy()
        random.shuffle(random_order)

        if i % 2 == 0:
            s = greedy(random_order)
        else:
            s = greedy_random(random_order, random)

        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search: try to improve by re-assigning residues
    # For each modulus in the set, try to change its residue to allow adding a new one.
    def local_search(current_set, time_limit):
        """Try to improve the set by re-assigning residues."""
        current_set = list(current_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add each missing modulus by re-assigning residues
            current_moduli = {m for r, m in current_set}
            for m_new in moduli:
                if m_new in current_moduli:
                    continue
                # Try to add m_new by re-assigning residues of existing moduli
                # Simple approach: try all possible residues for m_new, and if none work,
                # try changing one existing residue
                r_new = try_add(m_new, current_set)
                if r_new is not None:
                    current_set.append((r_new, m_new))
                    improved = True
                    break
                # Try changing one existing residue to make room
                for idx, (r_old, m_old) in enumerate(current_set):
                    # Try all possible new residues for m_old
                    for r_try in range(m_old):
                        if r_try == r_old:
                            continue
                        # Check if r_try is valid for m_old given other moduli
                        valid = True
                        for r_j, m_j in current_set:
                            if m_j == m_old:
                                continue
                            g = gcd_val[m_old][m_j]
                            if (r_try - r_j) % g == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Temporarily replace
                        old_pair = current_set[idx]
                        current_set[idx] = (r_try, m_old)
                        # Try to add m_new
                        r_new2 = try_add(m_new, current_set)
                        if r_new2 is not None:
                            current_set.append((r_new2, m_new))
                            improved = True
                            break
                        # Restore
                        current_set[idx] = old_pair
                    if improved:
                        break
            if not improved:
                break
        return current_set

    # Apply local search to best_set
    if best_set and time.time() < deadline - 5:
        improved_set = local_search(best_set, deadline - 2)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    result = [[r, m] for r, m in best_set]

    if not result:
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return result

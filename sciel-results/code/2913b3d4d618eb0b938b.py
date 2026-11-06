"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute lcm for all pairs using a 2D list for speed
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    def get_lcm(a, b):
        return lcm_table[a][b]
    
    # Precompute for each x, which pairs (a,b) in [1..N] would form a bad triple with x
    # i.e., lcm(a,x) == lcm(b,x) == lcm(a,b)
    # For each x, group elements by their LCM with x
    lcm_groups = {}
    for x in range(1, N + 1):
        groups = defaultdict(list)
        for y in range(1, N + 1):
            if y != x:
                groups[get_lcm(x, y)].append(y)
        lcm_groups[x] = groups
    
    def has_bad_triple_with_x(selected, x):
        """Check if adding x to selected creates a bad triple using precomputed groups."""
        sel_set = set(selected)
        # For each LCM value L, if two selected elements a,b have lcm(a,x)=L and lcm(b,x)=L
        # then we need lcm(a,b) == L
        groups = lcm_groups[x]
        for L, elems in groups.items():
            # Filter to only selected elements
            sel_in_group = [e for e in elems if e in sel_set]
            if len(sel_in_group) < 2:
                continue
            # Check all pairs in this group
            n = len(sel_in_group)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = sel_in_group[i], sel_in_group[j]
                    if get_lcm(a, b) == L:
                        return True
        return False
    
    def is_valid_set(selected):
        """Full validation that no bad triple exists in selected."""
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = selected[i], selected[j], selected[k]
                    if get_lcm(a, b) == get_lcm(a, c) == get_lcm(b, c):
                        return False
        return True
    
    def greedy(order):
        """Greedy construction with given element order."""
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in selected_set:
                if not has_bad_triple_with_x(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected
    
    # Precompute omega (total number of prime factors with multiplicity)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    
    omega_vals = {x: omega(x) for x in nums}
    
    def is_pow2(x):
        return (x & (x - 1)) == 0
    
    # Try multiple orderings and keep the best
    best = []
    
    orderings = [
        ("asc", nums),
        ("desc", nums[::-1]),
        ("odd_first", sorted(nums, key=lambda x: (x & 1, x))),
        ("omega_asc", sorted(nums, key=omega_vals.get)),
        ("pow2_first", sorted(nums, key=lambda x: (not is_pow2(x), x))),
        ("odd_desc", sorted(nums, key=lambda x: (x & 1, -x))),
        ("omega_desc", sorted(nums, key=lambda x: -omega_vals[x])),
        ("largest_first", sorted(nums, reverse=True)),
        ("prime_like_first", sorted(nums, key=lambda x: (omega_vals[x] == 1, x))),
        ("composite_first", sorted(nums, key=lambda x: (omega_vals[x] > 1, x))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline - 5:
            break
        s = greedy(order)
        if len(s) > len(best):
            best = list(s)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Local search: try adding remaining elements (multiple passes)
    improved = list(best)
    improved_set = set(improved)
    
    for _pass in range(3):
        if time.time() > deadline - 5:
            break
        remaining = [x for x in nums if x not in improved_set]
        added_any = False
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if not has_bad_triple_with_x(improved, x):
                improved.append(x)
                improved_set.add(x)
                added_any = True
        if not added_any:
            break
    
    if len(improved) > len(best):
        best = improved
    
    # Simulated annealing on the QUBO-style energy landscape
    # Energy = -cardinality + PENALTY * (number of bad triples)
    # We use the conflict structure to efficiently evaluate moves.
    
    import random
    random.seed(42)
    
    PENALTY = 100  # Large penalty per bad triple
    
    def count_bad_triples(selected_list):
        """Count number of bad triples in the selected set."""
        n = len(selected_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = selected_list[i], selected_list[j], selected_list[k]
                    if get_lcm(a, b) == get_lcm(a, c) == get_lcm(b, c):
                        count += 1
        return count
    
    def energy(selected_list):
        """QUBO energy: -|S| + PENALTY * (# bad triples)."""
        return -len(selected_list) + PENALTY * count_bad_triples(selected_list)
    
    # Start from current best
    sa_selected = list(best)
    sa_set = set(sa_selected)
    current_energy = energy(sa_selected)
    best_sa = list(sa_selected)
    best_sa_energy = current_energy
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    n_iterations = 5000
    anneal_deadline = deadline - 10
    
    for it in range(n_iterations):
        if time.time() > anneal_deadline:
            break
        
        # Temperature decay (geometric)
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.33:
            # Try to add a random element not in set
            candidates = [x for x in nums if x not in sa_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_list = sa_selected + [x]
            new_set = sa_set | {x}
        elif move_type < 0.66:
            # Try to remove a random element from set
            if not sa_selected:
                continue
            idx = random.randrange(len(sa_selected))
            x = sa_selected[idx]
            new_list = sa_selected[:idx] + sa_selected[idx+1:]
            new_set = sa_set - {x}
        else:
            # Swap: remove one, add one
            if not sa_selected:
                continue
            candidates = [x for x in nums if x not in sa_set]
            if not candidates:
                continue
            idx = random.randrange(len(sa_selected))
            x_remove = sa_selected[idx]
            x_add = random.choice(candidates)
            new_list = sa_selected[:idx] + sa_selected[idx+1:] + [x_add]
            new_set = sa_set - {x_remove} | {x_add}
        
        # Evaluate new energy
        new_energy = energy(new_list)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or (T > 0 and random.random() < min(1.0, 2.718281828 ** (-delta_e / T))):
            sa_selected = new_list
            sa_set = new_set
            current_energy = new_energy
            
            if current_energy < best_sa_energy:
                best_sa_energy = current_energy
                best_sa = list(sa_selected)
    
    # Use the best found by SA if it's better
    if len(best_sa) > len(best):
        best = list(best_sa)
    
    # Final greedy pass on remaining
    if time.time() < deadline - 5:
        final_set = set(best)
        remaining = [x for x in nums if x not in final_set]
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if not has_bad_triple_with_x(best, x):
                best.append(x)
                final_set.add(x)
    
    # Final validation
    if not is_valid_set(best):
        # Fallback to greedy
        best = greedy(nums)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END

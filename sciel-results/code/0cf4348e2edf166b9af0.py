import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    N = 128
    
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count("1")

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad triples bitmask B[i][j]
    # B[i][j] is a mask of k such that {i, j, k} is a bad triple.
    B = [[0] * (N + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            B[i][j] = mask
            B[j][i] = mask

    def get_bad_triples_count(S_mask):
        """
        Calculate the number of bad triples in the set represented by S_mask.
        We count each triple {a,b,c} once.
        Iterate a < b < c.
        """
        count = 0
        elements = [i for i in range(1, N + 1) if (S_mask >> i) & 1]
        n = len(elements)
        for i_idx in range(n):
            a = elements[i_idx]
            for j_idx in range(i_idx + 1, n):
                b = elements[j_idx]
                # Count c in S such that c > b and {a,b,c} is bad
                # B[a][b] contains c if {a,b,c} is bad.
                # We need to intersect with S_mask and count bits > b?
                # No, B[a][b] contains all k.
                # If k is in S, and k != a, k != b, it's a bad triple.
                # Since a,b are in S, and B[a][b] does not contain a or b (we skipped them in precomp),
                # popcount(B[a][b] & S_mask) gives the number of such k.
                # However, this counts the triple for each pair (a,b).
                # A triple {a,b,c} is counted when we look at pair (a,b) -> finds c.
                # It is also counted when we look at pair (a,c) -> finds b.
                # It is also counted when we look at pair (b,c) -> finds a.
                # So summing over all pairs in S counts each bad triple 3 times.
                count += popcount(B[a][b] & S_mask)
        return count // 3

    def get_add_cost(x, S_mask):
        """
        Calculate the number of NEW bad triples created by adding x to S_mask.
        New triples are {x, u, v} where u, v in S_mask.
        This is the number of pairs {u, v} in S_mask such that {x, u, v} is bad.
        For a fixed x, iterate u in S_mask.
        If v in S_mask and v in B[x][u], then {x, u, v} is bad.
        Sum over u in S_mask of popcount(B[x][u] & S_mask).
        Each pair {u, v} is counted twice (once for u, once for v).
        So result is 2 * (number of new bad triples).
        We want the count of triples, so divide by 2.
        """
        if (S_mask >> x) & 1:
            return 0 # Already in set
        count = 0
        temp = S_mask
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            # Count v in S_mask such that v in B[x][u]
            count += popcount(B[x][u] & S_mask)
            temp ^= lsb
        return count // 2

    def get_remove_cost(x, S_mask):
        """
        Calculate the number of bad triples REMOVED by removing x from S_mask.
        These are triples {x, u, v} where u, v in S_mask \ {x}.
        This is the same as get_add_cost(x, S_mask \ {x}).
        """
        S_without_x = S_mask & ~(1 << x)
        return get_add_cost(x, S_without_x)

    # Initial construction using Greedy
    def greedy_build(order):
        s = 0
        for x in order:
            # Check if adding x creates bad triples
            if get_add_cost(x, s) == 0:
                s |= (1 << x)
        return s

    random.seed(42)
    
    # Generate some initial sets
    best_set = 0
    best_size = 0
    best_energy = float('inf')
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: random.random()),
        sorted(range(1, N + 1), key=lambda x: random.random()),
    ]
    
    for order in orders:
        s = greedy_build(order)
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s

    # Simulated Annealing
    # Energy = 1000 * bad_triples - size
    # We want to minimize Energy.
    
    current_set = best_set
    current_bad = get_bad_triples_count(current_set)
    current_size = bin(current_set).count("1")
    current_energy = 1000 * current_bad - current_size
    
    T = 20.0 # Initial temperature
    cooling_rate = 0.99
    
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = [x for x in all_elements if not ((current_set >> x) & 1)]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = current_set | (1 << x)
            new_bad = current_bad + get_add_cost(x, current_set)
            new_size = current_size + 1
            new_energy = 1000 * new_bad - new_size
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                current_set = new_set
                current_bad = new_bad
                current_size = new_size
                current_energy = new_energy
                
                if current_bad == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                    
        elif move_type < 0.8:
            # Remove
            elements_in_s = [x for x in all_elements if (current_set >> x) & 1]
            if not elements_in_s:
                continue
            x = random.choice(elements_in_s)
            new_set = current_set & ~(1 << x)
            new_bad = current_bad - get_remove_cost(x, current_set)
            new_size = current_size - 1
            new_energy = 1000 * new_bad - new_size
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                current_set = new_set
                current_bad = new_bad
                current_size = new_size
                current_energy = new_energy
        else:
            # Swap
            elements_in_s = [x for x in all_elements if (current_set >> x) & 1]
            elements_not_in_s = [x for x in all_elements if not ((current_set >> x) & 1)]
            if not elements_in_s or not elements_not_in_s:
                continue
            u = random.choice(elements_in_s)
            v = random.choice(elements_not_in_s)
            
            # Remove u, add v
            # Cost = Cost(remove u) + Cost(add v to set without u)
            # Be careful with order.
            set_without_u = current_set & ~(1 << u)
            cost_remove_u = get_remove_cost(u, current_set)
            cost_add_v = get_add_cost(v, set_without_u)
            
            new_bad = current_bad - cost_remove_u + cost_add_v
            new_size = current_size # Size unchanged
            new_energy = 1000 * new_bad - new_size
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                current_set = set_without_u | (1 << v)
                current_bad = new_bad
                # current_size unchanged
                current_energy = new_energy
                
                if current_bad == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = current_set

        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Final Greedy Polish
    # Try to add any remaining elements that don't create bad triples
    for x in all_elements:
        if not ((best_set >> x) & 1):
            if get_add_cost(x, best_set) == 0:
                best_set |= (1 << x)
    
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    if not result:
        return [1]
        
    return result

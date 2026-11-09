import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute for each pair (a,b), the set of x that would form a bad triple with them
    # A triple (x, a, b) is bad if gcd(x,a) == gcd(x,b) == gcd(a,b)
    # For a given pair (a,b) with g = gcd(a,b), x is bad if gcd(x,a) == g and gcd(x,b) == g
    # This is equivalent to: g | x, and gcd(x/g, a/g) == 1, and gcd(x/g, b/g) == 1
    # Actually, let's just precompute which x are forbidden for each pair
    
    # bad_for_pair[a][b] = set of x (x != a, x != b) such that {x,a,b} is a bad triple
    bad_for_pair = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_table[x][a] == g_ab and gcd_table[x][b] == g_ab:
                    bad_for_pair[a][b].add(x)
                    bad_for_pair[b][a].add(x)
    
    def is_valid_set(selected):
        """Check if selected is a valid set (no bad triples)."""
        selected_set = set(selected)
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                # Check if any element in selected is in bad_for_pair[a][b]
                # This is O(k) per pair, so O(k^3) total - too slow for large k
                # Instead, we check during construction
                pass
        return True
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        selected_set = set(selected)
        for a in selected:
            if x in bad_for_pair[a][x]:
                # This means there exists b in selected such that {x,a,b} is bad
                # Wait, bad_for_pair[a][x] contains all z such that {z,a,x} is bad
                # So if x is in bad_for_pair[a][z] for some z in selected... 
                # Let me re-check the definition
                pass
        # Actually, let's just check directly:
        # x can be added if for all pairs (a,b) in selected, x is not in bad_for_pair[a][b]
        # AND for all a in selected, the pair (x,a) doesn't create a bad triple with any b in selected
        # The second condition is: for all a in selected, for all b in selected with b != a,
        #   x is not in bad_for_pair[a][b]
        # This is the same as: for all pairs (a,b) in selected, x not in bad_for_pair[a][b]
        
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if x in bad_for_pair[a][b]:
                    return False
        return True
    
    def can_add_fast(x, selected, selected_set):
        """Faster check using precomputed bad pairs."""
        for a in selected:
            # For each a in selected, check if adding x creates a bad triple with any b in selected
            # A bad triple {x, a, b} exists if x in bad_for_pair[a][b] for some b in selected
            # We can't easily speed this up without more precomputation
            pass
        # Let's use a different approach: for each a in selected, 
        # the set of b that would form a bad triple with x and a is bad_for_pair[x][a]
        # So we need: bad_for_pair[x][a] ∩ selected_set == empty for all a in selected
        for a in selected:
            if bad_for_pair[x][a] & selected_set:
                return False
        return True
    
    # Greedy construction with multiple orderings
    import random
    rng = random.Random(12345)
    
    best_set = []
    best_size = 0
    
    # Try several different ordering strategies
    orderings = []
    
    # Natural order
    orderings.append(list(range(1, n + 1)))
    
    # Reverse order
    orderings.append(list(range(n, 0, -1)))
    
    # Prime-heavy order (primes are "safe" to include)
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in range(1, n + 1) if is_prime(x)]
    non_primes = [x for x in range(1, n + 1) if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # Random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if x not in selected_set:
                if can_add_fast(x, selected, selected_set):
                    selected.append(x)
                    selected_set.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search: try to improve by swapping elements
    if time.time() < deadline - 5:
        current_set = best_set[:]
        current_set_set = set(current_set)
        current_size = len(current_set)
        no_improvement_count = 0
        max_no_improvement = 100
        
        while no_improvement_count < max_no_improvement and time.time() < deadline - 2:
            improved = False
            
            # Try adding each missing element
            for x in range(1, n + 1):
                if time.time() > deadline - 2:
                    break
                if x not in current_set_set:
                    if can_add_fast(x, current_set, current_set_set):
                        current_set.append(x)
                        current_set_set.add(x)
                        current_size += 1
                        improved = True
                        no_improvement_count = 0
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one element, add another
            for i in range(len(current_set)):
                if time.time() > deadline - 2:
                    break
                a = current_set[i]
                # Try removing a and adding some x not in set
                temp_set = [x for x in current_set if x != a]
                temp_set_set = set(temp_set)
                
                # Try each missing element
                for x in range(1, n + 1):
                    if x in current_set_set:
                        continue
                    if can_add_fast(x, temp_set, temp_set_set):
                        # This swap is valid
                        current_set = temp_set + [x]
                        current_set_set = temp_set_set | {x}
                        no_improvement_count = 0
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline:
        best_set_list = best_set[:]
        best_set_set = set(best_set_list)
        
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x not in best_set_set:
                if can_add_fast(x, best_set_list, best_set_set):
                    best_set_list.append(x)
                    best_set_set.add(x)
        
        best_set = best_set_list
    
    return sorted(best_set)

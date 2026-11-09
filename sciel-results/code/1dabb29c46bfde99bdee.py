import random
from math import lcm

def construct():
    max_n = 128
    # Precompute LCM table
    lcm_table = [[1]*(max_n+1) for _ in range(max_n+1)]
    for i in range(1, max_n+1):
        for j in range(i, max_n+1):
            l = lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Helper to count bad triples involving x and a set S (x not in S)
    # Returns count of pairs (a,b) in S such that L(a,b) == L(a,x) == L(b,x)
    def count_bad_triples_with_x(x, S):
        count = 0
        # Optimization: if S is a list, iterating is fine.
        # We can store S as a set for O(1) lookups? No, we need pairs.
        # Just iterate.
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            lax = lcm_table[a][x]
            # If L(a,x) is very large, it's less likely to match L(a,b) for b in S?
            # Actually, L(a,b) <= L(a,x) is not necessarily true.
            # Just check all b.
            for j in range(i+1, n):
                b = s_list[j]
                # Check if L(a,b) == L(a,x) and L(b,x) == L(a,x)
                # Note: L(b,x) must also be checked.
                if lcm_table[a][b] == lax and lcm_table[b][x] == lax:
                    count += 1
        return count

    # Initial solution: Greedy
    def greedy_select(order):
        selected = []
        for x in order:
            ok = True
            s_list = selected
            n = len(s_list)
            for i in range(n):
                a = s_list[i]
                lax = lcm_table[a][x]
                for j in range(i+1, n):
                    b = s_list[j]
                    if lcm_table[a][b] == lax and lcm_table[b][x] == lax:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected

    # Get a good starting point
    best_initial = []
    # Try a few random orders for greedy
    for _ in range(20):
        random.seed(42 + _)
        ord = list(range(1, 129))
        random.shuffle(ord)
        s = greedy_select(ord)
        if len(s) > len(best_initial):
            best_initial = s
    
    # Also try sorted by prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
            
    ord = sorted(range(1, 129), key=num_prime_factors)
    s = greedy_select(ord)
    if len(s) > len(best_initial):
        best_initial = s
        
    S = set(best_initial)
    all_elements = set(range(1, 129))
    
    # Simulated Annealing
    # Cost = -|S| + lambda * violations
    # We want to minimize Cost.
    # Start with valid set, so violations = 0.
    # T starts high, cools down.
    
    # We need to track violations count to update efficiently?
    # Or just re-calculate? 
    # Re-calculation of total violations is O(|S|^3). Too slow.
    # But we can calculate Delta Violations for a move in O(|S|^2).
    # We maintain current_violations.
    
    current_violations = 0 # Since S is valid from greedy
    current_size = len(S)
    current_cost = -current_size + 1000 * current_violations
    
    best_S = set(S)
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    iterations = 20000 # Adjust based on speed
    
    for it in range(iterations):
        # Choose move type
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_S = S.copy()
        delta_violations = 0
        delta_size = 0
        
        if move_type == 'add':
            candidates = all_elements - S
            if not candidates:
                continue
            x = random.choice(list(candidates))
            # Calculate violations introduced by x
            # These are triples (a,b,x) with a,b in S
            v_add = count_bad_triples_with_x(x, S)
            delta_violations = v_add
            delta_size = 1
            new_S.add(x)
            
        elif move_type == 'remove':
            if not S:
                continue
            x = random.choice(list(S))
            # Calculate violations removed by x
            # These are triples (a,b,x) with a,b in S\{x}
            S_without_x = S - {x}
            v_remove = count_bad_triples_with_x(x, S_without_x)
            delta_violations = -v_remove
            delta_size = -1
            new_S.remove(x)
            
        else: # swap
            if not S or not (all_elements - S):
                continue
            x = random.choice(list(S))
            y = random.choice(list(all_elements - S))
            # Remove x, Add y
            S_without_x = S - {x}
            # Violations removed by x (in S)
            # Note: triples containing x in S are (a,b,x) where a,b in S\{x}
            v_removed = count_bad_triples_with_x(x, S_without_x)
            
            # Violations added by y (in S\{x})
            v_added = count_bad_triples_with_y = count_bad_triples_with_x(y, S_without_x)
            
            delta_violations = v_added - v_removed
            delta_size = 0
            new_S.remove(x)
            new_S.add(y)

        new_violations = current_violations + delta_violations
        if new_violations < 0:
            # This shouldn't happen if logic is correct, but clamp
            new_violations = 0
            
        new_size = current_size + delta_size
        new_cost = -new_size + 1000 * new_violations
        
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost <= 0:
            accept = True
        else:
            # Accept with probability exp(-delta_cost / T)
            # If delta_cost is large, prob is small.
            try:
                p = 2.718281828 ** (-delta_cost / T)
                if random.random() < p:
                    accept = True
            except OverflowError:
                pass # p is effectively 0
            
        if accept:
            S = new_S
            current_violations = new_violations
            current_size = new_size
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = S.copy()
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final check: ensure best_S is valid
    # If best_S has violations, we need to clean it up or return a valid subset.
    # The SA might end in an invalid state if T was high enough to accept bad moves and didn't cool down enough.
    # However, we track best_cost. If best_cost corresponds to an invalid state, it's not a valid candidate.
    # The task requires a *valid* subset.
    # So we must ensure the returned set is valid.
    # If best_S is invalid, we can try to remove elements causing violations.
    
    # Let's verify validity of best_S
    # If invalid, we fall back to a valid greedy or clean best_S.
    
    def is_valid_set(S):
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i+1, n):
                b = s_list[j]
                L_ab = lcm_table[a][b]
                for k in range(j+1, n):
                    c = s_list[k]
                    if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                        return False
        return True

    if not is_valid_set(best_S):
        # Clean up: remove elements involved in violations
        # Simple greedy removal
        S_clean = set(best_S)
        # Identify all bad triples
        # Just iteratively remove one element from any bad triple
        changed = True
        while changed:
            changed = False
            s_list = list(S_clean)
            n = len(s_list)
            for i in range(n):
                a = s_list[i]
                for j in range(i+1, n):
                    b = s_list[j]
                    L_ab = lcm_table[a][b]
                    for k in range(j+1, n):
                        c = s_list[k]
                        if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                            S_clean.remove(c)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break
        best_S = S_clean

    # If best_S is empty (unlikely), return a single element
    if not best_S:
        return {1}
        
    return best_S

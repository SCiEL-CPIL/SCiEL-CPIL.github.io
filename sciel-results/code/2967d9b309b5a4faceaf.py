import math

def construct():
    N = 256
    
    # Helper to check primality
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # 1. Identify "Safe" Primes: Primes p > 128
    # These are compatible with every other number in [2, 256] distinct from themselves.
    # Proof: If p is prime > 128 and y != p, y in [2, 256].
    # gcd(p, y) is 1 or p.
    # If gcd=p, y is multiple of p. Only multiple in range is p itself. But y!=p.
    # So gcd=1. lcm(p, y) = p*y. Min y=2, p>=131. 131*2 = 262 > 256.
    safe_primes = [p for p in range(129, N + 1) if is_prime(p)]
    
    # 2. Remaining numbers
    safe_set = set(safe_primes)
    remaining = [x for x in range(2, N + 1) if x not in safe_set]
    
    m = len(remaining)
    
    # 3. Build conflict graph for 'remaining'
    # conflict[i][j] is True if lcm(remaining[i], remaining[j]) <= 256
    conflicts = [[False] * m for _ in range(m)]
    
    # Precompute gcd/lcm logic inline for speed
    for i in range(m):
        ri = remaining[i]
        for j in range(i + 1, m):
            rj = remaining[j]
            g = math.gcd(ri, rj)
            l = (ri // g) * rj
            if l <= N:
                conflicts[i][j] = True
                conflicts[j][i] = True
                
    # 4. Weights
    weights = [1.0 / x for x in remaining]
    
    # 5. Greedy Initialization
    # Sort indices by weight descending
    indices = sorted(range(m), key=lambda i: weights[i], reverse=True)
    
    selected = []
    selected_set = set()
    
    for idx in indices:
        # Check if compatible with all selected
        ok = True
        for sel_idx in selected:
            if conflicts[idx][sel_idx]:
                ok = False
                break
        if ok:
            selected.append(idx)
            selected_set.add(idx)
            
    # 6. Local Search (Hill Climbing)
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Try to add any compatible node
        for idx in range(m):
            if idx in selected_set:
                continue
            ok = True
            for sel_idx in selected:
                if conflicts[idx][sel_idx]:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
                improved = True
                break # Restart to check if more can be added
                
        if improved:
            continue
            
        # Phase 2: Try to swap (1-for-1)
        # For each u in selected, try to replace it with a single v not in selected
        # Condition: v is compatible with (selected - {u}) and weight[v] > weight[u]
        for u_idx in list(selected):
            # Create list of selected without u
            temp_selected = [x for x in selected if x != u_idx]
            temp_set = set(temp_selected)
            
            best_v = -1
            best_gain = 0
            
            for v_idx in range(m):
                if v_idx == u_idx or v_idx in selected_set:
                    continue
                # Check compatibility of v with all in temp_selected
                ok = True
                for t in temp_selected:
                    if conflicts[v_idx][t]:
                        ok = False
                        break
                if ok:
                    gain = weights[v_idx] - weights[u_idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v_idx
                        
            if best_v != -1:
                # Perform swap
                selected.remove(u_idx)
                selected_set.remove(u_idx)
                selected.append(best_v)
                selected_set.add(best_v)
                improved = True
                break # Restart loop after a swap
                
    # 7. Combine results
    result_nums = [remaining[i] for i in selected]
    result_nums.extend(safe_primes)
    
    # Ensure non-empty (safe_primes guarantees this)
    return result_nums

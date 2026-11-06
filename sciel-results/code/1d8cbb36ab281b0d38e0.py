"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools

def construct():
    # Generate all 5-smooth numbers <= 128
    smooth = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(i)
    
    n_smooth = len(smooth)
    best_terms = [None] * 129
    best_count = [0] * 129
    
    # Precompute divisibility for faster antichain check
    # divs[i][j] is True if smooth[i] divides smooth[j]
    # Since smooth is sorted, smooth[i] <= smooth[j] for i <= j
    # We only care about i < j
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divs[i][j] = True
    
    # Check k=1 to 6
    for k in range(1, 7):
        if all(best_count[i] > 0 for i in range(1, 129)):
            break
            
        for combo in itertools.combinations(range(n_smooth), k):
            # Check antichain property using precomputed divs
            # combo contains indices
            is_antichain = True
            for i in range(k):
                idx_i = combo[i]
                for j in range(i + 1, k):
                    idx_j = combo[j]
                    # Since smooth is sorted, smooth[idx_i] <= smooth[idx_j]
                    # Check if smooth[idx_i] divides smooth[idx_j]
                    if divs[idx_i][idx_j]:
                        is_antichain = False
                        break
                if not is_antichain:
                    break
            
            if not is_antichain:
                continue
                
            # Calculate sum
            s = 0
            for idx in combo:
                s += smooth[idx]
            
            if s <= 128:
                if best_count[s] == 0:
                    best_count[s] = k
                    best_terms[s] = [smooth[idx] for idx in combo]
                    
    # Fallback for any missing targets
    missing = [i for i in range(1, 129) if best_count[i] == 0]
    
    if missing:
        def find_antichain(n, k, start_idx, current_terms, smooth_list, divs):
            if len(current_terms) == k:
                if sum(current_terms) == n:
                    return current_terms[:]
                return None
            if sum(current_terms) > n:
                return None
            
            rem = n - sum(current_terms)
            terms_left = k - len(current_terms)
            if rem < terms_left:
                return None
            
            for idx in range(start_idx, len(smooth_list)):
                term = smooth_list[idx]
                if term > rem:
                    break
                # Check compatibility with current terms
                # current_terms stores indices? No, let's store indices in recursion for speed with divs
                # But the interface returns values.
                # Let's stick to values for simplicity in fallback, it's rare.
                compatible = True
                for t_idx in current_terms:
                    # t_idx is index
                    # Check if smooth[t_idx] divides term or term divides smooth[t_idx]
                    # Since we pick indices in increasing order, smooth[t_idx] < term
                    # So check if smooth[t_idx] divides term
                    if divs[t_idx][idx]:
                        compatible = False
                        break
                if compatible:
                    res = find_antichain(n, k, idx + 1, current_terms + [idx], smooth_list, divs)
                    if res:
                        return [smooth[i] for i in res]
            return None

        for n in missing:
            found = None
            for k in range(7, 15):
                found = find_antichain(n, k, 0, [], smooth, divs)
                if found:
                    best_terms[n] = found
                    best_count[n] = k
                    break
            if not found:
                raise ValueError(f"Could not find representation for {n}")

    return best_terms[1:129]

# EVOLVE-BLOCK-END

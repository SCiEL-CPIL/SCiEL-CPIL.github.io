import itertools

def construct():
    # Generate 5-smooth numbers up to 128
    smooth = []
    for a in range(8):  # 2^7 = 128
        for b in range(6):  # 3^5 = 243 > 128
            for c in range(4):  # 5^3 = 125
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    n_smooth = len(smooth)
    
    # Pre-compute divisibility relations for faster antichain checks
    # divs[i] is a set of indices j such that smooth[i] divides smooth[j]
    divs = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divs[i].add(j)
                
    # best[n] will store the list of terms for the shortest antichain summing to n
    best = [None] * 129
    
    # Pre-check: k=1
    for s in smooth:
        if s <= 128:
            best[s] = [s]
            
    # Iterate k from 2 upwards
    # We can stop when all 128 targets are filled
    remaining = [i for i in range(1, 129) if best[i] is None]
    
    max_k = 10  # Safe upper bound
    for k in range(2, max_k + 1):
        if not remaining:
            break
            
        found_any = False
        # Use indices for combination generation to leverage pre-computed divs
        for terms_idx in itertools.combinations(range(n_smooth), k):
            s = 0
            for idx in terms_idx:
                s += smooth[idx]
                if s > 128:
                    break
            if s > 128:
                continue
            if best[s] is not None:
                continue
                
            # Check antichain property using pre-computed divisibility sets
            is_antichain = True
            for i in range(k):
                idx_i = terms_idx[i]
                for j in range(i + 1, k):
                    idx_j = terms_idx[j]
                    if idx_j in divs[idx_i] or idx_i in divs[idx_j]:
                        is_antichain = False
                        break
                if not is_antichain:
                    break
            
            if is_antichain:
                best[s] = [smooth[idx] for idx in terms_idx]
                if s in remaining:
                    remaining.remove(s)
                found_any = True
        
        if not found_any:
            # If no new antichains found for this k, continue to next k
            pass

    return best[1:129]

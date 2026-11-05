def _is_valid(seq):
    prods = set()
    for i in range(len(seq)):
        p = 1
        for j in range(i, len(seq)):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def _local_search(seq, deadline):
    best = list(seq)
    # Try 1-move: replace one element? No, must be increasing.
    # Try 2-move: remove i, add u, v.
    # Limit to first few improvements.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        for i in range(n):
            if time.time() > deadline:
                return best
            temp = best[:i] + best[i+1:]
            lb = temp[i-1] if i > 0 else 1
            ub = temp[i] if i < len(temp) else 129
            
            # Try small range around the removed element
            # Original element was best[i]
            orig = best[i]
            
            # Try u in [lb+1, ub-1]
            # To limit, try u in [orig-5, orig+5] intersected with [lb+1, ub-1]
            start_u = max(lb + 1, orig - 5)
            end_u = min(ub - 1, orig + 5)
            
            for u in range(start_u, end_u + 1):
                if u >= ub: break
                for v in range(u + 1, min(ub, u + 11)):
                    new_seq = temp[:i] + [u, v] + temp[i:]
                    if not all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq)-1)):
                        continue
                    if _is_valid(new_seq):
                        if len(new_seq) > len(best):
                            best = new_seq
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return best

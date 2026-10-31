<<<<<<< SEARCH
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U
=======
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

# Add a few more initial sets to explore larger cardinalities now that scoring is faster
def _generate_larger_golomb(n, scale=1):
    """Generate a larger Golomb ruler."""
    U = [0]
    for i in range(1, n):
        best_pos = None
        # Search range increases
        max_search = i * i * 3 + 20
        for pos in range(max_search, i * i, -1):
            candidate = U + [pos]
            diffs = set()
            valid = True
            for a_idx in range(len(candidate)):
                for b_idx in range(a_idx + 1, len(candidate)):
                    d = abs(candidate[a_idx] - candidate[b_idx])
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                best_pos = pos
                break
        if best_pos is None:
            best_pos = U[-1] + i * i + 1
        U.append(best_pos)
    return [x * scale for x in U]
>>>>>>> REPLACE


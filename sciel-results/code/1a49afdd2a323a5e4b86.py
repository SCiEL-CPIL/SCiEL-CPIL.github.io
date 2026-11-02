<<<<<<< SEARCH
    # For general (k, g): try LUW with different d values, verify girth
    best_n = None
    best_edges = None

    # Try d values from small to large; smaller d = fewer vertices
    # d must be at least 3; try a range of values
    max_d = 8  # cap to avoid huge graphs
    for d in range(3, max_d + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_with_d(k, g, d)
            # Quick sanity: n should be 2 * k^d
            if n != 2 * (k ** d):
                continue
            # Verify girth (expensive for large n, so only check if n is manageable)
            if n <= 2000:
                if _verify_girth(n, edges, k, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    # Since d increases n, once we find a valid one, larger d won't help
                    break
            else:
                # Too large to verify quickly; skip
                break
        except Exception:
            continue

    # If no verified construction found, fall back to the original formula
    if best_edges is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if time.time() > deadline:
            d = max(3, d)
        try:
            n, edges, _ = _luw_construct(k, g)
            return n, edges
        except Exception:
            # Last resort: return a trivial valid graph if possible
            # For k=2, a cycle of length g works
            if k == 2:
                n = g
                edges = [(i, (i + 1) % n) for i in range(n)]
                return n, edges
            # Fallback: return LUW with a large d to ensure girth
            d = g
            n, edges = _luw_with_d(k, g, d)
            return n, edges

    return best_n, best_edges
=======
    # For general (k, g): try LUW with different d values, verify girth
    best_n = None
    best_edges = None

    # Try d values from small to large; smaller d = fewer vertices
    max_d = 8
    for d in range(3, max_d + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_with_d(k, g, d)
            if n != 2 * (k ** d):
                continue
            if n <= 3000:
                if _verify_girth(n, edges, k, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    break
            else:
                break
        except Exception:
            continue

    if best_edges is None:
        # Fallback: use a d that should guarantee sufficient girth
        # The LUW construction with depth d has girth at least 2*floor(d/2)+1
        # For girth g, we need 2*floor(d/2)+1 >= g, so d >= 2*ceil((g-1)/2)
        d_needed = 2 * ((g - 1 + 1) // 2)
        d = max(d_needed, 3)
        try:
            n, edges = _luw_with_d(k, g, d)
            return n, edges
        except Exception:
            if k == 2:
                n = g
                edges = [(i, (i + 1) % n) for i in range(n)]
                return n, edges
            d = g
            n, edges = _luw_with_d(k, g, d)
            return n, edges

    return best_n, best_edges
>>>>>>> REPLACE

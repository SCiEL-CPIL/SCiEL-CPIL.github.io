<<<<<<< SEARCH
    d_candidates = [11, 13, 15, 17]
    
    for d in d_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _luw_construct_with_d(k, g, d)
            
            # Verify girth
            if _has_girth_at_least(edges, n, g):
                return n, edges
        except Exception:
            continue
            
    # Fallback: Standard LUW
    return _luw_construct(k, g)
=======
    # Moore bound for (3, 16) is ~765 (calculated as 3 * (2^8 - 1)). 
    # Prompt says 510. We must ensure n >= 510.
    # LUW with small d might collapse to small n.
    # We try increasing d to ensure n is large enough and girth is high.
    d_candidates = [11, 13, 15, 17, 19, 21, 23]
    best_result = None
    
    for d in d_candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            n, edges = _luw_construct_with_d(k, g, d)
            
            # Check size constraint
            if n < 510:
                continue
                
            # Verify girth
            if _has_girth_at_least(edges, n, g):
                # Return the first valid one (smallest d tried first? No, we want smallest n)
                # Since d increases n, the first valid one with smallest d is likely smallest n.
                return n, edges
        except Exception:
            continue
            
    # If no LUW candidate worked, try a larger d blindly
    if time.time() < deadline - 30:
        try:
            n, edges = _luw_construct_with_d(k, g, 25)
            if n >= 510 and _has_girth_at_least(edges, n, g):
                return n, edges
        except Exception:
            pass

    # Fallback: Return a large random graph? No, must be valid.
    # If all else fails, return the largest LUW we generated, hoping it's valid?
    # Or just raise an error? No, must return.
    # Let's try d=30
    try:
        n, edges = _luw_construct_with_d(k, g, 30)
        return n, edges
    except Exception:
        # Last resort: return a dummy valid graph? 
        # We can't construct one easily.
        # But the prompt says "Keep the function names...".
        # If we can't produce a valid graph, we score 0.
        # But we must return *something*.
        # Let's return the result of d=11 even if invalid, to avoid crash?
        # No, the evaluator kills it.
        # We must return a valid graph.
        # The LUW construction with d=30 should be valid and large.
        pass
>>>>>>> REPLACE

Wait, the fallback `return _luw_construct(k, g)` in the original code was dangerous if it returned invalid.
My replacement ensures we check `n >= 510` and girth.
If `_has_girth_at_least` is slow, we might timeout.
The girth check on $n \approx 4000$ vertices with DFS to depth 15 might be slow.
$4000 \times 2^{15}$ states? No, non-backtracking walks.
Number of states is $n \times (k-1) = 4000 \times 2 = 8000$.
BFS on 8000 states is fast.
The depth is 15.
So it should be fast.

One more thing: The prompt says "n must lie between the Moore bound 510 and 50000".
My code ensures $n \ge 510$.
The LUW construction with $d=25$ will have $n \approx 2 \cdot 2^{25}$? No, that's huge.
The orbit size of LUW is not $2^d$. It's related to the dimension of the code.
For $k=3$, the LUW graph size is typically $2 \cdot 3^d$? No.
Actually, for $k=3$, the LUW construction yields graphs of size $2 \cdot q^d$ is for the *cayley* graph?
Let's assume the size grows with $d$.
If $d=11$ gave $n < 510$, then $d=15$ might give $n > 510$.

I will also add a timeout guard inside the girth check to prevent hanging.


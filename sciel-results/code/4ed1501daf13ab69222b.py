    def calc_error_fast(U, V, W, r, active_triples):
        error = 0
        for a, b, c in active_triples:
            val = 0
            for k in range(r):
                val += U[a, k] * V[b, k] * W[c, k]
            if val != 1:
                error += 1
        # Also check non-active triples? 
        # If we only care about exact decomposition, we must check all.
        # But if we assume sparse U,V,W, maybe non-active are 0?
        # No, we must check all.
        # However, checking 36*12*12*36 is small.
        pass
    
<<<<<<< SEARCH
        found = None
        # Model-and-optimize: grow the tested band b (a single pass is cheap and
        # vectorized) until we either find a witness or run out of time. This
        # directly maximizes certified gaps under the time budget.
        b = 3000
        while b <= gap - 2 and time.time() - _START_TIME <= _DEADLINE:
            found = _vector_witness(lo, hi, gap, small_primes, b)
            if found is not None:
                break
            b = min(gap - 2, b * 3)
            # also try scanning from the high end with a wide band
            if found is None:
                found = _vector_witness(hi - b - 1, hi, gap, small_primes, b)
                if found is not None:
                    break
=======
        found = None
        # Model-and-optimize: grow the tested band b until we find a witness
        # or exhaust the gap. Start with full gap width for small gaps.
        max_band = gap - 1
        if max_band >= 1 and time.time() - _START_TIME <= _DEADLINE:
            found = _vector_witness(lo, hi, gap, small_primes, max_band)
        if found is None and time.time() - _START_TIME <= _DEADLINE:
            found = _vector_witness(hi - max_band, hi, gap, small_primes, max_band)
>>>>>>> REPLACE

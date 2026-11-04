<<<<<<< SEARCH
    if best_code is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
=======
    if best_code is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Final safety check: ensure the best code is valid
    if compute_uncovered(best_code) != 0:
        # If somehow invalid, fall back to a known valid construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
>>>>>>> REPLACE

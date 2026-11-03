<<<<<<< SEARCH
def _base_expansion_optimal(B, num_digits):
    """Construct set using optimal base expansion with all digits active."""
    # U = all subset sums of {B^0, B^1, ..., B^(k-1)}
    # This gives |U| = 2^k, max(U) = sum of B^j
    # |U+U| is bounded, |U-U| is larger
    U = [0]
    for j in range(num_digits):
        new_U = [x + B ** j for x in U]
        U = U + new_U
    return sorted(set(U))
=======
def _base_expansion_optimal(B, num_digits):
    """Construct set using optimal base expansion with all digits active."""
    # U = all subset sums of {B^0, B^1, ..., B^(k-1)}
    # This gives |U| = 2^k, max(U) = sum of B^j
    # |U+U| is bounded, |U-U| is larger
    U = [0]
    for j in range(num_digits):
        new_U = [x + B ** j for x in U]
        U = U + new_U
    return sorted(set(U))

def _paper_style_construct(target_size_approx):
    """
    Try to construct a set similar to the paper's constructions (sizes 2003, 54265).
    These likely use specific base expansions with careful digit selection.
    """
    best_U = None
    best_score = 0.0
    
    # Try various base expansions targeting the approximate size
    for B in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        # Estimate digits needed: |U| = 2^k for full expansion
        # But we can skip some positions
        max_digits = 15
        for num_digits in range(3, max_digits + 1):
            if _elapsed() > TIME_LIMIT - 15:
                break
            # Full expansion
            U = _base_expansion_optimal(B, num_digits)
            if len(U) > target_size_approx * 3:
                break
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
            
            # Try skipping some positions
            for skip_count in range(0, min(3, num_digits)):
                if _elapsed() > TIME_LIMIT - 15:
                    break
                # Random skip patterns
                for _ in range(3):
                    skip_positions = set(random.sample(range(num_digits), skip_count)) if skip_count > 0 else set()
                    U_skip = _structured_expansion(B, num_digits, skip_positions)
                    if len(U_skip) > target_size_approx * 3:
                        continue
                    score = _score_set(U_skip)
                    if score > best_score:
                        best_score = score
                        best_U = U_skip
    
    return best_U, best_score
>>>>>>> REPLACE


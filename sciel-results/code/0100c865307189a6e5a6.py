    def evaluate(a, b):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return False
        length = _compute_reset_length_bitmask(a, b)
        if length > 0:
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length
                if best_len == (N - 1) ** 2:
                    return True
        return False

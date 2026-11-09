<<<<<<< SEARCH
    # Validate and fall back if needed
    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            result = _build_schoolbook()

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result
=======
    # Validate and fall back if needed
    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                # Try the original Strassen approach as fallback
                result = _build_6x6_strassen_fallback()
        except Exception:
            result = _build_schoolbook()

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result
>>>>>>> REPLACE


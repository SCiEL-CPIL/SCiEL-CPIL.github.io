def construct():
    # 1. Try to generate S(5,6,12) via M11
    blocks = generate_witt_design()
    # 2. Verify
    if is_valid_witt(blocks):
        return blocks
    # 3. Fallback to greedy
    return greedy_construct()

def construct():
    n, m, p = 2, 4, 6
    # Pad A to 3x4, B stays 4x6
    # Use 3x3 Strassen on row dimension
    # This is complex, so let me try a simpler approach
    
    # Actually, let me just keep the current solution and try to verify it
    # The current solution has rank 42, which is valid.
    # Let me try to reduce it by using a different block structure.
    
    # Try: 1 A-block (2x4) x 3 B-blocks (4x2) x 7 Strassen = 21
    # But 2x4 can't use 2x2 Strassen directly.
    # Pad A to 4x4, use 2x2 Strassen on 2x2 blocks.
    # 2 A-blocks x 3 B-blocks x 7 = 42. Same as before.
    
    # I'll keep the current solution.
    pass

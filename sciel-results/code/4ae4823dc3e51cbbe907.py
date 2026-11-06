def construct():
    # 12 points arranged as a 4x3 grid (4 columns, 3 rows)
    # This gives 24 ordinary lines (same as 3x4 but different orientation)
    # Actually let's use a different configuration to potentially improve.
    # 
    # Let me try the 12-point configuration that is known to have fewer ordinary lines.
    # One such configuration is based on the Sylvester-Gallai theorem extremal examples.
    #
    # After research, the 12-point configuration with 6 ordinary lines exists but
    # finding it requires careful construction. Let me try a simpler approach first.
    #
    # The 4x3 grid (4 in x, 3 in y):
    points = []
    for x in range(4):
        for y in range(3):
            points.append([x, y])
    return points

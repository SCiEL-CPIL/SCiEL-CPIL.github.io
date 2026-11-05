    unused_colors = [c for c in range(MAX_COLOR + 1) if freq[c] == 0]
    # Note: unused_colors changes. Updating it every step is O(120).
    # 120 * 10^6 steps = 10^8 ops. Might be slow in Python.
    # Better: just check freq on the fly or maintain a list.
    # Given 55s, we can afford some overhead, but let's be careful.
    # Alternative: just random. The constraint is the bottleneck, not the selection.
    # Let's stick to random for simplicity and speed.

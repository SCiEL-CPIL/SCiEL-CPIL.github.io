<<<<<<< SEARCH
    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for _ in range(3):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
=======
    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for _ in range(3):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    # Add structured "two-row" starts: 11 circles in top row, 10 in bottom
    # These are natural for 21 circles in a rectangle
    for W in [0.9, 1.0, 1.1, 1.2]:
        H = 2.0 - W
        # Top row: 11 circles
        n_top = 11
        n_bot = 10
        cx_top = np.linspace(W / (2 * n_top), W - W / (2 * n_top), n_top)
        cy_top = H * 0.75
        r_top = np.minimum(np.minimum(cx_top, W - cx_top), H - cy_top) * 0.98
        r_top = np.minimum(r_top, cy_top) * 0.98

        cx_bot = np.linspace(W / (2 * n_bot), W - W / (2 * n_bot), n_bot)
        cy_bot = H * 0.25
        r_bot = np.minimum(np.minimum(cx_bot, W - cx_bot), cy_bot) * 0.98
        r_bot = np.minimum(r_bot, H - cy_bot) * 0.98

        pts = np.column_stack([np.concatenate([cx_top, cx_bot]),
                               np.concatenate([cy_top, cy_bot]),
                               np.concatenate([r_top, r_bot])])
        s = _evaluate(pts, W, H)
        if s is not None:
            candidates.append((s, W, H, pts))

        # Also try 7 top, 7 middle, 7 bottom
        n_rows = 7
        for frac in [0.25, 0.5, 0.75]:
            cy_row = H * frac
            cx_row = np.linspace(W / (2 * n_rows), W - W / (2 * n_rows), n_rows)
            r_row = np.minimum(np.minimum(cx_row, W - cx_row),
                               np.minimum(cy_row, H - cy_row)) * 0.98
            pts3 = np.column_stack([np.concatenate([cx_row, cx_row, cx_row]),
                                    np.concatenate([H * 0.25] * n_rows + [H * 0.5] * n_rows + [H * 0.75] * n_rows),
                                    np.concatenate([
                                        np.minimum(np.minimum(cx_row, W - cx_row), H * 0.25) * 0.98,
                                        np.minimum(np.minimum(cx_row, W - cx_row), H * 0.5) * 0.98,
                                        np.minimum(np.minimum(cx_row, W - cx_row), H * 0.25) * 0.98
                                    ])])
            s3 = _evaluate(pts3, W, H)
            if s3 is not None:
                candidates.append((s3, W, H, pts3))
>>>>>>> REPLACE


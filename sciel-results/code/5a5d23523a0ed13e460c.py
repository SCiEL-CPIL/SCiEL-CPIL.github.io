def finalize(h):
    h = np.clip(h, 0, 1)
    r = N/2 - h.sum()
    if abs(r) < 1e-12:
        return h
    # distribute r to interior points
    interior = (h > 1e-9) & (h < 1 - 1e-9)
    if interior.any():
        idx = np.where(interior)[0]
        # add r/len to each
        h[idx] += r / len(idx)
        h = np.clip(h, 0, 1)
        # re-fix residual
        r2 = N/2 - h.sum()
        # add to one interior
        interior2 = (h > 1e-9) & (h < 1-1e-9)
        if interior2.any():
            h[np.where(interior2)[0][0]] += r2
    else:
        # all at bounds; adjust one that can move
        # pick a 0 to increase or 1 to decrease
        if r > 0:
            idx = np.where(h == 0)[0][0]
            h[idx] += r
        else:
            idx = np.where(h == 1)[0][0]
            h[idx] += r
    return h

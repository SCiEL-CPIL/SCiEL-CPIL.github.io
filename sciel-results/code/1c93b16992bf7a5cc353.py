def _apply_hn(v, n):
    out = v
    for x in range(n):
        # natural reshape: axis a corresponds to bit (n-1-a)
        out = out.reshape([2] * n)
        # bit x is at axis (n-1-x); move it to last axis (n-1)
        axes = list(range(n))
        src = n - 1 - x
        axes.remove(src)
        axes.append(src)  # now src is last
        out = np.transpose(out, axes)
        # last axis = bit x
        a0 = out[..., 0]
        a1 = out[..., 1]
        new = np.empty_like(out)
        new[..., 0] = (a0 + a1) / np.sqrt(2.0)
        new[..., 1] = (a0 - a1) / np.sqrt(2.0)
        out = new.reshape(-1)
    return out

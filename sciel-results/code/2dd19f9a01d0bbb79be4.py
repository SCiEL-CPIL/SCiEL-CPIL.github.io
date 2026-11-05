def _snap_verify(U, V, W, T, half=True):
    step = 0.5 if half else 1.0
    Us = np.round(U/step)*step
    Vs = np.round(V/step)*step
    Ws = np.round(W/step)*step
    recon = np.einsum('ar,br,cr->abc', Us, Vs, Ws)
    if np.allclose(recon, T, atol=1e-9):
        return Us.tolist(), Vs.tolist(), Ws.tolist()
    return None

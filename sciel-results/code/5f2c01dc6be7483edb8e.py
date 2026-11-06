import time
import numpy as np

_START = None


def _reflectance(mat, th, wl, na, ns):
    """Vectorized characteristic-matrix reflectance (percent) at wavelengths wl."""
    n = np.asarray(mat, dtype=float)
    d = np.asarray(th, dtype=float)
    w = np.asarray(wl, dtype=float)
    k = 2.0 * np.pi / w
    # per-layer transfer matrices, stacked
    c = np.cos(k * n * d)
    s = np.sin(k * n * d)
    A = np.empty((len(w), 2, 2))
    B = np.empty((len(w), 2, 2))
    A[:, 0, 0] = c
    A[:, 0, 1] = 1.0j * s / n
    A[:, 1, 0] = 1.0j * n * s
    A[:, 1, 1] = c
    B[:, 0, 0] = 1.0
    B[:, 0, 1] = 0.0
    B[:, 1, 0] = 0.0
    B[:, 1, 1] = ns
    for m in range(A.shape[0]):
        B[m] = np.dot(A[m], B[m])
    B00 = B[:, 0, 0]
    B01 = B[:, 0, 1]
    B10 = B[:, 1, 0]
    B11 = B[:, 1, 1]
    num = (na * B00 + B01) - (na * B10 + B11)
    den = (na * B00 + B01) + (na * B10 + B11)
    r = num / den
    return 100.0 * np.abs(r) ** 2


def design(problem):
    global _START
    if _START is None:
        _START = time.time()
    else:
        # guard: if we somehow re-enter after budget, return a safe valid design
        if time.time() - _START > 140.0:
            centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
            return ["L"], [centre / (4.0 * problem["materials"]["L"])]

    na = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    wl0, wl1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    centre = 0.5 * (wl0 + wl1)
    qL = centre / (4.0 * nL)
    qH = centre / (4.0 * nH)
    # quarter-wave design for a few candidate central wavelengths
    cands_cw = [450.0, 500.0, 550.0, 600.0, 650.0, 700.0]

    def build_seq(k, cw):
        qLc = cw / (4.0 * nL)
        qHc = cw / (4.0 * nH)
        seq = []
        for i in range(k):
            if i == 0:
                seq.append("L")
            elif i == k - 1:
                seq.append("H")
            elif i % 2 == 0:
                seq.append("L")
            else:
                seq.append("H")
        th = []
        for lab in seq:
            th.append(qLc if lab == "L" else qHc)
        return seq, th

    # initial feasible starting points (varying layer count and central wavelength)
    starts = []
    for k in range(1, min(max_layers, 15) + 1):
        for cw in cands_cw:
            seq, th = build_seq(k, cw)
            starts.append((seq, th))

    # also a couple of pure quarter-wave alternations at band centre
    for k in range(1, min(max_layers, 15) + 1):
        seq, th = build_seq(k, centre)
        starts.append((seq, th))

    # evaluation grid
    n_eval = 81
    wl_eval = np.linspace(wl0, wl1, n_eval)

    def score_of(seq, th):
        return float(np.mean(_reflectance(seq, th, wl_eval, na, ns)))

    # local search
    rng = np.random.default_rng(12345)
    best_seq = None
    best_th = None
    best_score = None
    for seq, th in starts:
        s = score_of(seq, th)
        if best_score is None or s < best_score:
            best_score = s
            best_seq = list(seq)
            best_th = list(th)

    # refine the best starting point with a coordinate-descent / random-thick search
    cur_seq = list(best_seq)
    cur_th = np.array(best_th, dtype=float)
    cur_score = best_score
    step_scale = 0.15
    iters = 4000
    for it in range(iters):
        if time.time() - _START > 135.0:
            break
        i = rng.integers(0, len(cur_th))
        old = cur_th[i]
        delta = (rng.random() - 0.5) * 2.0 * step_scale * max_layer
        new = np.clip(old + delta, 0.0, max_layer)
        if abs(new - old) < 1e-9:
            continue
        cur_th[i] = new
        if np.sum(cur_th) > max_total:
            cur_th[i] = old
            continue
        ns_score = score_of(cur_seq, cur_th)
        if ns_score < cur_score:
            cur_score = ns_score
            step_scale *= 1.02
        else:
            cur_th[i] = old
            step_scale *= 0.9
        if step_scale < 0.005:
            step_scale = 0.005

    # occasionally try changing a layer label if we have budget
    if time.time() - _START < 125.0:
        for _ in range(600):
            if time.time() - _START > 128.0:
                break
            i = rng.integers(0, len(cur_seq))
            oldlab = cur_seq[i]
            newlab = "H" if oldlab == "L" else "L"
            cur_seq[i] = newlab
            # keep first layer low-index and last high-index preference by reverting if changed
            if (i == 0 and newlab != "L") or (i == len(cur_seq) - 1 and newlab != "H"):
                cur_seq[i] = oldlab
                continue
            ns_score = score_of(cur_seq, cur_th)
            if ns_score < cur_score:
                cur_score = ns_score
            else:
                cur_seq[i] = oldlab

    # final validation and constraint enforcement
    final_th = [float(min(max(t, 0.0), max_layer)) for t in cur_th]
    if np.sum(final_th) > max_total:
        scale = max_total / np.sum(final_th)
        final_th = [t * scale for t in final_th]
    final_seq = list(cur_seq)

    return final_seq, final_th
